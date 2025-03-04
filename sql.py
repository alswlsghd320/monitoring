import os
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal, Union, Type, Callable, Tuple
import json
import pandas as pd
import numpy as np
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END

# Azure OpenAI API 설정
os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-openai-api-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource-name.openai.azure.com"
# Azure OpenAI API 버전은 필수 파라미터입니다
AZURE_OPENAI_API_VERSION = "2023-05-15"  # 사용하는 API 버전에 맞게 수정하세요

# Azure OpenAI 모델 배포 이름 설정
AZURE_CHAT_DEPLOYMENT_NAME = "gpt-35-turbo"  # Azure에 배포한 GPT 모델 이름
AZURE_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002"  # Azure에 배포한 임베딩 모델 이름

# SQL Agent 상태 정의
class SQLAgentState(TypedDict):
    # 사용자 입력과 AI 대화 이력
    messages: Annotated[List[BaseMessage], "대화 메시지 이력"]
    
    # 데이터베이스 연결 및 도구 정보
    db_uri: Annotated[str, "데이터베이스 URI"]
    toolkit: Annotated[Optional[Any], "SQLDatabaseToolkit 인스턴스"]
    custom_tools: Annotated[Optional[List[BaseTool]], "커스텀 도구 목록"]
    
    # Few-shot examples 관련
    few_shot_store: Annotated[Optional[Any], "Few-shot 예제 저장소"]
    similar_examples: Annotated[Optional[List[Dict[str, str]]], "유사한 few-shot 예제 목록"]
    
    # 쿼리 및 결과 정보
    query: Annotated[Optional[str], "생성된 SQL 쿼리"]
    query_result: Annotated[Optional[str], "SQL 쿼리 실행 결과"]
    
    # 에러 및 분기 제어
    error: Annotated[Optional[str], "발생한 오류 메시지"]
    next: Annotated[str, "다음 실행할 노드"]

# Few-Shot 예제 클래스
class FewShotStore:
    """
    Few-shot 예제를 저장하고 유사도 기반으로 검색하는 클래스
    """
    def __init__(self, examples: List[Dict[str, str]] = None):
        # 임베딩 모델 초기화 (Azure OpenAI)
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_EMBEDDING_DEPLOYMENT_NAME,
            api_version=AZURE_OPENAI_API_VERSION
        )
        
        # 예제 초기화
        self.examples = examples or self._get_default_examples()
        
        # 예제 임베딩 계산
        self._compute_embeddings()
    
    def _get_default_examples(self) -> List[Dict[str, str]]:
        """
        기본 Few-shot 예제 목록 반환
        """
        return [
            {
                "question": "각 국가별 고객 수를 알려주세요.",
                "sql": "SELECT Country, COUNT(*) as CustomerCount FROM customers GROUP BY Country ORDER BY CustomerCount DESC"
            },
            {
                "question": "가장 많이 팔린 상위 5개 제품은 무엇인가요?",
                "sql": "SELECT products.ProductName, SUM(order_details.Quantity) as TotalSold FROM products JOIN order_details ON products.ProductID = order_details.ProductID GROUP BY products.ProductID ORDER BY TotalSold DESC LIMIT 5"
            },
            {
                "question": "2022년 월별 총 매출을 계산해주세요.",
                "sql": "SELECT strftime('%Y-%m', orders.OrderDate) as Month, SUM(order_details.UnitPrice * order_details.Quantity) as TotalRevenue FROM orders JOIN order_details ON orders.OrderID = order_details.OrderID WHERE orders.OrderDate LIKE '2022%' GROUP BY Month ORDER BY Month"
            },
            {
                "question": "각 직원별로 처리한 주문 건수를 계산해주세요.",
                "sql": "SELECT employees.FirstName || ' ' || employees.LastName as EmployeeName, COUNT(orders.OrderID) as OrderCount FROM employees LEFT JOIN orders ON employees.EmployeeID = orders.EmployeeID GROUP BY employees.EmployeeID ORDER BY OrderCount DESC"
            },
            {
                "question": "평균 주문 금액보다 높은 주문을 한 고객 목록을 보여주세요.",
                "sql": "SELECT customers.CustomerID, customers.CompanyName, AVG(order_details.UnitPrice * order_details.Quantity) as AvgOrderAmount FROM customers JOIN orders ON customers.CustomerID = orders.CustomerID JOIN order_details ON orders.OrderID = order_details.OrderID GROUP BY customers.CustomerID HAVING AvgOrderAmount > (SELECT AVG(UnitPrice * Quantity) FROM order_details) ORDER BY AvgOrderAmount DESC"
            },
            {
                "question": "어떤 카테고리의 제품이 가장 많이 팔렸나요?",
                "sql": "SELECT categories.CategoryName, SUM(order_details.Quantity) as TotalSold FROM categories JOIN products ON categories.CategoryID = products.CategoryID JOIN order_details ON products.ProductID = order_details.ProductID GROUP BY categories.CategoryID ORDER BY TotalSold DESC"
            },
            {
                "question": "지난 3개월간 가장 활발한 고객 10명은 누구인가요?",
                "sql": "SELECT customers.CustomerID, customers.CompanyName, COUNT(orders.OrderID) as OrderCount FROM customers JOIN orders ON customers.CustomerID = orders.CustomerID WHERE orders.OrderDate >= date('now', '-3 months') GROUP BY customers.CustomerID ORDER BY OrderCount DESC LIMIT 10"
            },
            {
                "question": "각 지역별 매출 비중을 백분율로 계산해주세요.",
                "sql": "WITH RegionalSales AS (SELECT customers.Region, SUM(order_details.UnitPrice * order_details.Quantity) as Revenue FROM customers JOIN orders ON customers.CustomerID = orders.CustomerID JOIN order_details ON orders.OrderID = order_details.OrderID GROUP BY customers.Region) SELECT Region, Revenue, (Revenue * 100.0 / (SELECT SUM(Revenue) FROM RegionalSales)) as PercentageOfTotal FROM RegionalSales ORDER BY PercentageOfTotal DESC"
            }
        ]
    
    def _compute_embeddings(self):
        """
        모든 예제 질문에 대한 임베딩 계산
        """
        # 질문 텍스트 추출
        questions = [example["question"] for example in self.examples]
        
        # 임베딩 계산
        self.question_embeddings = self.embeddings.embed_documents(questions)
    
    def find_similar_examples(self, query: str, top_k: int = 3, threshold: float = 0.7) -> List[Dict[str, str]]:
        """
        주어진 쿼리와 가장 유사한 few-shot 예제를 찾아 반환
        
        Args:
            query: 자연어 질문
            top_k: 반환할 최대 예제 수
            threshold: 유사도 임계값 (이 값 이상의 유사도를 가진 예제만 반환)
            
        Returns:
            유사한 예제 목록
        """
        # 쿼리 임베딩 계산
        query_embedding = self.embeddings.embed_query(query)
        
        # 모든 예제와의 유사도 계산
        similarities = []
        for i, example_embedding in enumerate(self.question_embeddings):
            # 코사인 유사도 계산
            similarity = self._cosine_similarity(query_embedding, example_embedding)
            similarities.append((i, similarity))
        
        # 유사도 기준으로 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 임계값을 넘는 상위 K개 예제 선택
        similar_examples = []
        for i, similarity in similarities[:top_k]:
            if similarity >= threshold:
                example = self.examples[i].copy()
                example["similarity"] = similarity
                similar_examples.append(example)
        
        return similar_examples
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        두 벡터 간의 코사인 유사도 계산
        """
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 커스텀 도구 클래스 및 함수들

# 쿼리 결과 행 수 확인 도구 입력 스키마
class CountRowsInput(BaseModel):
    query: str = Field(description="The SQL query to check the row count")

# 쿼리 실행 및 요약 도구 입력 스키마
class SmartQueryInput(BaseModel):
    query: str = Field(description="The SQL query to execute")
    top_k: int = Field(default=100, description="Maximum number of rows to return without summarization")

# 쿼리 결과 행 수를 확인하는 도구
@tool("count_query_rows", args_schema=CountRowsInput)
def count_query_rows(query: str, db: SQLDatabase) -> int:
    """
    Count the number of rows that would be returned by the given SQL query
    without actually fetching the full result.
    """
    # 원본 쿼리에서 COUNT(*)로 변환
    count_query = f"SELECT COUNT(*) as row_count FROM ({query}) as subquery"
    
    try:
        # 행 수 쿼리 실행
        with db.engine.connect() as connection:
            result = connection.execute(count_query)
            row_count = result.fetchone()[0]
            return row_count
    except Exception as e:
        raise ValueError(f"Error counting rows: {str(e)}")

# 데이터프레임 요약 함수
def summarize_dataframe(df: pd.DataFrame, llm) -> str:
    """
    대규모 데이터프레임을 요약하는 함수
    """
    # 기본 통계 계산
    stats = {}
    
    # 레코드 수
    num_records = len(df)
    stats["total_records"] = num_records
    
    # 컬럼별 통계
    column_stats = {}
    for col in df.columns:
        col_data = df[col]
        col_stats = {}
        
        # 데이터 타입에 따른 통계
        if pd.api.types.is_numeric_dtype(col_data):
            col_stats["min"] = col_data.min()
            col_stats["max"] = col_data.max()
            col_stats["mean"] = col_data.mean()
            col_stats["median"] = col_data.median()
        elif pd.api.types.is_string_dtype(col_data):
            # 고유값이 너무 많지 않으면 고유값 목록과 빈도 계산
            unique_values = col_data.nunique()
            if unique_values <= 10:  # 고유값이 10개 이하인 경우
                value_counts = col_data.value_counts().to_dict()
                col_stats["unique_values"] = unique_values
                col_stats["value_counts"] = value_counts
            else:
                col_stats["unique_values"] = unique_values
                top_values = col_data.value_counts().head(5).to_dict()
                col_stats["top_values"] = top_values
        
        column_stats[col] = col_stats
    
    stats["column_stats"] = column_stats
    
    # 샘플 데이터 (처음 5개 행)
    stats["sample_data"] = df.head(5).to_dict(orient="records")
    
    # LLM을 사용한 요약 생성
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 데이터 분석 전문가입니다. 제공된 데이터셋의 통계 정보를 바탕으로 명확하고 유익한 요약을 생성해 주세요.
요약에는 다음이 포함되어야 합니다:
1. 데이터셋의 전체 크기와 구조
2. 주요 컬럼의 분포와 통계 (최소값, 최대값, 평균, 중앙값 등)
3. 중요한 패턴이나 이상점
4. 특히 사용자 질문과 관련된 통계적 인사이트

요약은 구조화되고 읽기 쉬워야 하며, 데이터가 어떤 의미를 갖는지 설명해야 합니다.
"""),
        ("human", """
다음은 SQL 쿼리 결과에 대한 통계 정보입니다:
{stats}

이 통계 정보를 바탕으로 데이터셋에 대한 유익한 요약을 제공해 주세요.
""")
    ])
    
    # 통계 정보를 JSON 문자열로 변환
    stats_str = json.dumps(stats, indent=2, default=str)
    
    # LLM을 사용하여 요약 생성
    summary = llm.invoke(prompt.format(stats=stats_str)).content
    
    return summary

# 스마트 쿼리 실행 및 요약 도구
class SmartQuerySQLTool(BaseTool):
    """
    쿼리 결과 행 수를 확인하고, 필요한 경우 결과를 요약하는 도구
    """
    name: str = "smart_query_sql"
    description: str = "Executes an SQL query and summarizes results if they exceed a threshold"
    args_schema: Type[BaseModel] = SmartQueryInput
    
    def __init__(self, db: SQLDatabase, llm):
        self.db = db
        self.llm = llm
        super().__init__()
    
    def _run(self, query: str, top_k: int = 100) -> str:
        # 먼저 쿼리 결과의 행 수 확인
        try:
            row_count = count_query_rows(query, self.db)
        except Exception as e:
            return f"행 수 확인 중 오류 발생: {str(e)}"
        
        # 행 수가 top_k보다 적으면 일반 쿼리 실행
        if row_count <= top_k:
            try:
                # 일반 방식으로 쿼리 실행
                with self.db.engine.connect() as connection:
                    result = connection.execute(query)
                    rows = result.fetchall()
                    columns = result.keys()
                    
                    # 결과를 표 형식 문자열로 변환
                    table_result = self._format_results(rows, columns)
                    return f"쿼리 결과 (총 {row_count}행):\n{table_result}"
            except Exception as e:
                return f"쿼리 실행 중 오류 발생: {str(e)}"
        
        # 행 수가 top_k보다 많으면 요약 실행
        else:
            try:
                # DataFrame으로 가져오기
                df = pd.read_sql(query, self.db.engine)
                
                # 결과 요약
                summary = summarize_dataframe(df, self.llm)
                
                return f"""
쿼리 결과가 너무 많습니다 (총 {row_count}행, 제한 {top_k}행).
대신 결과에 대한 요약을 제공합니다:

{summary}

참고: 처음 5개 행의 샘플 데이터가 요약에 포함되어 있습니다.
"""
            except Exception as e:
                return f"결과 요약 중 오류 발생: {str(e)}"
    
    def _format_results(self, rows, columns) -> str:
        """
        SQL 쿼리 결과를 읽기 쉬운 표 형식으로 변환
        """
        if not rows:
            return "No results found."
        
        # 컬럼 이름 목록
        header = " | ".join(columns)
        separator = "-" * len(header)
        
        # 각 행 포맷팅
        formatted_rows = []
        for row in rows:
            formatted_row = " | ".join(str(item) for item in row)
            formatted_rows.append(formatted_row)
        
        # 전체 테이블 조합
        table = f"{header}\n{separator}\n" + "\n".join(formatted_rows)
        return table

# SQL Agent 그래프의 초기 상태 설정
def initialize_state(db_uri: str, user_query: str) -> SQLAgentState:
    """
    SQL Agent 상태를 초기화합니다.
    """
    return SQLAgentState(
        messages=[HumanMessage(content=user_query)],
        db_uri=db_uri,
        toolkit=None,
        custom_tools=None,
        few_shot_store=None,
        similar_examples=None,
        query=None,
        query_result=None,
        error=None,
        next="setup_toolkit"
    )

# 1. 툴킷 설정 노드
def setup_toolkit(state: SQLAgentState) -> SQLAgentState:
    """
    SQLDatabaseToolkit과 커스텀 도구를 설정하는 노드
    """
    try:
        # 데이터베이스 연결
        db = SQLDatabase.from_uri(state["db_uri"])
        
        # Azure OpenAI LLM 설정
        llm = AzureChatOpenAI(
            azure_deployment=AZURE_CHAT_DEPLOYMENT_NAME,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=0
        )
        
        # SQLDatabaseToolkit 생성
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        
        # 커스텀 도구 생성
        smart_query_tool = SmartQuerySQLTool(db=db, llm=llm)
        custom_tools = [smart_query_tool]
        
        # Few-shot 예제 저장소 초기화
        few_shot_store = FewShotStore()
        
        return {
            **state,
            "toolkit": toolkit,
            "custom_tools": custom_tools,
            "few_shot_store": few_shot_store,
            "next": "get_table_list"
        }
    
    except Exception as e:
        error_message = f"데이터베이스 연결 중 오류 발생: {str(e)}"
        return {
            **state,
            "error": error_message,
            "messages": state["messages"] + [AIMessage(content=error_message)],
            "next": "END"
        }

# 2. 테이블 목록 가져오기 노드
def get_table_list(state: SQLAgentState) -> SQLAgentState:
    """
    ListSQLDatabaseTool을 사용하여 테이블 목록을 가져오는 노드
    """
    try:
        # ListSQLDatabaseTool 가져오기
        list_tables_tool = state["toolkit"].get_tools()[0]  # ListSQLDatabaseTool이 첫 번째 도구
        
        # 테이블 목록 가져오기
        table_list = list_tables_tool.invoke({})
        
        # 테이블 목록 메시지 추가
        messages = state["messages"] + [
            FunctionMessage(
                name="list_tables",
                content=f"데이터베이스 테이블 목록:\n{table_list}"
            )
        ]
        
        return {
            **state,
            "messages": messages,
            "next": "get_table_info"
        }
    
    except Exception as e:
        error_message = f"테이블 목록 가져오기 중 오류 발생: {str(e)}"
        return {
            **state,
            "error": error_message,
            "messages": state["messages"] + [AIMessage(content=error_message)],
            "next": "END"
        }

# 3. 테이블 정보 가져오기 노드
def get_table_info(state: SQLAgentState) -> SQLAgentState:
    """
    InfoSQLDatabaseTool을 사용하여 테이블 정보를 가져오는 노드
    """
    try:
        # InfoSQLDatabaseTool 가져오기
        info_tool = state["toolkit"].get_tools()[1]  # InfoSQLDatabaseTool이 두 번째 도구
        
        # 테이블 목록 가져오기 (이전 단계에서 받은 결과)
        table_list_message = state["messages"][-1].content
        tables = [table.strip() for table in table_list_message.split("\n")[1].split(",")]
        
        # 각 테이블의 정보 가져오기
        table_info_messages = []
        for table in tables:
            table_info = info_tool.invoke({"table_name": table})
            table_info_messages.append(f"테이블: {table}\n{table_info}")
        
        # 테이블 정보 메시지 추가
        schema_message = "데이터베이스 스키마 정보:\n\n" + "\n\n".join(table_info_messages)
        
        messages = state["messages"] + [
            FunctionMessage(
                name="get_schema_info",
                content=schema_message
            )
        ]
        
        return {
            **state,
            "messages": messages,
            "next": "find_similar_examples"
        }
    
    except Exception as e:
        error_message = f"테이블 정보 가져오기 중 오류 발생: {str(e)}"
        return {
            **state,
            "error": error_message,
            "messages": state["messages"] + [AIMessage(content=error_message)],
            "next": "END"
        }

# 4. 유사한 예제 찾기 노드
def find_similar_examples(state: SQLAgentState) -> SQLAgentState:
    """
    사용자 질문과 유사한 few-shot 예제를 찾는 노드
    """
    try:
        # 사용자 질문 추출
        user_query = state["messages"][0].content
        
        # Few-shot 예제 저장소에서 유사한 예제 찾기
        similar_examples = state["few_shot_store"].find_similar_examples(
            query=user_query,
            top_k=3,
            threshold=0.7  # 유사도 70% 이상만 고려
        )
        
        # 유사한 예제가 있으면 로그 메시지 추가
        if similar_examples:
            examples_str = "\n\n".join([
                f"질문: {ex['question']}\nSQL: {ex['sql']}\n유사도: {ex['similarity']:.2f}"
                for ex in similar_examples
            ])
            
            messages = state["messages"] + [
                FunctionMessage(
                    name="find_similar_examples",
                    content=f"유사한 few-shot 예제를 찾았습니다:\n{examples_str}"
                )
            ]
        else:
            # 유사한 예제가 없으면 간단한 메시지만 추가
            messages = state["messages"] + [
                FunctionMessage(
                    name="find_similar_examples",
                    content="유사한 few-shot 예제를 찾지 못했습니다."
                )
            ]
        
        return {
            **state,
            "similar_examples": similar_examples,
            "messages": messages,
            "next": "generate_sql"
        }
    
    except Exception as e:
        error_message = f"유사한 예제 찾기 중 오류 발생: {str(e)}"
        return {
            **state,
            "error": error_message,
            "messages": state["messages"] + [AIMessage(content=error_message)],
            "next": "END"
        }

# 5. SQL 생성 노드
def generate_sql(state: SQLAgentState) -> SQLAgentState:
    """
    사용자 쿼리와 스키마 정보를 기반으로 SQL 쿼리를 생성하는 노드
    """
    try:
        # 사용자 질문 추출
        user_query = state["messages"][0].content
        
        # 테이블 정보 추출 (get_table_info에서 생성된 메시지)
        for message in reversed(state["messages"]):
            if isinstance(message, FunctionMessage) and message.name == "get_schema_info":
                table_info = message.content
                break
        
        # 유사한 예제가 있는지 확인
        similar_examples = state.get("similar_examples", [])
        
        # 기본 시스템 프롬프트
        system_prompt = """
당신은 데이터베이스 전문가입니다. 사용자의 질문을 분석하여 적절한 SQL 쿼리를 생성해야 합니다.
데이터베이스 스키마 정보가 제공되면, 그것을 기반으로 정확한 SQL 쿼리를 작성하세요.
쿼리 작성 시 고려 사항:
1. 테이블 이름과 컬럼 이름을 정확히 사용하세요
2. SQL 구문이 정확해야 합니다
3. 필요한 경우 JOIN을 사용하세요
4. 결과를 사용자가 요청한 방식으로 정렬하거나 그룹화하세요

SQL 쿼리만 출력하세요. 설명이나 주석을 포함하지 마세요.

중요: 행 수 제한을 절대 사용하지 마세요. LIMIT이나 TOP 구문을 포함하지 마세요!
소량의 데이터를 반환하더라도 구체적인 제한 구문을 사용하지 마세요.
"""
        
        # 유사한 예제가 있으면 프롬프트에 추가
        if similar_examples:
            examples_text = "\n\n".join([
                f"질문: {ex['question']}\nSQL: {ex['sql']}"
                for ex in similar_examples
            ])
            
            system_prompt += f"""

다음은 유사한 질문과 해당하는 SQL 쿼리의 예시입니다. 이를 참고하여 현재 질문에 적합한 SQL 쿼리를 생성하세요:

{examples_text}
"""
        
        # SQL 생성 프롬프트
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
            ("system", "{schema_info}")
        ])
        
        # Azure OpenAI LLM 사용
        llm = AzureChatOpenAI(
            azure_deployment=AZURE_CHAT_DEPLOYMENT_NAME,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=0
        )
        
        sql_query = llm.invoke(
            prompt.format(
                question=user_query, 
                schema_info=table_info
            )
        ).content
        
        # SQL 쿼리 로그 메시지 추가
        messages = state["messages"] + [
            FunctionMessage(
                name="generate_sql",
                content=f"생성된 SQL 쿼리: \n```sql\n{sql_query}\n```"
            )
        ]
        
        return {
            **state,
            "query": sql_query,
            "messages": messages,
            "next": "execute_smart_sql"
        }
    
    except Exception as e:
        error_message = f"SQL 쿼리 생성 중 오류 발생: {str(e)}"
        return {
            **state,
            "error": error_message,
            "messages": state["messages"] + [AIMessage(content=error_message)],
            "next": "END"
        }

# 6. 스마트 SQL 실행 노드
def execute_smart_sql(state: SQLAgentState) -> SQLAgentState:
    """
    SmartQuerySQLTool을 사용하여 SQL 쿼리를 실행하는 노드
    """
    try:
        # SQL 쿼리 추출
        sql_query = state["query"]
        
        # SmartQuerySQLTool 가져오기
        smart_query_tool = state["custom_tools"][0]
        
        # 스마트 쿼리 실행 (기본 top_k=100 사용)
        try:
            query_result = smart_query_tool.invoke({"query": sql_query, "top_k": 100})
            
            # 결과 메시지 추가
            messages = state["messages"] + [
                FunctionMessage(
                    name="execute_smart_sql",
                    content=f"SQL 쿼리 스마트 실행 결과:\n{query_result}"
                )
            ]
            
            return {
                **state,
                "query_result": query_result,
                "messages": messages,
                "next": "interpret_results" 
            }
            
        except Exception as e:
            # SQL 실행 오류
            error_message = f"SQL 쿼리 실행 중 오류 발생: {str(e)}"
            return {
                **state,
                "error": error_message,
                "messages": state["messages"] + [FunctionMessage(name="execute_smart_sql", content=error_message)],
                "next": "fix_sql"
            }
    
    except Exception as e:
        error_message = f"SQL 실행 도구 사용 중 오류 발생: {str(e)}"
        return {
            **state,
            "error": error_message,
            "messages": state["messages"] + [AIMessage(content=error_message)],
            "next": "END"
        }

# 7. SQL 수정 노드
def fix_sql(state: SQLAgentState) -> SQLAgentState:
    """
    오류가 발생한 SQL을 수정하는 노드
    """
    error_message = state["error"]
    original_query = state["query"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
SQL 쿼리 실행 중 오류가 발생했습니다. 오류 메시지를 분석하고 SQL 쿼리를 수정해 주세요.
수정된 SQL 쿼리만 출력하세요. 설명이나 주석을 포함하지 마세요.

중요: 행 수 제한을 절대 사용하지 마세요. LIMIT이나 TOP 구문을 포함하지 마세요!
"""),
        ("human", "원본 쿼리: {query}\n오류 메시지: {error}")
    ])
    
    # Azure OpenAI LLM 사용
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_CHAT_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0
    )
    
    fixed_query = llm.invoke(
        prompt.format(
            query=original_query,
            error=error_message
        )
    ).content
    
    messages = state["messages"] + [
        FunctionMessage(
            name="fix_sql",
            content=f"SQL 쿼리 수정: \n```sql\n{fixed_query}\n```"
        )
    ]
    
    return {
        **state,
        "query": fixed_query,
        "messages": messages,
        "error": None,
        "next": "execute_smart_sql"
    }

# 8. 결과 해석 노드
def interpret_results(state: SQLAgentState) -> SQLAgentState:
    """
    SQL 쿼리 실행 결과를 해석하는 노드
    """
    user_query = state["messages"][0].content
    query = state["query"]
    query_result = state["query_result"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 데이터베이스 결과를 해석하는 전문가입니다. SQL 쿼리 결과를 분석하여 사용자의 원래 질문에 대한 답변을 제공해야 합니다.
결과를 명확하고 이해하기 쉬운 방식으로 설명하세요. 관련 통계나 인사이트를 포함할 수 있습니다.
테이블 형식 또는 글머리 기호를 사용하여 결과를 구조화하면 더 좋습니다.

결과에 '요약'이라는 단어가 포함되어 있으면, 결과가 많아서 요약되었음을 사용자에게 알리고,
요약 내용을 기반으로 인사이트를 제공하세요.
"""),
        ("human", "원래 질문: {question}\n\n실행된 쿼리: {query}\n\n쿼리 결과: {results}")
    ])
    
    # Azure OpenAI LLM 사용
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_CHAT_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.3
    )
    
    interpretation = llm.invoke(
        prompt.format(
            question=user_query,
            query=query,
            results=query_result
        )
    ).content
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=interpretation)],
        "next": "END"
    }

# SQL Agent 상태에 따른 다음 노드 라우팅 함수
def route_next(state: SQLAgentState) -> str:
    """
    현재 상태에 따라 다음 노드를 결정하는 라우터 함수
    """
    return state["next"]

# SQL Agent 그래프 생성 함수
def create_sql_agent_graph() -> StateGraph:
    """
    SQL Agent 그래프를 생성합니다.
    """
    # 상태 그래프 초기화
    workflow = StateGraph(SQLAgentState)
    
    # 노드 추가
    workflow.add_node("setup_toolkit", setup_toolkit)
    workflow.add_node("get_table_list", get_table_list)
    workflow.add_node("get_table_info", get_table_info)
    workflow.add_node("find_similar_examples", find_similar_examples)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_smart_sql", execute_smart_sql)
    workflow.add_node("interpret_results", interpret_results)
    workflow.add_node("fix_sql", fix_sql)
    
    # 조건부 엣지 추가
    workflow.add_conditional_edges(
        "setup_toolkit",
        route_next,
        {
            "get_table_list": "get_table_list",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "get_table_list",
        route_next,
        {
            "get_table_info": "get_table_info",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "get_table_info",
        route_next,
        {
            "find_similar_examples": "find_similar_examples",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "find_similar_examples",
        route_next,
        {
            "generate_sql": "generate_sql",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "generate_sql",
        route_next,
        {
            "execute_smart_sql": "execute_smart_sql",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "execute_smart_sql",
        route_next,
        {
            "interpret_results": "interpret_results",
            "fix_sql": "fix_sql",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "fix_sql",
        route_next,
        {
            "execute_smart_sql": "execute_smart_sql",
            "END": END
        }
    )
    
    workflow.add_conditional_edges(
        "interpret_results",
        route_next,
        {
            "END": END
        }
    )
    
    # 시작 노드 설정
    workflow.set_entry_point("setup_toolkit")
    
    # 그래프 컴파일
    return workflow.compile()

# SQL Agent 사용 함수
def query_database(db_uri: str, user_query: str) -> List[BaseMessage]:
    """
    사용자 질문을 받아 데이터베이스를 쿼리하고 결과를 반환하는 함수
    """
    # 상태 초기화
    initial_state = initialize_state(db_uri, user_query)
    
    # 그래프 생성
    graph = create_sql_agent_graph()
    
    # 그래프 실행
    result = graph.invoke(initial_state)
    
    # 결과 메시지 반환
    return result["messages"]

# Few-shot 예제 관리 함수들
def add_example(few_shot_store: FewShotStore, question: str, sql: str) -> None:
    """
    Few-shot 예제 저장소에 새 예제를 추가하는 함수
    """
    few_shot_store.examples.append({
        "question": question,
        "sql": sql
    })
    # 임베딩 다시 계산
    few_shot_store._compute_embeddings()

def remove_example(few_shot_store: FewShotStore, index: int) -> None:
    """
    Few-shot 예제 저장소에서 예제를 제거하는 함수
    """
    if 0 <= index < len(few_shot_store.examples):
        few_shot_store.examples.pop(index)
        # 임베딩 다시 계산
        few_shot_store._compute_embeddings()

# 사용 예시
if __name__ == "__main__":
    # 데이터베이스 URI (chinook.db는 SQLite 샘플 데이터베이스)
    db_uri = "sqlite:///chinook.db"
    
    # 사용자 질문
    user_query = "모든 고객의 구매 기록을 분석하고 평균 구매 금액과 구매 빈도를 알려주세요."
    
    # 쿼리 실행
    messages = query_database(db_uri, user_query)
    
    # 최종 결과 출력
    for i, message in enumerate(messages):
        if i == 0:  # 첫 번째 메시지는 사용자 질문
            print(f"질문: {message.content}\n")
        elif isinstance(message, AIMessage):
            print(f"\n답변: {message.content}")
        elif isinstance(message, FunctionMessage):
            print(f"\n[{message.name}] {message.content}")
