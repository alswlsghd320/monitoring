import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Any

def summarize_dataframe(df: pd.DataFrame, df_name: str = "data") -> Dict:
    """
    데이터프레임의 종합적인 요약 정보를 생성합니다.
    값 정보뿐만 아니라 해당 값이 있는 행들도 함께 반환합니다.
    
    Args:
        df: 요약할 데이터프레임
        df_name: 데이터프레임의 이름이나 식별자
        
    Returns:
        요약 정보를 담은 딕셔너리
    """
    # 기본 통계 정보
    basic_info = {
        "name": df_name,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "missing_rows": df[df.isnull().any(axis=1)].to_dict("records") if df.isnull().any().any() else []
    }
    
    # 시계열 데이터 확인 및 요약
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if date_cols:
        time_col = date_cols[0]  # 첫 번째 날짜 열 사용
        basic_info["time_range"] = {
            "start": df[time_col].min().strftime("%Y-%m-%d"),
            "end": df[time_col].max().strftime("%Y-%m-%d"),
            "periods": len(df[time_col].unique()),
            "start_rows": df[df[time_col] == df[time_col].min()].to_dict("records"),
            "end_rows": df[df[time_col] == df[time_col].max()].to_dict("records")
        }
    
    # 수치형 열에 대한 통계 계산
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_stats = {}
    
    if numeric_cols:
        for col in numeric_cols:
            # 기본 통계
            stats = {
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "q1": df[col].quantile(0.25),
                "q3": df[col].quantile(0.75)
            }
            
            # 관련 행 정보 추가
            stats["min_rows"] = df[df[col] == stats["min"]].to_dict("records")
            stats["max_rows"] = df[df[col] == stats["max"]].to_dict("records")
            stats["outliers"] = df[(df[col] < stats["q1"] - 1.5 * (stats["q3"] - stats["q1"])) | 
                                  (df[col] > stats["q3"] + 1.5 * (stats["q3"] - stats["q1"]))].to_dict("records")
            
            # 시계열이 있는 경우 트렌드 요약 추가
            if date_cols:
                try:
                    # 시계열 그룹화
                    time_col = date_cols[0]
                    grouped = df.groupby(time_col)[col].mean()
                    
                    # 최근 변화 추세 (최근 3개 기간)
                    periods = min(3, len(grouped))
                    if periods >= 2:
                        recent_trend = grouped.iloc[-periods:].pct_change().iloc[1:].mean() * 100
                        stats["recent_trend_pct"] = recent_trend
                        
                        # 트렌드와 관련된 행들
                        trend_period_start = grouped.index[-periods]
                        stats["trend_period_rows"] = df[df[time_col] >= trend_period_start].to_dict("records")
                except Exception as e:
                    stats["trend_error"] = str(e)
            
            numeric_stats[col] = stats
    
    # 범주형 열에 대한 통계 계산
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_stats = {}
    
    if categorical_cols:
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            top_n = min(5, len(value_counts))
            
            cat_stats = {
                "unique_count": len(value_counts),
                "top_values": value_counts.head(top_n).to_dict(),
                "top_percentage": (value_counts.head(top_n) / len(df) * 100).to_dict()
            }
            
            # 각 카테고리 값별 행 정보
            cat_stats["category_rows"] = {}
            for category in value_counts.index[:top_n]:
                cat_stats["category_rows"][str(category)] = df[df[col] == category].to_dict("records")
            
            categorical_stats[col] = cat_stats
    
    # 매출 관련 특수 처리 (금액 관련 열 감지)
    sales_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                 ['revenue', 'sales', 'amount', 'price', 'income', '매출', '판매', '금액'])]
    
    sales_summary = {}
    if sales_cols:
        for col in sales_cols:
            if col in numeric_cols:
                # 매출 총액
                total = df[col].sum()
                sales_stats = {
                    "total": total,
                    "daily_avg": total / basic_info["time_range"]["periods"] if "time_range" in basic_info else None,
                    "top_sales_rows": df.nlargest(5, col).to_dict("records")
                }
                
                # 시계열이 있는 경우 매출 트렌드 추가
                if date_cols:
                    try:
                        # 기간별 매출 추이
                        time_col = date_cols[0]
                        periodic_sales = df.groupby(time_col)[col].sum()
                        
                        sales_stats["trend"] = {
                            "first_period": periodic_sales.iloc[0],
                            "last_period": periodic_sales.iloc[-1],
                            "growth_rate": ((periodic_sales.iloc[-1] / periodic_sales.iloc[0]) - 1) * 100 if periodic_sales.iloc[0] != 0 else None,
                            "first_period_rows": df[df[time_col] == periodic_sales.index[0]].to_dict("records"),
                            "last_period_rows": df[df[time_col] == periodic_sales.index[-1]].to_dict("records")
                        }
                        
                        # 매출이 가장 높은/낮은 기간
                        max_sales_period = periodic_sales.idxmax()
                        min_sales_period = periodic_sales.idxmin()
                        
                        sales_stats["best_period"] = {
                            "time": max_sales_period.strftime("%Y-%m-%d") if hasattr(max_sales_period, 'strftime') else str(max_sales_period),
                            "value": periodic_sales.max(),
                            "rows": df[df[time_col] == max_sales_period].to_dict("records")
                        }
                        
                        sales_stats["worst_period"] = {
                            "time": min_sales_period.strftime("%Y-%m-%d") if hasattr(min_sales_period, 'strftime') else str(min_sales_period),
                            "value": periodic_sales.min(),
                            "rows": df[df[time_col] == min_sales_period].to_dict("records")
                        }
                    except Exception as e:
                        sales_stats["trend_error"] = str(e)
                
                sales_summary[col] = sales_stats
    
    # 트래픽 관련 특수 처리
    traffic_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                   ['visit', 'view', 'traffic', 'session', 'user', 'click', 'impression', 
                    '방문', '조회', '트래픽', '클릭', '노출'])]
    
    traffic_summary = {}
    if traffic_cols:
        for col in traffic_cols:
            if col in numeric_cols:
                # 트래픽 총량
                total = df[col].sum()
                traffic_stats = {
                    "total": total,
                    "daily_avg": total / basic_info["time_range"]["periods"] if "time_range" in basic_info else None,
                    "peak_traffic_rows": df.nlargest(5, col).to_dict("records")
                }
                
                # 시계열이 있는 경우 트래픽 트렌드 추가
                if date_cols:
                    try:
                        # 기간별 트래픽 추이
                        time_col = date_cols[0]
                        periodic_traffic = df.groupby(time_col)[col].sum()
                        
                        traffic_stats["trend"] = {
                            "first_period": periodic_traffic.iloc[0],
                            "last_period": periodic_traffic.iloc[-1],
                            "growth_rate": ((periodic_traffic.iloc[-1] / periodic_traffic.iloc[0]) - 1) * 100 if periodic_traffic.iloc[0] != 0 else None,
                            "first_period_rows": df[df[time_col] == periodic_traffic.index[0]].to_dict("records"),
                            "last_period_rows": df[df[time_col] == periodic_traffic.index[-1]].to_dict("records")
                        }
                        
                        # 트래픽이 가장 높은/낮은 기간
                        max_traffic_period = periodic_traffic.idxmax()
                        min_traffic_period = periodic_traffic.idxmin()
                        
                        traffic_stats["peak_period"] = {
                            "time": max_traffic_period.strftime("%Y-%m-%d") if hasattr(max_traffic_period, 'strftime') else str(max_traffic_period),
                            "value": periodic_traffic.max(),
                            "rows": df[df[time_col] == max_traffic_period].to_dict("records")
                        }
                        
                        traffic_stats["lowest_period"] = {
                            "time": min_traffic_period.strftime("%Y-%m-%d") if hasattr(min_traffic_period, 'strftime') else str(min_traffic_period),
                            "value": periodic_traffic.min(),
                            "rows": df[df[time_col] == min_traffic_period].to_dict("records")
                        }
                    except Exception as e:
                        traffic_stats["trend_error"] = str(e)
                
                traffic_summary[col] = traffic_stats
    
    # 제품 퍼널 관련 특수 처리
    funnel_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                  ['conversion', 'funnel', 'step', 'stage', 'cart', 'checkout', 'purchase',
                   '전환', '퍼널', '단계', '장바구니', '결제', '구매'])]
    
    funnel_summary = {}
    if funnel_cols and len(funnel_cols) >= 2:
        funnel_data = {
            "steps": funnel_cols,
            "steps_data": {}
        }
        
        # 각 퍼널 단계 데이터
        for col in funnel_cols:
            if col in numeric_cols:
                funnel_data["steps_data"][col] = {
                    "total": df[col].sum(),
                    "avg": df[col].mean(),
                    "top_rows": df.nlargest(5, col).to_dict("records")
                }
        
        # 퍼널 단계 간 전환율 계산 시도
        try:
            conversions = []
            for i in range(len(funnel_cols) - 1):
                current_col = funnel_cols[i]
                next_col = funnel_cols[i + 1]
                
                if current_col in numeric_cols and next_col in numeric_cols:
                    current_total = df[current_col].sum()
                    next_total = df[next_col].sum()
                    
                    if current_total > 0:
                        conversion_rate = (next_total / current_total) * 100
                        
                        # 전환율 관련 데이터
                        conversion_data = {
                            "from_step": current_col,
                            "to_step": next_col,
                            "conversion_rate": conversion_rate,
                            "current_step_total": current_total,
                            "next_step_total": next_total
                        }
                        
                        # 전환율이 높은/낮은 행 찾기 (개별 행 기준)
                        if current_total > 0 and next_total > 0:
                            # 개별 행의 전환율 계산 (0으로 나누기 방지)
                            df_filtered = df[(df[current_col] > 0)].copy()
                            if not df_filtered.empty:
                                df_filtered['conversion_rate'] = (df_filtered[next_col] / df_filtered[current_col]) * 100
                                
                                # 전환율 높은/낮은 행
                                conversion_data["high_conversion_rows"] = df_filtered.nlargest(5, 'conversion_rate').to_dict("records")
                                conversion_data["low_conversion_rows"] = df_filtered.nsmallest(5, 'conversion_rate').to_dict("records")
                        
                        conversions.append(conversion_data)
            
            funnel_data["conversions"] = conversions
            
            # 시계열 데이터가 있는 경우, 기간별 전환율 추이
            if date_cols:
                time_col = date_cols[0]
                time_based_conversions = []
                
                # 기간별로 그룹화하여 전환율 계산
                periods = df[time_col].dt.to_period('M') if hasattr(df[time_col], 'dt') else df[time_col]
                period_groups = df.groupby(periods)
                
                for period, group in period_groups:
                    period_conversions = []
                    
                    for i in range(len(funnel_cols) - 1):
                        current_col = funnel_cols[i]
                        next_col = funnel_cols[i + 1]
                        
                        if current_col in numeric_cols and next_col in numeric_cols:
                            current_total = group[current_col].sum()
                            next_total = group[next_col].sum()
                            
                            if current_total > 0:
                                conversion_rate = (next_total / current_total) * 100
                                period_conversions.append({
                                    "from_step": current_col,
                                    "to_step": next_col,
                                    "conversion_rate": conversion_rate,
                                    "period": str(period)
                                })
                    
                    if period_conversions:
                        time_based_conversions.append({
                            "period": str(period),
                            "conversions": period_conversions,
                            "rows": group.to_dict("records")
                        })
                
                funnel_data["time_based_conversions"] = time_based_conversions
        except Exception as e:
            funnel_data["conversion_error"] = str(e)
        
        funnel_summary = funnel_data
    
    # 최종 요약 정보
    summary = {
        "basic_info": basic_info,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "sales_summary": sales_summary,
        "traffic_summary": traffic_summary,
        "funnel_summary": funnel_summary
    }
    
    return summary


def get_relevant_summary(summary: Dict, question: str) -> Dict:
    """
    Extracts only the relevant summary information from the dataframe based on the user's question.
    
    Args:
        summary: Complete summary information returned by the summarize_dataframe function
        question: User's question
        
    Returns:
        Dictionary containing only the relevant summary information for the question
    """
    relevant_summary = {"basic_info": summary["basic_info"]}
    
    # Keywords organized by data type/domain to identify relevant sections
    keywords = {
        "numeric": [
            # Statistical terms
            "mean", "average", "median", "mode", "std", "standard deviation", "variance",
            "max", "maximum", "min", "minimum", "highest", "lowest", "peak", "bottom",
            "percentile", "quartile", "range", "iqr", "interquartile", "distribution",
            
            # Analysis terms
            "statistics", "statistical", "metrics", "measurement", "value", "figure", "number",
            "outlier", "anomaly", "abnormal", "unusual", "extreme", "significant", "deviation",
            "trend", "pattern", "fluctuation", "volatility", "stable", "unstable", "correlation",
            "regression", "forecast", "predict", "estimate", "calculate", "compute",
            
            # Comparison terms
            "compare", "comparison", "difference", "gap", "exceed", "surpass", "higher than", "lower than",
            "increase", "decrease", "growth", "decline", "rise", "fall", "change", "shift", "movement",
            "above average", "below average", "over", "under", "exceed", "deficit", "surplus",
            
            # Time-related
            "period", "interval", "duration", "timeframe", "monthly", "weekly", "daily", "yearly", "annual",
            "quarter", "seasonal", "cycle", "frequency", "regular", "irregular"
        ],
        
        "categorical": [
            # Category terms
            "category", "group", "class", "classification", "type", "kind", "sort", "variety",
            "segment", "section", "division", "bucket", "cluster", "genre", "species", "family",
            
            # Analysis terms
            "distribution", "composition", "makeup", "structure", "proportion", "ratio",
            "percentage", "fraction", "share", "portion", "allocation", "breakdown",
            "dominant", "prevalent", "common", "rare", "unique", "distinct", "diverse",
            "homogeneous", "heterogeneous", "similar", "different", "various", "mixed",
            
            # Actions
            "categorize", "classify", "group", "segment", "divide", "separate", "split",
            "organize", "arrange", "sort", "order", "rank", "label", "tag", "name",
            
            # Comparison
            "compare", "contrast", "versus", "vs", "against", "relative to", "compared to",
            "comparison", "difference", "similarity", "pattern", "relationship"
        ],
        
        "sales": [
            # General sales terms
            "revenue", "sales", "income", "earnings", "proceeds", "turnover", "profit", "margin",
            "gross", "net", "return", "yield", "roi", "bottom line", "cash flow", "monetization",
            "transaction", "payment", "monetize", "commercialize", "sell", "sold", "purchase",
            
            # Sales metrics
            "amount", "volume", "quantity", "units", "pieces", "items", "order", "invoice",
            "average order value", "aov", "lifetime value", "ltv", "customer acquisition cost", "cac",
            "cost of sales", "cogs", "overhead", "expense", "cost", "price", "pricing", "discount",
            "markup", "margin", "mrr", "arr", "recurring revenue", "retention", "churn",
            
            # Performance terms
            "target", "goal", "quota", "forecast", "projection", "estimate", "budget", "plan",
            "actual", "achievement", "performance", "kpi", "benchmark", "threshold", "milestone",
            "growth", "increase", "decrease", "trend", "peak", "seasonal", "holiday", "promotion",
            
            # Best/worst performance
            "best-selling", "top-selling", "popular", "hit", "blockbuster", "high-performing", 
            "low-performing", "underperforming", "lagging", "leading", "successful", "unsuccessful",
            "profitable", "unprofitable", "highest revenue", "lowest revenue"
        ],
        
        "traffic": [
            # General traffic terms
            "traffic", "visitor", "visit", "view", "pageview", "impression", "session", "engagement",
            "interaction", "activity", "footfall", "attendance", "audience", "crowd", "volume", "flow",
            
            # User terms
            "user", "client", "customer", "consumer", "browser", "viewer", "visitor", "guest",
            "member", "subscriber", "account", "profile", "audience", "demographic", "segment",
            
            # Web analytics
            "click", "tap", "swipe", "scroll", "navigation", "bounce", "exit", "landing", "referral",
            "source", "medium", "campaign", "utm", "organic", "direct", "social", "email", "ppc",
            "seo", "serp", "ranking", "position", "visibility", "reach", "impression", "exposure",
            
            # Metrics
            "unique", "returning", "new", "repeat", "loyal", "casual", "engaged", "active", "inactive",
            "dormant", "churned", "retained", "acquisition", "retention", "frequency", "recency",
            "duration", "time spent", "depth", "pages per visit", "screens per session",
            
            # Performance
            "peak", "off-peak", "busy", "quiet", "spike", "surge", "drop", "decline", "increase",
            "growth", "trend", "pattern", "seasonal", "daily", "weekly", "monthly", "real-time"
        ],
        
        "funnel": [
            # Funnel stages
            "funnel", "pipeline", "journey", "path", "flow", "process", "sequence", "lifecycle",
            "stage", "step", "phase", "touchpoint", "milestone", "checkpoint", "waypoint",
            "awareness", "interest", "consideration", "intent", "evaluation", "decision", "action",
            "acquisition", "activation", "retention", "referral", "revenue", "lead", "prospect",
            
            # E-commerce specific
            "cart", "basket", "wishlist", "checkout", "payment", "shipping", "delivery", "purchase",
            "order", "transaction", "browse", "view", "add to cart", "abandon", "complete",
            "product page", "category page", "search results", "landing page", "confirmation",
            
            # Conversion terms
            "conversion", "convert", "transformer", "completion", "success", "goal", "objective",
            "target", "finish", "end", "complete", "incomplete", "partial", "abandoned", "dropped",
            "exit", "bounce", "fall-off", "leak", "bottleneck", "obstacle", "barrier", "friction",
            
            # Rates and metrics
            "rate", "ratio", "percentage", "proportion", "efficiency", "effectiveness", "performance",
            "optimization", "improvement", "enhancement", "progression", "advancement", "drop-off",
            "abandonment", "completion", "bounce rate", "exit rate", "conversion rate", "success rate",
            
            # Analysis
            "analyze", "measure", "track", "monitor", "observe", "study", "examine", "investigate",
            "diagnose", "troubleshoot", "identify", "determine", "evaluate", "assess", "quantify"
        ],
        
        "time": [
            # Time periods
            "time", "period", "duration", "interval", "timeframe", "span", "range", "window",
            "moment", "instant", "point", "date", "day", "week", "month", "quarter", "year",
            "hour", "minute", "second", "decade", "century", "millennium", "era", "epoch",
            "season", "semester", "trimester", "fiscal", "calendar", "annual", "quarterly", 
            "monthly", "weekly", "daily", "hourly", "yesterday", "today", "tomorrow",
            
            # Time-based analysis
            "trend", "pattern", "cycle", "frequency", "periodicity", "seasonal", "recurring",
            "regular", "irregular", "sporadic", "continuous", "discontinuous", "intermittent",
            "start", "begin", "end", "finish", "launch", "conclude", "commence", "terminate",
            
            # Temporal comparisons
            "before", "after", "during", "while", "meanwhile", "simultaneously", "concurrently",
            "previously", "subsequently", "formerly", "lately", "recently", "historically",
            "earlier", "later", "sooner", "past", "present", "future", "upcoming", "previous",
            "next", "last", "first", "current", "ongoing", "upcoming", "approaching", "elapsed",
            
            # Time series terms
            "time series", "chronological", "sequential", "temporal", "timeline", "timestamp",
            "datetime", "date range", "period over period", "year over year", "yoy", "month over month",
            "mom", "week over week", "wow", "day over day", "dod", "quarter over quarter", "qoq"
        ]
    }
    
    # Check if question contains keywords and include relevant sections
    question_lower = question.lower()
    
    # Track which sections were included
    included_sections = []
    
    for key, terms in keywords.items():
        if any(term in question_lower for term in terms):
            if key == "numeric" and "numeric_stats" in summary:
                relevant_summary["numeric_stats"] = summary["numeric_stats"]
                included_sections.append("numeric_stats")
            elif key == "categorical" and "categorical_stats" in summary:
                relevant_summary["categorical_stats"] = summary["categorical_stats"]
                included_sections.append("categorical_stats")
            elif key == "sales" and "sales_summary" in summary:
                relevant_summary["sales_summary"] = summary["sales_summary"]
                included_sections.append("sales_summary")
            elif key == "traffic" and "traffic_summary" in summary:
                relevant_summary["traffic_summary"] = summary["traffic_summary"]
                included_sections.append("traffic_summary")
            elif key == "funnel" and "funnel_summary" in summary:
                relevant_summary["funnel_summary"] = summary["funnel_summary"]
                included_sections.append("funnel_summary")
            elif key == "time" and "basic_info" in summary and "time_range" in summary["basic_info"]:
                # If time-related question but no specific domain mentioned,
                # include time-related information from all available sections
                if not any(section in included_sections for section in ["sales_summary", "traffic_summary", "funnel_summary"]):
                    if "sales_summary" in summary:
                        relevant_summary["sales_summary"] = summary["sales_summary"]
                        included_sections.append("sales_summary")
                    if "traffic_summary" in summary:
                        relevant_summary["traffic_summary"] = summary["traffic_summary"]
                        included_sections.append("traffic_summary")
    
    # If no specific sections matched or only basic info is included,
    # include numeric stats as default to provide some useful information
    if len(relevant_summary.keys()) <= 1 and "numeric_stats" in summary:
        relevant_summary["numeric_stats"] = summary["numeric_stats"]
    
    return relevant_summary


# 사용 예시:
if __name__ == "__main__":
    # 샘플 데이터 생성
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # 매출 데이터
    sales_data = pd.DataFrame({
        'date': dates,
        'revenue': np.random.normal(loc=10000, scale=2000, size=len(dates)),
        'transactions': np.random.poisson(lam=500, size=len(dates)),
        'avg_order_value': np.random.normal(loc=20, scale=5, size=len(dates)),
        'product_category': np.random.choice(['A', 'B', 'C', 'D'], size=len(dates))
    })
    
    # 데이터프레임 요약
    summary = summarize_dataframe(sales_data, "sales_data")
    
    # 특정 질문에 관련된 요약 정보 추출
    question = "지난 달 대비 매출 성장률은 얼마인가요?"
    relevant_info = get_relevant_summary(summary, question)
    
    print("전체 요약 정보 키:", summary.keys())
    print("관련 요약 정보 키:", relevant_info.keys())
