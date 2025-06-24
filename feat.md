# CCTV Object Detection 오탐 감소 기능 구현 명세서

## 1. 기능 개요

### 1.1 목적
- 기존 CCTV Object Detection 시스템에서 발생하는 오탐(False Positive)을 사용자 피드백을 통해 학습하여 점진적으로 감소시키는 기능 구현
- 사용자가 자연어로 오탐 피드백을 제공하면, LLM 에이전트가 이를 해석하여 Object Detection 모델이 해당 케이스를 학습하도록 하는 시스템

### 1.2 주요 기능
- 사용자의 자연어 오탐 피드백 해석 및 분류
- 오탐된 객체의 시각적 특성 학습 및 저장
- 실시간 추론 시 학습된 오탐 케이스와의 유사도 비교를 통한 필터링
- 사용자별 개별 오탐 학습 데이터 관리

## 2. 시스템 아키텍처

### 2.1 전체 구조
```
사용자 피드백 → LLM 에이전트 → Object Detection 모듈 → 오탐 학습 저장소
                     ↓
            Task Type 분류 및 응답
```

### 2.2 주요 컴포넌트
- **LLM 에이전트**: 사용자 피드백 해석 및 태스크 분류
- **Object Detection 모듈**: 객체 탐지 및 오탐 학습 기능
- **임베딩 저장소**: 사용자별 오탐 이미지 임베딩 및 클래스 정보 저장
- **유사도 비교 엔진**: 실시간 추론 시 오탐 케이스와의 유사도 계산

## 3. 상세 기능 명세

### 3.1 LLM 에이전트 모듈

#### 3.1.1 입력 처리
**입력 형태:**
```json
{
  "user_id": "string",
  "message": "string",
  "timestamp": "datetime",
  "detection_context": {
    "image_path": "string",
    "detected_objects": "array",
    "confidence_scores": "array"
  }
}
```

**입력 예시:**
- "지금 잘못 감지됐어. 저런 케이스는 탐지하지 말아줘."
- "방금 탐지된 건 잘못된거야."
- "이건 사람이 아니라 마네킹이야."

#### 3.1.2 태스크 분류
**분류 대상 태스크 타입:**
- `FALSE_POSITIVE_FEEDBACK`: 오탐 피드백
- `GENERAL_INQUIRY`: 일반 문의
- `SETTING_CHANGE`: 설정 변경 요청
- `STATUS_CHECK`: 상태 확인

#### 3.1.3 출력 처리
**출력 형태:**
```json
{
  "task_type": "string",
  "response_message": "string",
  "extracted_info": {
    "target_class": "string",
    "confidence_level": "float",
    "action_required": "boolean"
  }
}
```

### 3.2 Object Detection 모듈

#### 3.2.1 오탐 학습 기능

**입력 처리:**
```json
{
  "user_id": "string",
  "image_data": "base64 encoded image",
  "false_positive_class": "string",
  "bounding_box": {
    "x": "int",
    "y": "int", 
    "width": "int",
    "height": "int"
  }
}
```

**처리 과정:**
1. 입력 이미지에서 지정된 bounding_box 영역 추출
2. 추출된 영역을 Vision Encoder를 통해 임베딩 벡터 생성
3. 사용자별 오탐 데이터베이스에 저장

**저장 데이터 구조:**
```json
{
  "user_id": "string",
  "false_positive_id": "string",
  "class_name": "string",
  "embedding_vector": "array[float]",
  "timestamp": "datetime",
  "image_metadata": {
    "width": "int",
    "height": "int",
    "channels": "int"
  }
}
```

#### 3.2.2 실시간 필터링 기능

**처리 단계:**
1. **기본 Object Detection 수행**
   - 입력 이미지에 대해 일반적인 객체 탐지 수행
   - 각 탐지된 객체의 confidence score 계산

2. **오탐 유사도 검사**
   - 탐지된 각 객체 영역을 임베딩으로 변환
   - 해당 사용자의 저장된 오탐 임베딩과 코사인 유사도 계산
   - 동일한 클래스명에 대해서만 유사도 비교 수행

3. **필터링 적용**
   - 유사도가 임계값(기본값: 0.85) 이상인 경우 해당 탐지 결과 제외
   - 최종 탐지 결과 반환

**유사도 계산 알고리즘:**
```python
def calculate_similarity(current_embedding, stored_embeddings, class_name):
    filtered_embeddings = [emb for emb in stored_embeddings 
                          if emb['class_name'] == class_name]
    
    similarities = []
    for stored_emb in filtered_embeddings:
        cosine_sim = cosine_similarity(current_embedding, 
                                     stored_emb['embedding_vector'])
        similarities.append(cosine_sim)
    
    return max(similarities) if similarities else 0.0
```

## 4. 데이터 구조 및 저장소

### 4.1 사용자별 오탐 학습 데이터
```sql
CREATE TABLE user_false_positives (
    id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    class_name VARCHAR(100) NOT NULL,
    embedding_vector JSON NOT NULL,
    created_at TIMESTAMP NOT NULL,
    image_metadata JSON,
    INDEX idx_user_class (user_id, class_name)
);
```

### 4.2 임베딩 벡터 관리
- **차원**: 512차원 (Vision Transformer 기준)
- **정규화**: L2 normalization 적용
- **저장 형태**: JSON array 또는 바이너리 형태

## 5. 성능 요구사항

### 5.1 응답시간
- LLM 에이전트 응답: 3초 이내
- 오탐 학습 처리: 5초 이내
- 실시간 필터링 추가 지연: 100ms 이내

### 5.2 정확도
- 오탐 피드백 분류 정확도: 90% 이상
- 유사 케이스 필터링 정확도: 85% 이상
- 새로운 정탐 케이스 오인식률: 5% 이하

### 5.3 저장 용량
- 사용자당 최대 1,000개 오탐 케이스 저장
- 임베딩 벡터당 약 2KB (512 float32)

## 6. 예외 처리 및 엣지 케이스

### 6.1 LLM 에이전트 예외 처리
- 모호한 피드백 메시지 처리
- 컨텍스트 정보 부족 시 추가 질문 생성
- 잘못된 태스크 분류 시 재분류 요청

### 6.2 Object Detection 모듈 예외 처리
- 임베딩 생성 실패 시 재시도 로직
- 저장소 용량 초과 시 오래된 데이터 삭제
- 유사도 계산 시간 초과 시 기본 탐지 결과 반환

### 6.3 데이터 일관성
- 동일한 케이스의 중복 학습 방지
- 사용자 삭제 시 관련 오탐 데이터 정리
- 임베딩 모델 업데이트 시 기존 데이터 호환성 처리

## 7. 테스트 시나리오

### 7.1 기능 테스트
1. **오탐 피드백 처리 테스트**
   - 다양한 자연어 피드백 입력
   - 태스크 타입 분류 정확도 검증
   - 응답 메시지 적절성 확인

2. **오탐 학습 테스트**
   - 이미지 임베딩 생성 및 저장 검증
   - 사용자별 데이터 격리 확인
   - 동일 케이스 중복 처리 테스트

3. **실시간 필터링 테스트**
   - 학습된 오탐 케이스 필터링 검증
   - 새로운 정탐 케이스 통과 확인
   - 임계값 조정에 따른 성능 변화 측정

### 7.2 성능 테스트
- 동시 사용자 처리 성능
- 대용량 오탐 데이터 처리 성능
- 실시간 추론 지연시간 측정

## 8. 컴포넌트별 Pseudo Code

### 8.1 LLM 에이전트 모듈

#### 8.1.1 메인 처리 함수
```python
class LLMAgent:
    def __init__(self, model_name, prompt_templates):
        self.llm_model = load_model(model_name)
        self.prompt_templates = prompt_templates
        self.task_classifier = TaskClassifier()
    
    def process_user_feedback(self, user_input):
        """
        사용자 피드백을 처리하고 적절한 태스크로 분류
        """
        # 1. 입력 전처리
        processed_input = self.preprocess_input(user_input)
        
        # 2. 태스크 분류
        task_type = self.classify_task(processed_input)
        
        # 3. 태스크별 처리
        if task_type == "FALSE_POSITIVE_FEEDBACK":
            return self.handle_false_positive_feedback(processed_input)
        elif task_type == "GENERAL_INQUIRY":
            return self.handle_general_inquiry(processed_input)
        elif task_type == "SETTING_CHANGE":
            return self.handle_setting_change(processed_input)
        else:
            return self.handle_unknown_task(processed_input)
    
    def classify_task(self, user_input):
        """
        사용자 입력을 분석하여 태스크 타입 분류
        """
        # 분류를 위한 프롬프트 생성
        classification_prompt = self.prompt_templates["task_classification"].format(
            user_input=user_input["message"],
            context=user_input.get("detection_context", "")
        )
        
        # LLM을 통한 분류
        llm_response = self.llm_model.generate(
            prompt=classification_prompt,
            max_tokens=50,
            temperature=0.1
        )
        
        # 응답에서 태스크 타입 추출
        task_type = self.extract_task_type(llm_response)
        
        return task_type
    
    def handle_false_positive_feedback(self, user_input):
        """
        오탐 피드백 처리
        """
        # 1. 오탐 정보 추출
        extraction_prompt = self.prompt_templates["false_positive_extraction"].format(
            message=user_input["message"],
            detected_objects=user_input["detection_context"]["detected_objects"]
        )
        
        llm_response = self.llm_model.generate(
            prompt=extraction_prompt,
            max_tokens=200,
            temperature=0.2
        )
        
        # 2. 구조화된 정보 파싱
        extracted_info = self.parse_extraction_response(llm_response)
        
        # 3. Object Detection 모듈에 전달할 데이터 준비
        od_request = {
            "user_id": user_input["user_id"],
            "image_path": user_input["detection_context"]["image_path"],
            "false_positive_class": extracted_info["target_class"],
            "bounding_box": extracted_info["bounding_box"],
            "confidence": extracted_info["confidence"]
        }
        
        # 4. 응답 메시지 생성
        response_message = self.generate_response_message(
            "false_positive_acknowledged", 
            extracted_info
        )
        
        return {
            "task_type": "FALSE_POSITIVE_FEEDBACK",
            "response_message": response_message,
            "od_request": od_request,
            "extracted_info": extracted_info
        }
```

#### 8.1.2 프롬프트 템플릿
```python
PROMPT_TEMPLATES = {
    "task_classification": """
    사용자의 메시지를 분석하여 다음 중 하나로 분류하세요:
    - FALSE_POSITIVE_FEEDBACK: 잘못된 탐지에 대한 피드백
    - GENERAL_INQUIRY: 일반적인 질문이나 문의
    - SETTING_CHANGE: 설정 변경 요청
    - STATUS_CHECK: 상태 확인 요청
    
    사용자 메시지: "{user_input}"
    탐지 컨텍스트: "{context}"
    
    분류 결과:
    """,
    
    "false_positive_extraction": """
    사용자가 잘못된 탐지에 대해 피드백을 주었습니다.
    다음 정보를 추출하세요:
    1. 잘못 탐지된 객체의 클래스명
    2. 해당 객체의 위치 정보 (가능한 경우)
    3. 사용자의 확신 정도 (1-10)
    
    사용자 메시지: "{message}"
    탐지된 객체들: "{detected_objects}"
    
    추출된 정보:
    클래스명: 
    위치정보: 
    확신정도: 
    """
}
```

### 8.2 Object Detection 모듈

#### 8.2.1 메인 클래스
```python
class ObjectDetectionModule:
    def __init__(self, detection_model, vision_encoder, embedding_storage):
        self.detection_model = detection_model
        self.vision_encoder = vision_encoder
        self.embedding_storage = embedding_storage
        self.similarity_threshold = 0.85
    
    def detect_objects_with_filtering(self, image, user_id):
        """
        오탐 필터링이 적용된 객체 탐지
        """
        # 1. 기본 객체 탐지 수행
        raw_detections = self.detection_model.detect(image)
        
        # 2. 각 탐지 결과에 대해 오탐 필터링 적용
        filtered_detections = []
        
        for detection in raw_detections:
            if not self.is_false_positive(detection, image, user_id):
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def is_false_positive(self, detection, image, user_id):
        """
        탐지 결과가 오탐인지 확인
        """
        # 1. 탐지된 영역 이미지 추출
        bbox = detection["bounding_box"]
        cropped_image = self.crop_image(image, bbox)
        
        # 2. 이미지 임베딩 생성
        current_embedding = self.vision_encoder.encode(cropped_image)
        
        # 3. 저장된 오탐 임베딩들과 유사도 비교
        stored_embeddings = self.embedding_storage.get_user_embeddings(
            user_id, detection["class_name"]
        )
        
        max_similarity = 0.0
        for stored_embedding in stored_embeddings:
            similarity = self.calculate_cosine_similarity(
                current_embedding, 
                stored_embedding["embedding_vector"]
            )
            max_similarity = max(max_similarity, similarity)
        
        # 4. 임계값 이상이면 오탐으로 판단
        return max_similarity >= self.similarity_threshold
    
    def learn_false_positive(self, user_id, image_path, class_name, bounding_box):
        """
        오탐 케이스 학습
        """
        try:
            # 1. 이미지 로드
            image = self.load_image(image_path)
            
            # 2. 지정된 영역 추출
            cropped_image = self.crop_image(image, bounding_box)
            
            # 3. 이미지 임베딩 생성
            embedding_vector = self.vision_encoder.encode(cropped_image)
            
            # 4. 중복 체크 (유사한 임베딩이 이미 존재하는지)
            if self.is_duplicate_embedding(user_id, class_name, embedding_vector):
                return {"success": False, "reason": "duplicate_embedding"}
            
            # 5. 오탐 데이터 저장
            false_positive_data = {
                "user_id": user_id,
                "class_name": class_name,
                "embedding_vector": embedding_vector.tolist(),
                "timestamp": datetime.now(),
                "image_metadata": {
                    "width": cropped_image.width,
                    "height": cropped_image.height,
                    "channels": cropped_image.channels
                }
            }
            
            success = self.embedding_storage.save_false_positive(false_positive_data)
            
            return {"success": success}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def calculate_cosine_similarity(self, embedding1, embedding2):
        """
        코사인 유사도 계산
        """
        # L2 정규화
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 코사인 유사도 계산
        dot_product = np.dot(embedding1, embedding2)
        similarity = dot_product / (norm1 * norm2)
        
        return similarity
    
    def is_duplicate_embedding(self, user_id, class_name, new_embedding):
        """
        중복 임베딩 체크
        """
        existing_embeddings = self.embedding_storage.get_user_embeddings(
            user_id, class_name
        )
        
        for existing in existing_embeddings:
            similarity = self.calculate_cosine_similarity(
                new_embedding, 
                np.array(existing["embedding_vector"])
            )
            
            # 매우 높은 유사도면 중복으로 간주
            if similarity > 0.95:
                return True
        
        return False
```

### 8.3 임베딩 저장소 모듈

#### 8.3.1 저장소 인터페이스
```python
class EmbeddingStorage:
    def __init__(self, database_config):
        self.db_connection = self.connect_database(database_config)
        self.max_embeddings_per_user = 1000
    
    def save_false_positive(self, false_positive_data):
        """
        오탐 데이터 저장
        """
        try:
            # 1. 사용자의 기존 임베딩 개수 확인
            current_count = self.count_user_embeddings(
                false_positive_data["user_id"]
            )
            
            # 2. 최대 개수 초과 시 오래된 데이터 삭제
            if current_count >= self.max_embeddings_per_user:
                self.cleanup_old_embeddings(
                    false_positive_data["user_id"],
                    current_count - self.max_embeddings_per_user + 1
                )
            
            # 3. 새로운 임베딩 데이터 삽입
            insert_query = """
            INSERT INTO user_false_positives 
            (id, user_id, class_name, embedding_vector, created_at, image_metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            
            embedding_id = self.generate_uuid()
            
            self.db_connection.execute(insert_query, [
                embedding_id,
                false_positive_data["user_id"],
                false_positive_data["class_name"],
                json.dumps(false_positive_data["embedding_vector"]),
                false_positive_data["timestamp"],
                json.dumps(false_positive_data["image_metadata"])
            ])
            
            self.db_connection.commit()
            return True
            
        except Exception as e:
            self.db_connection.rollback()
            print(f"Error saving false positive: {e}")
            return False
    
    def get_user_embeddings(self, user_id, class_name=None):
        """
        사용자의 오탐 임베딩 조회
        """
        try:
            if class_name:
                query = """
                SELECT embedding_vector, class_name, created_at
                FROM user_false_positives 
                WHERE user_id = ? AND class_name = ?
                ORDER BY created_at DESC
                """
                results = self.db_connection.execute(query, [user_id, class_name])
            else:
                query = """
                SELECT embedding_vector, class_name, created_at
                FROM user_false_positives 
                WHERE user_id = ?
                ORDER BY created_at DESC
                """
                results = self.db_connection.execute(query, [user_id])
            
            embeddings = []
            for row in results:
                embeddings.append({
                    "embedding_vector": json.loads(row[0]),
                    "class_name": row[1],
                    "created_at": row[2]
                })
            
            return embeddings
            
        except Exception as e:
            print(f"Error retrieving embeddings: {e}")
            return []
    
    def cleanup_old_embeddings(self, user_id, count_to_delete):
        """
        오래된 임베딩 데이터 정리
        """
        try:
            delete_query = """
            DELETE FROM user_false_positives 
            WHERE user_id = ? 
            AND id IN (
                SELECT id FROM user_false_positives 
                WHERE user_id = ? 
                ORDER BY created_at ASC 
                LIMIT ?
            )
            """
            
            self.db_connection.execute(delete_query, [
                user_id, user_id, count_to_delete
            ])
            
            self.db_connection.commit()
            
        except Exception as e:
            print(f"Error cleaning up embeddings: {e}")
```

### 8.4 유사도 비교 엔진

#### 8.4.1 유사도 계산 최적화
```python
class SimilarityEngine:
    def __init__(self, embedding_storage):
        self.embedding_storage = embedding_storage
        self.similarity_cache = {}  # 캐시를 통한 성능 최적화
    
    def batch_similarity_check(self, user_id, detections, image):
        """
        여러 탐지 결과에 대한 배치 유사도 검사
        """
        # 사용자의 모든 오탐 임베딩을 한 번에 로드
        all_stored_embeddings = self.embedding_storage.get_user_embeddings(user_id)
        
        # 클래스별로 그룹화
        embeddings_by_class = {}
        for emb in all_stored_embeddings:
            class_name = emb["class_name"]
            if class_name not in embeddings_by_class:
                embeddings_by_class[class_name] = []
            embeddings_by_class[class_name].append(emb)
        
        # 각 탐지 결과에 대해 유사도 검사
        results = []
        for detection in detections:
            is_fp = self.check_single_detection(
                detection, image, embeddings_by_class
            )
            results.append({
                "detection": detection,
                "is_false_positive": is_fp
            })
        
        return results
    
    def check_single_detection(self, detection, image, embeddings_by_class):
        """
        단일 탐지 결과에 대한 유사도 검사
        """
        class_name = detection["class_name"]
        
        # 해당 클래스의 오탐 임베딩이 없으면 정탐으로 간주
        if class_name not in embeddings_by_class:
            return False
        
        # 탐지된 영역 추출 및 임베딩 생성
        bbox = detection["bounding_box"]
        cropped_image = self.crop_image(image, bbox)
        current_embedding = self.generate_embedding(cropped_image)
        
        # 저장된 임베딩들과 유사도 비교
        max_similarity = 0.0
        stored_embeddings = embeddings_by_class[class_name]
        
        for stored_emb in stored_embeddings:
            similarity = self.calculate_optimized_similarity(
                current_embedding, 
                np.array(stored_emb["embedding_vector"])
            )
            max_similarity = max(max_similarity, similarity)
            
            # 임계값 초과 시 조기 종료
            if max_similarity >= 0.85:
                return True
        
        return False
    
    def calculate_optimized_similarity(self, emb1, emb2):
        """
        최적화된 유사도 계산 (벡터화 연산 사용)
        """
        # 미리 정규화된 임베딩 사용 (저장 시 정규화)
        dot_product = np.dot(emb1, emb2)
        return dot_product  # 이미 정규화된 벡터의 내적 = 코사인 유사도
```

### 8.5 메인 시스템 통합

#### 8.5.1 시스템 컨트롤러
```python
class CCTVFalsePositiveSystem:
    def __init__(self, config):
        self.llm_agent = LLMAgent(
            config["llm_model"], 
            PROMPT_TEMPLATES
        )
        self.od_module = ObjectDetectionModule(
            config["detection_model"],
            config["vision_encoder"],
            EmbeddingStorage(config["database"])
        )
        self.similarity_engine = SimilarityEngine(
            self.od_module.embedding_storage
        )
    
    def process_user_feedback(self, user_input):
        """
        사용자 피드백 처리 메인 함수
        """
        # 1. LLM 에이전트를 통한 피드백 분석
        llm_response = self.llm_agent.process_user_feedback(user_input)
        
        # 2. 오탐 피드백인 경우 학습 수행
        if llm_response["task_type"] == "FALSE_POSITIVE_FEEDBACK":
            od_request = llm_response["od_request"]
            learning_result = self.od_module.learn_false_positive(
                od_request["user_id"],
                od_request["image_path"],
                od_request["false_positive_class"],
                od_request["bounding_box"]
            )
            
            # 학습 결과에 따른 응답 메시지 업데이트
            if learning_result["success"]:
                llm_response["response_message"] += " 학습이 완료되었습니다."
            else:
                llm_response["response_message"] += f" 학습 중 오류가 발생했습니다: {learning_result.get('reason', 'unknown')}"
        
        return llm_response
    
    def detect_objects_with_filtering(self, image, user_id):
        """
        오탐 필터링이 적용된 객체 탐지 수행
        """
        return self.od_module.detect_objects_with_filtering(image, user_id)
    
    def get_system_status(self, user_id):
        """
        시스템 상태 조회
        """
        stored_embeddings = self.od_module.embedding_storage.get_user_embeddings(user_id)
        
        status = {
            "user_id": user_id,
            "total_false_positives_learned": len(stored_embeddings),
            "classes_with_false_positives": list(set([emb["class_name"] for emb in stored_embeddings])),
            "similarity_threshold": self.od_module.similarity_threshold,
            "system_status": "active"
        }
        
        return status
```

## 9. 향후 개선 방향

### 9.1 단기 개선사항
- 사용자별 임계값 자동 조정 기능
- 오탐 케이스의 시간별 가중치 적용
- 배치 학습을 통한 성능 최적화

### 9.2 장기 개선사항
- Few-shot learning을 통한 빠른 적응
- 사용자 간 오탐 패턴 공유 및 활용
- 능동 학습(Active Learning)을 통한 효율적 데이터 수집
