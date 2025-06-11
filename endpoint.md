# API 엔드포인트 명세서

## 엔드포인트 정보
- **엔드포인트명**: [엔드포인트 이름]
- **HTTP 메서드**: `GET` / `POST` / `PUT` / `DELETE`
- **URL**: `{baseUrl}/api/v1/endpoint-path`
- **설명**: [엔드포인트의 기능과 목적]
- **작성자**: [작성자명]
- **작성일**: YYYY-MM-DD
- **최종 수정일**: YYYY-MM-DD

## 요청 (Request)

### URL 구조
```
{baseUrl}/api/v1/users/{userId}/orders/{orderId}
```

### 경로 매개변수 (Path Parameters)
| 매개변수 | 타입 | 필수여부 | 설명 | 예시 |
|----------|------|----------|------|------|
| userId | integer | 필수 | 사용자 고유 ID | 123 |
| orderId | string | 필수 | 주문 고유 ID | "ORD-2024-001" |

### 쿼리 매개변수 (Query Parameters)
| 매개변수 | 타입 | 필수여부 | 기본값 | 설명 | 예시 |
|----------|------|----------|--------|------|------|
| page | integer | 선택 | 1 | 페이지 번호 | 1 |
| limit | integer | 선택 | 20 | 페이지당 항목 수 (최대 100) | 50 |
| status | string | 선택 | all | 주문 상태 필터 | "pending" |
| sort | string | 선택 | created_at | 정렬 기준 | "amount" |

### 헤더 (Headers)
| 헤더명 | 타입 | 필수여부 | 설명 | 예시 |
|--------|------|----------|------|------|
| Authorization | string | 필수 | 인증 토큰 | "Bearer eyJ0eXAiOiJKV1Q..." |
| Content-Type | string | 필수 | 요청 데이터 형식 | "application/json" |
| Accept | string | 선택 | 응답 데이터 형식 | "application/json" |

### 요청 본문 (Request Body)
> POST, PUT 요청의 경우에만 작성

```json
{
  "name": "string",
  "email": "string",
  "age": "integer",
  "address": {
    "street": "string",
    "city": "string",
    "zipCode": "string"
  },
  "preferences": ["string"]
}
```

#### 요청 본문 스키마
| 필드명 | 타입 | 필수여부 | 제약조건 | 설명 |
|--------|------|----------|----------|------|
| name | string | 필수 | 2-50자 | 사용자 이름 |
| email | string | 필수 | 이메일 형식 | 사용자 이메일 |
| age | integer | 선택 | 1-120 | 사용자 나이 |
| address | object | 선택 | - | 주소 정보 |
| address.street | string | 선택 | 최대 100자 | 도로명 주소 |
| address.city | string | 선택 | 최대 50자 | 도시명 |
| address.zipCode | string | 선택 | 5-10자 | 우편번호 |
| preferences | array | 선택 | 최대 10개 | 사용자 선호사항 |

## 응답 (Response)

### 성공 응답 (200 OK)
```json
{
  "success": true,
  "data": {
    "id": 123,
    "name": "홍길동",
    "email": "hong@example.com",
    "age": 30,
    "address": {
      "street": "서울특별시 강남구 테헤란로 123",
      "city": "서울",
      "zipCode": "06234"
    },
    "preferences": ["tech", "music"],
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T12:00:00Z"
  },
  "message": "요청이 성공적으로 처리되었습니다."
}
```

#### 응답 스키마
| 필드명 | 타입 | 설명 |
|--------|------|------|
| success | boolean | 요청 성공 여부 |
| data | object | 응답 데이터 |
| data.id | integer | 사용자 고유 ID |
| data.name | string | 사용자 이름 |
| data.email | string | 사용자 이메일 |
| data.created_at | string (ISO 8601) | 생성 일시 |
| data.updated_at | string (ISO 8601) | 수정 일시 |
| message | string | 성공 메시지 |

### 오류 응답

#### 400 Bad Request
```json
{
  "success": false,
  "error": {
    "code": "INVALID_REQUEST",
    "message": "요청 데이터가 올바르지 않습니다.",
    "details": [
      {
        "field": "email",
        "message": "올바른 이메일 형식이 아닙니다."
      }
    ]
  }
}
```

#### 401 Unauthorized
```json
{
  "success": false,
  "error": {
    "code": "UNAUTHORIZED",
    "message": "인증이 필요합니다."
  }
}
```

#### 404 Not Found
```json
{
  "success": false,
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "요청한 리소스를 찾을 수 없습니다."
  }
}
```

## 사용 예시

### cURL
```bash
curl -X GET "https://api.example.com/v1/users/123?page=1&limit=20" \
  -H "Authorization: Bearer your-token-here" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json"
```

### JavaScript
```javascript
const response = await fetch('https://api.example.com/v1/users/123?page=1&limit=20', {
  method: 'GET',
  headers: {
    'Authorization': 'Bearer your-token-here',
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  }
});

const data = await response.json();
console.log(data);
```

### Python
```python
import requests

url = "https://api.example.com/v1/users/123"
params = {"page": 1, "limit": 20}
headers = {
    "Authorization": "Bearer your-token-here",
    "Content-Type": "application/json"
}

response = requests.get(url, params=params, headers=headers)
data = response.json()
print(data)
```

## 비즈니스 로직
- [엔드포인트의 비즈니스 로직 설명]
- [데이터 처리 과정]
- [유효성 검증 규칙]
- [관련 비즈니스 규칙]

## 제약사항 및 주의사항
- **Rate Limiting**: 분당 100회 요청 제한
- **데이터 크기**: 요청 본문 최대 1MB
- **권한**: 관리자 권한 필요
- **기타**: [추가 제약사항]

## 관련 엔드포인트
- `GET /users` - 사용자 목록 조회
- `POST /users` - 사용자 생성
- `PUT /users/{id}` - 사용자 정보 수정
- `DELETE /users/{id}` - 사용자 삭제

## 변경 이력
| 버전 | 날짜 | 변경사항 | 작성자 |
|------|------|----------|--------|
| 1.0 | 2024-01-01 | 초기 버전 작성 | 개발팀 |
| 1.1 | 2024-01-15 | 응답 스키마 수정 | 개발팀 |
