# API Documentation

The SHL Assessment Recommender provides a RESTful JSON API built with FastAPI. It follows a stateless design where the client is responsible for maintaining and sending the conversation history.

If running in development mode, the interactive OpenAPI UI is available at `http://localhost:8000/docs`.

---

## 1. POST `/api/v1/chat`

Submit a conversational message (with full history) and receive assessment recommendations from the SHL catalog.

**Endpoint:** `/api/v1/chat`  
**Method:** `POST`  
**Content-Type:** `application/json`

### Request Payload

| Field | Type | Required | Description |
|---|---|---|---|
| `messages` | `Array[Object]` | Yes | The complete conversation history. |
| `messages[].role` | `string` | Yes | Either `"user"` or `"assistant"`. |
| `messages[].content`| `string` | Yes | The text content of the message. |
| `max_recommendations`| `integer` | No | Maximum number of results to return (default: `5`, max: `10`). |

**Example Request:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "I am hiring a software engineer and need a test for python."
    }
  ],
  "max_recommendations": 3
}
```

### Response Payload

| Field | Type | Description |
|---|---|---|
| `reply` | `string` | Natural-language response from the assistant. May contain a clarification question. |
| `recommendations` | `Array[Object]` | Ranked list of recommended assessments. Empty if clarification is needed. |
| `end_of_conversation`| `boolean` | `true` if recommendations are provided and the flow is complete, `false` otherwise. |

**Example Response:**
```json
{
  "reply": "Based on your needs, I recommend the following Python assessments.",
  "recommendations": [
    {
      "name": "Python 3 Programming",
      "url": "https://www.shl.com/products/...",
      "test_type": "Knowledge & Skills"
    }
  ],
  "end_of_conversation": true
}
```

---

## 2. POST `/api/v1/compare`

Compare 2 to 5 assessments side-by-side based on catalog data.

**Endpoint:** `/api/v1/compare`  
**Method:** `POST`  
**Content-Type:** `application/json`

### Request Payload

| Field | Type | Required | Description |
|---|---|---|---|
| `assessments` | `Array[string]` | Yes | Array of 2-5 assessment names to compare. |

**Example Request:**
```json
{
  "assessments": [
    "Verify - Numerical Reasoning",
    "Verify - Inductive Reasoning"
  ]
}
```

### Response Payload

Returns a standard response wrapper containing the comparison text.

**Example Response:**
```json
{
  "data": {
    "comparison_text": "Here is a comparison between Verify - Numerical Reasoning and Verify - Inductive Reasoning:\n\n..."
  },
  "message": "Comparison generated successfully",
  "success": true
}
```

---

## 3. GET `/api/v1/health`

Health check endpoint for deployment monitoring (Render/Kubernetes).

**Endpoint:** `/api/v1/health`  
**Method:** `GET`

**Example Response:**
```json
{
  "data": {
    "status": "healthy",
    "version": "0.1.0",
    "environment": "production",
    "service": "SHL Assessment Recommender"
  },
  "message": "Service is healthy",
  "success": true
}
```
