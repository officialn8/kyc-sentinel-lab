# API Documentation

Base URL: `http://localhost:8000`

Interactive documentation available at `/docs` (Swagger UI) or `/redoc`.

## Sessions

### Create Session

```http
POST /api/sessions
Content-Type: application/json

{
  "source": "upload",
  "attack_family": null,
  "attack_severity": null,
  "device_os": "iOS 17.2",
  "device_model": "iPhone 15 Pro",
  "ip_country": "US",
  "selfie_filename": "selfie.jpg",
  "selfie_content_type": "image/jpeg",
  "selfie_size_bytes": 532112,
  "id_filename": "id-front.jpg",
  "id_content_type": "image/jpeg",
  "id_size_bytes": 804221
}
```

**Response:**

```json
{
  "session": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z",
    "status": "pending",
    "source": "upload",
    "attack_family": null,
    "attack_severity": null
  },
  "selfie_upload": {
    "method": "PUT",
    "url": "https://...",
    "fields": {},
    "headers": {
      "Content-Type": "image/jpeg"
    },
    "asset_key": "sessions/550e.../selfie.jpg",
    "expires_in": 3600,
    "ticket": "<opaque-signed-ticket>"
  },
  "id_upload": {
    "method": "PUT",
    "url": "https://...",
    "fields": {},
    "headers": {
      "Content-Type": "image/jpeg"
    },
    "asset_key": "sessions/550e.../id.jpg",
    "expires_in": 3600,
    "ticket": "<opaque-signed-ticket>"
  }
}
```

### Finalize Session

After uploading assets, finalize to start processing:

```http
POST /api/sessions/{session_id}/finalize
Content-Type: application/json

{
  "selfie_ticket": "<opaque-signed-ticket>",
  "id_ticket": "<opaque-signed-ticket>"
}
```

**Response:**

```json
{
  "status": "processing",
  "message": "Session processing started"
}
```

### List Sessions

```http
GET /api/sessions?page=1&page_size=20&status=completed&attack_family=replay
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `page` | int | Page number (default: 1) |
| `page_size` | int | Items per page (default: 20, max: 100) |
| `status` | string | Filter by status |
| `source` | string | Filter by source |
| `attack_family` | string | Filter by attack family |
| `decision` | string | Filter by decision |

**Response:**

```json
{
  "items": [...],
  "total": 150,
  "page": 1,
  "page_size": 20,
  "pages": 8
}
```

### Get Session Detail

```http
GET /api/sessions/{session_id}
```

**Response:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:35:00Z",
  "status": "completed",
  "source": "upload",
  "attack_family": null,
  "attack_severity": null,
  "device_os": "iOS 17.2",
  "device_model": "iPhone 15 Pro",
  "ip_country": "US",
  "selfie_url": "https://...",
  "id_url": "https://...",
  "result": {
    "id": "...",
    "risk_score": 25,
    "decision": "pass",
    "face_similarity": 0.92,
    "pad_score": 0.08,
    "doc_score": 0.12,
    "model_version": "v1",
    "rules_version": "v1"
  },
  "reasons": []
}
```

### Delete Session

```http
DELETE /api/sessions/{session_id}
```

### Find Similar Sessions

```http
GET /api/sessions/{session_id}/similar?limit=10
```

## Simulation

### List Attack Families

```http
GET /api/simulate/families
```

**Response:**

```json
[
  {
    "id": "replay",
    "name": "Replay Attack",
    "description": "Screen capture of a genuine user's session...",
    "severities": ["low", "medium", "high"]
  },
  ...
]
```

### Generate Synthetic Sessions

```http
POST /api/simulate
Content-Type: application/json

{
  "attack_family": "replay",
  "attack_severity": "medium",
  "count": 5
}
```

**Response:**

```json
[
  {
    "id": "...",
    "status": "processing",
    "source": "synthetic",
    "attack_family": "replay",
    "attack_severity": "medium"
  },
  ...
]
```

## Metrics

### Summary Metrics

```http
GET /api/metrics/summary
```

**Response:**

```json
{
  "total_sessions": 1500,
  "completed_sessions": 1450,
  "pass_count": 1100,
  "review_count": 250,
  "fail_count": 100,
  "avg_risk_score": 28.5,
  "detection_rate": 94.5
}
```

### Attack Family Breakdown

```http
GET /api/metrics/by-attack-family
```

**Response:**

```json
{
  "families": [
    {
      "family": "replay",
      "total": 200,
      "detected": 190,
      "missed": 10,
      "detection_rate": 95.0,
      "avg_risk_score": 72.3
    },
    ...
  ]
}
```

### Confusion Matrix

```http
GET /api/metrics/confusion-matrix
```

**Response:**

```json
{
  "cells": [
    {"actual": "benign", "predicted": "pass", "count": 950},
    {"actual": "benign", "predicted": "review", "count": 40},
    {"actual": "benign", "predicted": "fail", "count": 10},
    {"actual": "replay", "predicted": "pass", "count": 10},
    {"actual": "replay", "predicted": "review", "count": 40},
    {"actual": "replay", "predicted": "fail", "count": 150},
    ...
  ],
  "total": 1500
}
```

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message here"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad request (validation error) |
| 404 | Resource not found |
| 422 | Unprocessable entity |
| 500 | Internal server error |

## Outbound Webhook Signing

Webhook deliveries are signed with HMAC-SHA256 using `WEBHOOK_SECRET`.

Headers on each delivery:

- `X-KYC-Event`
- `X-KYC-Timestamp` (unix epoch seconds)
- `X-KYC-Delivery-Id` (UUID)
- `X-KYC-Signature` (`v1=<hex_hmac>`)
- `X-Webhook-Event` (legacy compatibility header)

Signature input format:

```text
{timestamp}.{delivery_id}.{raw_json_body}
```

Verification sketch (Python):

```python
import hashlib
import hmac

def verify(secret: str, timestamp: str, delivery_id: str, body: bytes, signature_header: str) -> bool:
    if not signature_header.startswith("v1="):
        return False
    expected = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.{delivery_id}.{body.decode('utf-8')}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(f"v1={expected}", signature_header)
```

Replay guidance: reject timestamps older than 5 minutes and deduplicate by `X-KYC-Delivery-Id`.










