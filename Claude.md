# Claude.md - KYC Sentinel Lab

This document provides context for AI assistants working on the KYC Sentinel Lab codebase.

## Project Overview

KYC Sentinel Lab is a **red-team simulator** for Know Your Customer (KYC) identity verification flows. It enables fraud teams to:

1. **Upload** real KYC sessions (selfie + ID document) for analysis
2. **Simulate** synthetic attack patterns (replay, injection, face-swap, document tampering)
3. **Evaluate** detection capabilities with explainable reason codes
4. **Review** results via a metrics dashboard with confusion matrices and score distributions

The core value proposition: test your KYC fraud detection without using real deepfakes, using synthetic artifacts that simulate attack patterns.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                 │
│                    (Next.js 16 App Router)                      │
├─────────────────────────────────────────────────────────────────┤
│  Dashboard  │  Sessions  │  Upload  │  Simulate  │  Metrics    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP/REST (proxied via /api/*)
┌──────────────────────────▼──────────────────────────────────────┐
│                         Backend                                  │
│                        (FastAPI)                                 │
├─────────────────────────────────────────────────────────────────┤
│  API Layer  │  Services  │  Detection  │  Processing Backend    │
└──────┬─────────────┬─────────────┬──────────────────────────────┘
       │             │             │
┌──────▼─────┐ ┌─────▼─────┐ ┌─────▼─────────────────────────────┐
│ PostgreSQL │ │    R2     │ │      Processing                   │
│ + pgvector │ │  Storage  │ │  (BackgroundTasks / Modal)        │
└────────────┘ └───────────┘ └───────────────────────────────────┘
```

### Processing Pipeline

1. **Create Session** → Generate presigned upload URLs
2. **Upload Assets** → Client uploads directly to R2/MinIO
3. **Finalize Session** → Triggers background processing
4. **Detection Pipeline**:
   - Frame Extraction (video → frames, if video)
   - Face Analysis (InsightFace: detection, embedding, similarity)
   - PAD Analysis (OpenCV: moiré, motion entropy, injection artifacts)
   - Document Analysis (PaddleOCR: OCR, tampering detection)
   - Scoring (weighted risk score, decision: pass/review/fail)
   - Reason Generation (explainable codes with evidence)

## Tech Stack

### Frontend (`/frontend`)
| Package | Version | Purpose |
|---------|---------|---------|
| Next.js | 16.x | App Router with React Server Components |
| React | 18.x | UI framework |
| TypeScript | 5.x | Type safety (strict mode) |
| Tailwind CSS | 3.x | Styling |
| shadcn/ui | - | Radix-based accessible components |
| TanStack Query | 5.x | Server state management |
| Recharts | 2.x | Data visualization |
| react-dropzone | 14.x | File upload handling |

### Backend (`/backend`)
| Package | Version | Purpose |
|---------|---------|---------|
| FastAPI | 0.109.x | Async web framework |
| SQLAlchemy | 2.0.x | Async ORM with type hints |
| Alembic | 1.13.x | Database migrations |
| Pydantic | 2.x | Validation and settings |
| asyncpg | 0.29.x | PostgreSQL async driver |
| pgvector | 0.2.x | Vector similarity search |
| boto3/aioboto3 | - | S3/R2 storage |
| InsightFace | 0.7.x | Face detection & embeddings |
| PaddleOCR | 2.7.x | Document OCR |
| OpenCV | 4.8.x | Image processing & PAD heuristics |
| Modal | 0.64.x | Optional serverless GPU processing |

### Infrastructure
- **Database**: PostgreSQL 16 + pgvector extension
- **Storage**: Cloudflare R2 (prod) / MinIO (dev)
- **GPU Processing**: Modal (optional, for production scale)
- **Deployment**: Vercel (frontend) + Railway (backend)

## Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- pnpm (recommended)

### Quick Start

```bash
# 1. Start local services (Postgres + MinIO)
docker-compose up -d

# 2. Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload --port 8000

# 3. Frontend setup (new terminal)
cd frontend
pnpm install
pnpm dev
```

### Environment Variables

Copy `env.example` to `.env` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection |
| `R2_ENDPOINT` | `http://localhost:9000` | S3-compatible endpoint |
| `R2_ACCESS_KEY` | `minioadmin` | Storage access key |
| `R2_SECRET_KEY` | `minioadmin` | Storage secret key |
| `R2_BUCKET` | `kyc-sentinel-media` | Bucket name |
| `USE_MODAL` | `false` | Enable Modal GPU backend |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins |
| `BACKEND_API_KEY` | `` | Optional API auth key |

## Code Conventions

### Backend

- **Async-first**: All database and I/O operations use `async`/`await`
- **Lazy loading**: ML models load on first use to avoid slow startup
- **Singleton pattern**: Detection analyzers are cached (model loading is expensive)
- **Protocol-based backends**: `ProcessingBackend` protocol allows swapping `LocalBackend` ↔ `ModalBackend`
- **Pydantic schemas**: Separate from SQLAlchemy models, in `/app/schemas/`

### Frontend

- **App Router**: File-based routing in `/app`
- **API proxy**: All backend calls go through `/api/[...path]/route.ts` to handle auth
- **React Query**: Server state with automatic caching/invalidation
- **shadcn/ui**: Components in `/components/ui/` following Radix patterns

### Database

- **UUID primary keys**: All tables use `uuid.uuid4`
- **pgvector embeddings**: 512-dimensional face embeddings for similarity search
- **Cascade deletes**: Session deletion removes all related results/reasons

## Key Files

| Path | Purpose |
|------|---------|
| `backend/app/main.py` | FastAPI app entry point |
| `backend/app/config.py` | Pydantic settings from env |
| `backend/app/services/processing.py` | Processing backend abstraction |
| `backend/app/detection/face_analyzer.py` | InsightFace wrapper |
| `backend/app/detection/document_analyzer.py` | PaddleOCR wrapper |
| `backend/app/detection/pad_heuristics.py` | OpenCV-based PAD signals |
| `backend/app/services/scoring.py` | Risk score computation |
| `backend/app/services/reason_codes.py` | Reason code definitions |
| `frontend/app/api/[...path]/route.ts` | Backend proxy |
| `frontend/lib/api.ts` | API client functions |
| `simulator/generator.py` | Synthetic artifact generator |
| `simulator/base_images.py` | Synthetic selfie/ID document image generation |

## Reason Code System

Detection modules emit explainable reason codes:

| Category | Code | Severity | Trigger |
|----------|------|----------|---------|
| Face | `FACE_MISMATCH` | high | Similarity < 0.45 |
| Face | `FACE_NOT_DETECTED_SELFIE` | high | No face in selfie |
| Face | `FACE_NOT_DETECTED_ID` | high | No face in ID |
| Face | `MULTIPLE_FACES_SELFIE` | warn | Multiple faces detected |
| PAD | `PAD_SUSPECT_REPLAY` | high | Color temp variance high |
| PAD | `PAD_SCREEN_ARTIFACTS` | warn | Moiré patterns detected |
| PAD | `PAD_LOW_MOTION_ENTROPY` | warn | Insufficient movement |
| PAD | `PAD_FRAME_STUTTER` | warn | Duplicate frames |
| PAD | `PAD_INJECTION_ARTIFACTS` | high | Unnatural sharpness boundaries |
| PAD | `PAD_FACE_BOUNDARY_MISMATCH` | high | Face boundary inconsistent with background |
| Doc | `DOC_TEMPLATE_MISMATCH` | warn | Document structure doesn't match ID card pattern |
| Doc | `DOC_OCR_LOW_CONFIDENCE` | warn | OCR confidence < 0.5 |
| Doc | `DOC_FONT_INCONSISTENT` | warn | Font height variance > 0.5 |
| Doc | `DOC_TEXT_MISALIGNED` | warn | Alignment anomalies |
| Doc | `DOC_EDGE_ARTIFACTS` | warn | Rectangular edge patterns |

### Scoring Logic

```python
risk_score = 0.45 * (1 - face_similarity) + 0.35 * pad_score + 0.20 * doc_score
# Scaled to 0-100

if any high severity reasons: decision = "fail"
elif risk_score >= 70: decision = "fail"
elif risk_score >= 40: decision = "review"
else: decision = "pass"
```

## Testing

```bash
cd backend
pytest                      # Run all tests
pytest -v                   # Verbose
pytest --cov=app            # With coverage
pytest tests/test_detection.py  # Specific file
```

Tests use aiosqlite for in-memory async SQLite testing.

## Database Migrations

```bash
cd backend

# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1
```

## Common Tasks

### Adding a New Reason Code

1. Add enum value to `ReasonCode` in `backend/app/services/reason_codes.py`
2. Add message template to `REASON_MESSAGES`
3. Set severity in `get_reason_severity()`
4. Update `docs/reason-codes.md`

### Adding a New API Endpoint

1. Create/update router in `backend/app/api/`
2. Add Pydantic schemas to `backend/app/schemas/`
3. Include router in `backend/app/main.py`
4. Update `frontend/lib/api.ts` with client function

### Adding a New Attack Family

1. Create artifact module in `simulator/artifacts/`
2. Add case to `ArtifactGenerator.generate()` in `simulator/generator.py`
3. Add detection logic in appropriate analyzer
4. Create corresponding reason codes

## Future Upgrades & TODOs

### Recently Completed

1. ~~**Complete Simulator Integration**~~: `generate_synthetic_session()` now generates synthetic selfie/ID images via `simulator/base_images.py`, applies attack artifacts using `ArtifactGenerator`, uploads to storage, and runs the detection pipeline. Works for both `LocalBackend` and `ModalBackend`.

2. ~~**Video PAD for Modal Backend**~~: `ModalBackend.process_session()` now downloads extracted frames from storage and runs PAD analysis locally (CPU-based). PAD reason codes and frame metrics are properly saved.

3. ~~**Template Matching**~~: `DocumentAnalyzer` now computes `template_match_score` based on structural analysis: aspect ratio, card edge detection, text distribution, and photo region detection. Emits `DOC_TEMPLATE_MISMATCH` when score < 0.5.

4. ~~**Face Boundary Mismatch Detection**~~: `PADAnalyzer` now implements `_detect_face_boundary_mismatch()` using skin segmentation, boundary extraction, and analysis of color discontinuity, texture mismatch, and edge sharpness. Emits `PAD_FACE_BOUNDARY_MISMATCH` for suspected face swaps.

### Medium Priority

1. **Worker Service**: Implement durable job queue processing in `backend/app/worker.py` for Railway worker deployment.

2. **Batch Processing**: Add batch upload and simulation endpoints for bulk testing.

3. **Webhook Notifications**: Add webhook support for processing completion events.

4. **Model Versioning**: Track which ML model versions produced each result for reproducibility.

### Low Priority / Nice to Have

5. **Real-time Processing Progress**: WebSocket updates for processing status.

6. **Export/Import Sessions**: CSV/JSON export of session data for external analysis.

7. **A/B Testing Framework**: Compare different detection rule versions.

8. **Rate Limiting**: Add rate limiting to public endpoints.

9. **Audit Logging**: Comprehensive audit trail for compliance.

## Deployment

See `docs/deployment.md` for full details:

- **Frontend**: Vercel (auto-deploy from main branch)
- **Backend API**: Railway service with `scripts/railway_start.sh`
- **Worker**: Railway service with `scripts/railway_worker.sh`
- **Database**: Railway Postgres with pgvector
- **Storage**: Cloudflare R2 with CORS for presigned uploads
- **GPU Processing**: Modal with `r2-credentials` secret

## Dependencies to Watch

| Dependency | Note |
|------------|------|
| PaddleOCR | v2.8+ uses new `predict()` API. Code updated for this. |
| InsightFace | Uses ONNX Runtime. GPU requires `onnxruntime-gpu`. |
| pgvector | Requires PostgreSQL extension enabled in production. |
| Next.js | Currently on v16.x. App Router is stable. |
| Modal | API changes frequently. Pin version carefully. |

## Troubleshooting

### "No module named 'paddleocr'"
PaddlePaddle must be installed first: `pip install paddlepaddle` then `pip install paddleocr`

### "CUDAExecutionProvider not found"
Install GPU version: `pip install onnxruntime-gpu` (requires CUDA)

### MinIO bucket doesn't exist
Create manually via MinIO Console at `http://localhost:9001` or use mc CLI.

### Migrations fail with "relation does not exist"
Ensure pgvector extension is enabled: `CREATE EXTENSION IF NOT EXISTS vector;`

### Face detection returns empty
InsightFace downloads models on first run (~500MB). Check network/disk space.
