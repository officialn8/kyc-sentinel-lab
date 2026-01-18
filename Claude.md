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
                           │ WebSocket at /ws/sessions/{id}
┌──────────────────────────▼──────────────────────────────────────┐
│                         Backend                                  │
│                        (FastAPI)                                 │
├─────────────────────────────────────────────────────────────────┤
│  API Layer  │  Services  │  Detection  │  Processing Backend    │
│             │  Webhooks  │             │  (emits WS progress)   │
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
| cmdk | 1.x | Command palette (⌘K) |
| next-themes | 0.4.x | Dark/light mode with system preference |
| @tanstack/react-virtual | 3.x | List virtualization for large tables |
| framer-motion | 11.x | Animations and micro-interactions |
| @vercel/analytics | - | Page view tracking (auto-enabled on Vercel) |

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
| geoip2 | 4.8.x | MaxMind IP geolocation (optional) |
| redis | 5.0.x | Distributed rate limiting (optional) |

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
| `USE_WORKER` | `false` | Enable durable job queue (requires separate worker service) |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins |
| `BACKEND_API_KEY` | `` | Optional API auth key |
| `SCORING_PROFILE` | `default` | Scoring profile (default, fintech_high_risk, crypto_exchange, social_verification) |
| `PAD_PROFILE` | `default` | PAD profile (default, strict, lenient) |
| `VELOCITY_DEVICE_1H_LIMIT` | `3` | Max submissions per device per hour |
| `VELOCITY_DEVICE_24H_LIMIT` | `10` | Max submissions per device per 24 hours |
| `VELOCITY_IP_1H_LIMIT` | `5` | Max submissions per IP per hour |
| `VELOCITY_IP_24H_LIMIT` | `20` | Max submissions per IP per 24 hours |
| `AUTH_DISABLED` | `false` | **SECURITY**: Must be `true` to disable auth (explicit opt-out) |
| `TRUSTED_PROXY_CIDRS` | `` | Comma-separated CIDRs for trusted proxies (e.g., `10.0.0.0/8,172.16.0.0/12`) |
| `MAX_UPLOAD_SIZE_BYTES` | `52428800` | Maximum upload file size (50MB default) |
| `GEOIP_DATABASE_PATH` | `` | Path to MaxMind GeoLite2-Country.mmdb (optional, for IP geolocation) |
| `REDIS_URL` | `` | Redis URL for distributed rate limiting (e.g., `redis://localhost:6379/0`) |

## Security Architecture

### Authentication (Fail-Closed by Default)

**CRITICAL**: The API refuses to start without proper auth configuration. This prevents accidental exposure.

```python
# config.py - Startup validation
@model_validator(mode="after")
def _validate_auth_config(self) -> "Settings":
    if self.auth_disabled:
        return self  # Explicit opt-out

    if not has_api_key and not has_basic:
        raise ValueError("SECURITY: No authentication configured...")
```

**Options:**
1. Set `BACKEND_API_KEY` for API key auth (header: `X-API-Key`)
2. Set `BASIC_AUTH_USERNAME` + `BASIC_AUTH_PASSWORD` for Basic Auth
3. Set `AUTH_DISABLED=true` for development only

### Trusted Proxy Configuration

**Problem**: X-Forwarded-For can be spoofed if attackers bypass your proxy.

**Solution**: Only trust proxy headers from known networks.

```bash
# Railway/Render (private network)
TRUSTED_PROXY_CIDRS=10.0.0.0/8,172.16.0.0/12,192.168.0.0/16

# Cloudflare (use their published ranges)
TRUSTED_PROXY_CIDRS=173.245.48.0/20,103.21.244.0/22,...

# Local development
TRUSTED_PROXY_CIDRS=127.0.0.0/8,::1/128
```

If `TRUSTED_PROXY_CIDRS` is empty (default), the system always uses the direct connection IP, which is the safest default.

### Presigned URL Security

**Protections implemented:**

1. **Key Validation**: Only allows keys matching `sessions/{uuid}/[selfie|id|frames/NNNN].[ext]`
2. **Path Traversal Blocked**: Rejects `..`, absolute paths, backslashes
3. **Session Ownership**: Verifies session exists and is in `pending` state
4. **Content-Type Enforcement**: Signed URLs require matching Content-Type header

```python
# upload.py - Key validation pattern
_ALLOWED_KEY_PATTERN = re.compile(
    r"^sessions/(?P<session_id>[0-9a-f-]{36})/"
    r"(selfie|id|frames/\d{4})\.(jpg|jpeg|png|webp|mp4|mov|webm)$"
)
```

### Fraud Detection Security

**Client-Provided (Untrusted):**
- `device_fingerprint` - Can be spoofed
- `device_timezone` - Can be spoofed
- `ip_country` - Use for display only, not fraud decisions

**Server-Captured (Trusted):**
- `client_ip` - Captured via `extract_client_ip()` with trusted proxy support
- Document country from MRZ/OCR

### IP Geolocation (Server-Side Enrichment)

**Problem**: Client-provided `ip_country` can be spoofed and is untrusted.

**Solution**: Server-side IP geolocation using MaxMind GeoIP2.

```bash
# Download free GeoLite2-Country database
# https://dev.maxmind.com/geoip/geolite2-free-geolocation-data

# Configure path (or use common system locations)
GEOIP_DATABASE_PATH=/path/to/GeoLite2-Country.mmdb
```

**How it works:**
1. `create_session()` captures `client_ip` server-side
2. `lookup_ip_country()` queries MaxMind database
3. Server-enriched country is stored (falls back to client-provided if GeoIP unavailable)
4. Fraud detection uses trusted server-side value

**Files:**
- `backend/app/services/geolocation.py` - GeoIP service
- `backend/app/api/sessions.py` - Integration in session creation

### Rate Limiting

**Backends available:**

1. **In-Memory** (default): Single-instance only, bounded memory with eviction
2. **Redis** (production): Distributed across multiple workers/instances

```bash
# Enable Redis rate limiting
REDIS_URL=redis://localhost:6379/0
```

**Automatic fallback**: If Redis is configured but unavailable, falls back to in-memory.

**Algorithm**: Sliding window using Redis sorted sets with TTL.

**Files:**
- `backend/app/api/security.py` - Rate limiter implementation

### Risk Score Calculation

Uses `math.ceil()` for risk-increasing bias:
```python
# scoring.py - Never under-report risk
risk_score = math.ceil(raw_score * 100)  # 39.01 → 40, not 39
```

### Session Finalize Idempotency

Uses `SELECT FOR UPDATE` to prevent race conditions:
```python
query = (
    select(KYCSession)
    .where(KYCSession.id == session_id)
    .with_for_update(nowait=False)
)
```

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
- **Error handling**: User-friendly error messages via `lib/errors.ts`, error boundaries, inline retry states
- **Mobile-first**: Responsive layouts with mobile nav drawer, command palette, bottom-safe padding

### Database

- **UUID primary keys**: All tables use `uuid.uuid4`
- **pgvector embeddings**: 512-dimensional face embeddings for similarity search
- **Cascade deletes**: Session deletion removes all related results/reasons

## Key Files

| Path | Purpose |
|------|---------|
| `backend/app/main.py` | FastAPI app entry point |
| `backend/app/config.py` | Pydantic settings from env |
| `backend/app/constants.py` | Documented magic numbers with rationale |
| `backend/app/models/enums.py` | SessionStatus, AttackFamily, Decision enums |
| `backend/app/services/processing/` | **Processing backend package** (modular) |
| `backend/app/services/processing/protocol.py` | ProcessingBackend Protocol, dataclasses |
| `backend/app/services/processing/local.py` | LocalBackend (CPU/dev mode) |
| `backend/app/services/processing/modal.py` | ModalBackend (GPU/production) |
| `backend/app/services/processing/helpers.py` | Shared utilities (decode_image, build_reasons) |
| `backend/app/detection/face_analyzer.py` | InsightFace wrapper |
| `backend/app/detection/document_analyzer.py` | PaddleOCR wrapper |
| `backend/app/detection/pad_heuristics.py` | OpenCV-based PAD signals |
| `backend/app/services/scoring.py` | Risk score computation (Decimal precision) |
| `backend/app/services/reason_codes.py` | Reason code definitions |
| `backend/app/services/fraud_detection.py` | Velocity rules & geo anomaly detection |
| `backend/app/services/geolocation.py` | MaxMind GeoIP2 IP-to-country lookup |
| `backend/app/services/storage.py` | S3/R2 storage with connection pooling |
| `backend/app/api/security.py` | Auth, rate limiting (in-memory + Redis) |
| `backend/app/api/websocket.py` | WebSocket endpoint for real-time session progress |
| `backend/app/api/webhooks.py` | Webhook CRUD API endpoints |
| `backend/app/services/webhook_service.py` | Webhook dispatch with retry logic |
| `backend/app/models/webhook.py` | Webhook SQLAlchemy model |
| `backend/app/utils/background_tasks.py` | Background task error wrapper |
| `frontend/app/api/[...path]/route.ts` | Backend proxy |
| `frontend/lib/api.ts` | API client functions |
| `frontend/lib/errors.ts` | Error parsing and user-friendly messages |
| `simulator/generator.py` | Synthetic artifact generator |
| `simulator/base_images.py` | Synthetic selfie/ID document image generation |

## Frontend Components

### UI Components (`/frontend/components/ui/`)
| Component | Purpose |
|-----------|---------|
| `animations.tsx` | Framer Motion animations: SuccessAnimation, ProgressRing, AnimatedNumber, FadeIn, ScaleIn, Pulse, ProcessingSpinner |
| `sheet.tsx` | Slide-out drawer (Radix Dialog-based) |
| `command.tsx` | Command palette using cmdk |
| `skeleton.tsx` | Shimmer loading skeletons with variants |
| `empty-state.tsx` | Illustrated empty states with CTAs |
| `lightbox.tsx` | Full-screen image viewer with zoom/rotate |
| `error-boundary.tsx` | Error boundary + inline error states |
| `optimized-image.tsx` | next/image wrapper with blur placeholder, loading states |

### Layout Components (`/frontend/components/layout/`)
| Component | Purpose |
|-----------|---------|
| `navbar.tsx` | Top navigation with mobile menu trigger + command palette |
| `sidebar.tsx` | Desktop sidebar navigation with theme toggle |
| `mobile-nav.tsx` | Mobile slide-out navigation drawer |
| `command-palette.tsx` | ⌘K command palette with navigation |
| `connection-status.tsx` | Offline/online status indicator |
| `getting-started.tsx` | Onboarding wizard card |
| `theme-toggle.tsx` | Dark/light/system theme toggle (icon, compact, select variants) |

### Session Components (`/frontend/components/sessions/`)
| Component | Purpose |
|-----------|---------|
| `face-comparison.tsx` | Side-by-side face comparison with similarity |
| `evidence-viewer.tsx` | Structured evidence display (scores as bars) |
| `processing-timeline.tsx` | Processing step timeline/stepper |
| `session-card.tsx` | Session list item card |
| `media-preview.tsx` | Image preview with loading states |
| `reason-code-badge.tsx` | Reason code display badge |

### Metrics Components (`/frontend/components/metrics/`)
| Component | Purpose |
|-----------|---------|
| `confusion-matrix.tsx` | Attack type vs decision matrix |

## Reason Code System

Detection modules emit explainable reason codes:

| Category | Code | Severity | Trigger |
|----------|------|----------|---------|
| Face | `FACE_MISMATCH` | high | Similarity < threshold (adaptive 0.35-0.50) |
| Face | `FACE_NOT_DETECTED_SELFIE` | high | No face in selfie |
| Face | `FACE_NOT_DETECTED_ID` | high | No face in ID |
| Face | `MULTIPLE_FACES_SELFIE` | warn | Multiple faces detected |
| Face | `FACE_LOW_QUALITY_SELFIE` | warn | Selfie quality score < 0.4 or has critical issues |
| Face | `FACE_LOW_QUALITY_ID` | warn | ID photo quality score < 0.4 or has critical issues |
| Face | `FACE_EXTREME_POSE` | warn | Pose deviation > 30 degrees from frontal |
| PAD | `PAD_SUSPECT_REPLAY` | high | Color temp variance high |
| PAD | `PAD_SCREEN_ARTIFACTS` | warn | Moiré patterns detected |
| PAD | `PAD_LOW_MOTION_ENTROPY` | warn | Insufficient movement |
| PAD | `PAD_FRAME_STUTTER` | warn | Duplicate frames |
| PAD | `PAD_INJECTION_ARTIFACTS` | high | Unnatural sharpness boundaries |
| PAD | `PAD_FACE_BOUNDARY_MISMATCH` | high | Face boundary inconsistent with background |
| PAD | `PAD_NO_BLINK_DETECTED` | warn | No natural blinks in video (>30 frames) |
| PAD | `PAD_PRINTED_TEXTURE` | high | LBP texture analysis indicates printed photo |
| PAD | `PAD_FLAT_DEPTH` | high | Depth analysis indicates flat surface |
| Doc | `DOC_TEMPLATE_MISMATCH` | warn | Document structure doesn't match ID card pattern |
| Doc | `DOC_OCR_LOW_CONFIDENCE` | warn | OCR confidence < 0.5 |
| Doc | `DOC_FONT_INCONSISTENT` | warn | Font height variance > 0.5 |
| Doc | `DOC_TEXT_MISALIGNED` | warn | Alignment anomalies |
| Doc | `DOC_EDGE_ARTIFACTS` | warn | Rectangular edge patterns |
| Doc | `DOC_MRZ_INVALID` | high | MRZ check digits fail validation |
| Doc | `DOC_MRZ_MISMATCH` | warn | MRZ data doesn't match OCR text |
| Doc | `DOC_METADATA_SUSPICIOUS` | warn | EXIF shows editing software or date mismatch |
| Doc | `DOC_BARCODE_MISMATCH` | warn | Barcode data doesn't match document text |
| Doc | `DOC_ELA_MANIPULATION` | warn | Error Level Analysis detects image editing |
| Doc | `DOC_COPY_MOVE_DETECTED` | high | Copy-move forgery detected |
| Face | `FACE_AGE_DISCREPANCY` | warn | Apparent age differs significantly from ID DOB |
| PAD | `PAD_VIRTUAL_CAMERA` | high | Virtual camera software detected (OBS, ManyCam, etc.) |
| PAD | `PAD_LIP_SYNC_MISMATCH` | high | Lip movement doesn't match audio |
| Fraud | `FRAUD_HIGH_VELOCITY` | warn | High submission rate from device/IP |
| Fraud | `FRAUD_GEO_MISMATCH` | warn | Document country differs from device location |
| Fraud | `FRAUD_SIMILAR_FACE` | warn | Similar face found in other sessions |

### Scoring Logic

Scoring uses configurable profiles defined in `backend/app/services/scoring.py`:

```python
# Available profiles (set via SCORING_PROFILE env var)
SCORING_PROFILES = {
    "default":             ScoringConfig(0.45, 0.35, 0.20, fail=70, review=40),
    "fintech_high_risk":   ScoringConfig(0.50, 0.35, 0.15, fail=60, review=35),
    "crypto_exchange":     ScoringConfig(0.40, 0.40, 0.20, fail=55, review=30),
    "social_verification": ScoringConfig(0.60, 0.25, 0.15, fail=70, review=45),
}

# Default scoring formula
risk_score = face_weight * (1 - face_similarity) + pad_weight * pad_score + doc_weight * doc_score
# Scaled to 0-100

if any high severity reasons: decision = "fail"
elif risk_score >= fail_threshold: decision = "fail"
elif risk_score >= review_threshold: decision = "review"
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

5. ~~**Mobile Navigation & Command Palette**~~: Implemented slide-out mobile drawer using Radix Sheet, command palette (⌘K) with cmdk for quick navigation across all breakpoints.

6. ~~**Empty States & Onboarding**~~: Added illustrated empty states with CTAs for sessions/metrics pages. Getting Started onboarding card shows on dashboard when no sessions exist.

7. ~~**Skeleton Loading States**~~: Created reusable skeleton components with shimmer animation. Variants for stat cards, session rows, tables. Applied throughout dashboard, sessions, and metrics pages.

8. ~~**Error Handling & Feedback**~~: Added `lib/errors.ts` for user-friendly error messages, `ErrorBoundary` component with recovery, `InlineError` for section-level errors with retry, `ConnectionStatus` for offline detection.

9. ~~**Confusion Matrix Visualization**~~: The existing `ConfusionMatrix` component is now integrated into the metrics page, showing actual attack types vs predicted decisions.

10. ~~**Chart Drill-down Navigation**~~: Pie chart and bar chart on metrics page now support click-to-filter navigation to sessions page.

11. ~~**Session Detail Enhancements**~~: Added `Lightbox` for full-screen image viewing with zoom/rotate/download, `FaceComparison` for side-by-side face matching with similarity score, `EvidenceViewer` for structured evidence display (scores as progress bars), `ProcessingTimeline` showing step-by-step processing status.

12. ~~**Form UX Improvements**~~: Upload page now has client-side validation with real-time feedback (file size, format, dimensions), upload progress bars per file, auto-detection of device info from `navigator.userAgent`, and image quality checks that warn if images are blurry or too small.

13. ~~**Dark/Light Mode Toggle**~~: Added `next-themes` integration with system preference detection and manual override. Theme toggle in navbar and sidebar footer. CSS variables updated for proper light mode support.

14. ~~**Performance & Polish**~~: Switched to `next/image` with blur placeholders via `OptimizedImage` component. Added virtualization with `@tanstack/react-virtual` for session tables with 50+ rows. Session rows prefetch on hover via `router.prefetch()`. Optimistic updates when generating synthetic sessions.

15. ~~**Micro-interactions & Visual Feedback**~~: Added Framer Motion for animations. Cards now have `glass-interactive` class with hover scale/shadow/glow effects. Created `SuccessAnimation` component with checkmark + ripple effect shown after upload completion. Added `ProgressRing` animated component for upload progress. Button press animations, input focus glow, and smooth transitions throughout.

16. ~~**Search Functionality**~~: Sessions page now has a search input with 300ms debounce that filters by session ID, attack family, source, or status. Command palette (⌘K) enhanced to search sessions in real-time as you type, showing up to 5 results with ability to view all matches. Backend `GET /sessions` endpoint supports `search` query parameter.

### High Priority - Phase 1: Detection Enhancements (COMPLETED)

17. ~~**Liveness Detection Enhancements**~~: Added blink detection using Eye Aspect Ratio (EAR) from 68-point facial landmarks. Implemented LBP (Local Binary Pattern) texture analysis to detect printed photos vs real skin. Added `BlinkAnalysis` dataclass for aggregating blink patterns across frames. New reason codes: `PAD_NO_BLINK_DETECTED`, `PAD_PRINTED_TEXTURE`, `PAD_FLAT_DEPTH`. PAD analyzer now computes per-frame EAR and texture scores.

18. ~~**Document Verification Improvements**~~: Added MRZ (Machine Readable Zone) validation with ICAO 9303 checksum verification for TD1/TD2/TD3 formats. Implemented EXIF metadata analysis to detect editing software (Photoshop, GIMP, etc.) and suspicious metadata patterns. New dataclasses: `MRZValidation`, `EXIFAnalysis`. New reason codes: `DOC_MRZ_INVALID`, `DOC_MRZ_MISMATCH`, `DOC_BARCODE_MISMATCH`. Added `analyze_with_bytes()` method for enhanced EXIF analysis from original image bytes.

19. ~~**Face Analysis Accuracy**~~: Implemented adaptive similarity thresholds based on image quality scores. Added `FaceQuality` dataclass with comprehensive metrics: sharpness, lighting, pose deviation, face size ratio. Quality scoring uses Laplacian variance for sharpness, histogram analysis for lighting, and landmark geometry for pose estimation. Thresholds range from 0.35 (poor quality) to 0.50 (high quality). New reason codes: `FACE_LOW_QUALITY_SELFIE`, `FACE_LOW_QUALITY_ID`, `FACE_EXTREME_POSE`.

20. ~~**Scoring Service Updates**~~: Enhanced `compute_risk_score()` to incorporate liveness signals and quality adjustments. Added `compute_liveness_score()` for combining blink detection, texture analysis, and motion entropy. Critical liveness failures (`PAD_PRINTED_TEXTURE`, `PAD_FLAT_DEPTH`) now boost risk score. Document integrity failures (`DOC_MRZ_INVALID`) also increase risk.

### Phase 2: Production Deployment (COMPLETED)

21. ~~**Modal GPU Deployment**~~: Deployed Modal app with three GPU-accelerated functions (`extract_frames`, `analyze_face`, `analyze_document`) running on T4 instances. Created `r2-credentials` Modal secret for direct R2 storage access. Fixed function invocation to use `modal.Function.from_name()` for calling deployed functions from Railway. Increased memory allocation to 4GB for PaddleOCR models.

22. ~~**Railway Backend Deployment**~~: Deployed FastAPI backend to Railway with proper environment configuration. Created `Dockerfile` with OpenCV dependencies (`libgl1`, `libglib2.0-0`, etc.) and build tools (`build-essential`, `g++`) for compiling InsightFace Cython extensions.

23. ~~**Railway Worker Service**~~: Set up separate Railway service for durable job queue processing. Uses same Dockerfile with different start command (`python -m app.worker`). Configured with `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` for Modal API access.

24. ~~**Cloudflare R2 CORS Configuration**~~: Configured CORS policy on R2 bucket to allow presigned uploads from Vercel frontend origin. Enables direct browser-to-R2 file uploads.

25. ~~**Vercel Frontend Deployment**~~: Deployed Next.js frontend to Vercel. Fixed Suspense boundary issue in sessions page (wrapped `useSearchParams` in Suspense). Configured `BACKEND_URL` environment variable pointing to Railway backend.

26. ~~**Vercel Analytics**~~: Added `@vercel/analytics` for automatic page view tracking. Analytics component added to root layout.

### Phase 2: Risk Scoring Calibration, Fraud Pattern Intelligence, Detection Improvements (COMPLETED)

27. ~~**Configurable Risk Scoring**~~: Added `ScoringConfig` and `ScoringProfile` enum with 4 preset profiles (DEFAULT, FINTECH_HIGH_RISK, CRYPTO_EXCHANGE, SOCIAL_VERIFICATION). Each profile has configurable weights (face, PAD, doc) and thresholds (fail, review). Profiles selectable via `SCORING_PROFILE` environment variable.

28. ~~**Enhanced Similar Face Search**~~: Updated `GET /sessions/{id}/similar` endpoint with `threshold` parameter (0-1, default 0.7) for fraud ring detection. Response now includes similarity scores and distance for each match.

29. ~~**Virtual Camera Detection**~~: Added detection for virtual camera software (OBS, ManyCam, Snap Camera, XSplit, CamTwist, etc.) by analyzing video metadata. New `detect_virtual_camera()` method in PAD analyzer. Emits `PAD_VIRTUAL_CAMERA` reason code.

30. ~~**Audio Liveness (Lip Sync)**~~: Added `analyze_lip_sync()` to PAD analyzer for video submissions with audio. Correlates lip aperture motion with audio energy. Poor correlation suggests pre-recorded video or audio injection. Emits `PAD_LIP_SYNC_MISMATCH`.

31. ~~**Fraud Detection Service**~~: Created `backend/app/services/fraud_detection.py` with velocity rules (`check_velocity()`) and geographic anomaly detection (`check_geo_anomaly()`). Tracks submissions per device/IP within configurable time windows.

32. ~~**Database Migration for Fraud Detection**~~: Added `003_fraud_detection.py` migration with new columns: `device_fingerprint`, `client_ip`, `device_timezone`. Created indexes for efficient velocity queries.

33. ~~**Multi-language OCR**~~: Extended `DocumentAnalyzer` to support 50+ languages via `lang` parameter. OCR instances cached per language. Supports Latin, Asian, Cyrillic, Arabic scripts.

34. ~~**Error Level Analysis (ELA)**~~: Added `compute_ela()` and `detect_ela_manipulation()` to `DocumentAnalyzer` for detecting Photoshopped regions. Re-compresses image and compares error levels to find manipulated areas. Emits `DOC_ELA_MANIPULATION`.

35. ~~**Age Estimation**~~: Added `AgeEstimation` dataclass and age extraction from InsightFace. New `check_age_discrepancy()` method compares estimated age with ID DOB. Emits `FACE_AGE_DISCREPANCY` when difference exceeds tolerance (default 10 years).

36. ~~**New Reason Codes**~~: Added 9 new reason codes for Phase 2 features: `FACE_AGE_DISCREPANCY`, `PAD_VIRTUAL_CAMERA`, `PAD_LIP_SYNC_MISMATCH`, `DOC_ELA_MANIPULATION`, `DOC_COPY_MOVE_DETECTED`, `FRAUD_HIGH_VELOCITY`, `FRAUD_GEO_MISMATCH`, `FRAUD_SIMILAR_FACE`.

### Phase 3: Security Hardening (COMPLETED)

37. ~~**Auth Fail-Closed at Startup**~~: Added `@model_validator` in `config.py` that raises `ValueError` during settings initialization if auth is misconfigured. App refuses to boot without proper auth config. Removed runtime 500 error fallback.

38. ~~**Presigned URL Session Ownership**~~: The `/presigned` endpoint now extracts session UUID from key, verifies session exists in database, and checks session is in `pending` state. Prevents writes to arbitrary UUIDs or already-processed sessions.

39. ~~**Trusted Proxy Configuration**~~: Added `TRUSTED_PROXY_CIDRS` setting. `X-Forwarded-For` is only trusted if direct connection comes from configured trusted network. Empty = always use direct IP (safest default). Shared `extract_client_ip()` function used by both rate limiter and session creation.

40. ~~**Bounded Parallel Downloads**~~: Added `asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS=10)` to Modal backend frame downloads. Prevents memory exhaustion and storage saturation under load.

41. ~~**Risk Score Ceiling Bias**~~: Changed from `round()` to `math.ceil()` for risk score calculation. Explicit policy: never under-report risk. `39.01 → 40`, not `39`.

42. ~~**Finalize Idempotency with Row Locking**~~: Added `SELECT FOR UPDATE` query in finalize endpoint. Locks row before checking status, uses HTTP 409 for concurrent access, returns success for already-completed sessions.

43. ~~**Rate Limiter Improvements**~~: Added periodic eviction via `_cleanup_stale_buckets()`, emergency eviction at `_RATE_BUCKET_MAX_KEYS=10000`, proper X-Forwarded-For handling with trusted proxy check, `Retry-After` header on 429 responses.

44. ~~**Storage Key Injection Prevention**~~: Regex validation with named capture group for session UUID, path traversal blocking (`..`, `/`, `\`), content-type enforcement in presigned URLs.

45. ~~**Server-Side IP Capture**~~: `client_ip` now captured server-side in `create_session()` using trusted proxy config. Fraud detection uses server-captured IP for velocity checks.

46. ~~**Fraud Detection Wiring**~~: `compute_fraud_signals()` now properly receives `device_country` parameter. Integrated into `process_session()` pipeline. Added documentation about server-side IP geolocation enrichment for production.

47. ~~**Search Performance Index**~~: Added `005_search_performance.py` migration with `pg_trgm` extension and GIN indexes on session ID, attack_family, status, source columns. Partial indexes for common status values.

48. ~~**Removed sys.path Hacks**~~: Fixed simulator package to use relative imports. Removed all `sys.path` manipulation from `processing.py`.

---

### Medium Priority

1. ~~**Worker Service**~~: Implemented durable job queue processing in `backend/app/worker.py` for Railway worker deployment.

2. **Batch Processing**: Add batch upload and simulation endpoints for bulk testing.

3. ~~**Webhook Notifications**~~: Added webhook support for processing completion events. CRUD API at `/api/webhooks`, dispatch service with 3 retries + exponential backoff (1s, 2s, 4s). Auto-disables after 10 consecutive failures. Events: `session.completed`, `session.failed`. Files: `backend/app/api/webhooks.py`, `backend/app/services/webhook_service.py`, `backend/app/models/webhook.py`, `backend/alembic/versions/008_webhooks.py`.

4. ~~**Model Versioning**~~: Added `version` property to `FaceAnalyzer` (returns `insightface-{model_pack}`) and `DocumentAnalyzer` (returns `paddleocr-{lang}`). Processing pipeline populates `model_version` and `rules_version` fields in `KYCResult` for reproducibility tracking.

### Low Priority / Nice to Have

5. ~~**Real-time Processing Progress**~~: Added WebSocket endpoint at `/ws/sessions/{session_id}` for real-time processing updates. Events: `started`, `progress`, `completed`, `failed`. Progress events include step name (face_analysis, document_analysis, pad_analysis, scoring) and percentage. Connection manager in `backend/app/api/websocket.py`. Note: Single-instance implementation; use Redis pub/sub for multi-instance deployments.

6. **Export/Import Sessions**: CSV/JSON export of session data for external analysis.

7. **A/B Testing Framework**: Compare different detection rule versions.

8. ~~**Rate Limiting**~~: Implemented per-process rate limiting with memory eviction and Redis-based distributed rate limiting for multi-instance production.

9. **Audit Logging**: Comprehensive audit trail for compliance.

### Phase 4: Production Infrastructure (COMPLETED)

49. ~~**MaxMind GeoIP2 Integration**~~: Added server-side IP geolocation via `backend/app/services/geolocation.py`. Session creation now enriches `ip_country` using server-captured `client_ip`. Falls back to client-provided value if GeoIP unavailable. Configuration via `GEOIP_DATABASE_PATH` env var.

50. ~~**Redis Rate Limiting**~~: Added distributed rate limiting support in `backend/app/api/security.py`. Uses Redis sorted sets with sliding window algorithm. Automatically falls back to in-memory if Redis unavailable. Configuration via `REDIS_URL` env var.

51. ~~**pgvector Extension Check**~~: Added `006_ensure_pgvector.py` migration that verifies pgvector extension availability, creates it if possible, checks face_embedding column exists, and creates appropriate index (IVFFlat for large datasets, HNSW for small). Provides helpful error messages for Railway/Supabase/self-hosted PostgreSQL.

## Deployment

See `docs/deployment.md` for full details:

- **Frontend**: Vercel (auto-deploy from main branch, `BACKEND_URL` env var points to Railway)
- **Backend API**: Railway service using `Dockerfile` builder, start command: `./scripts/railway_start.sh`
- **Worker**: Railway service using same `Dockerfile`, start command: `python -m app.worker`
- **Database**: Railway Postgres with pgvector extension
- **Storage**: Cloudflare R2 with CORS configured for Vercel origin
- **GPU Processing**: Modal app `kyc-sentinel` with `r2-credentials` secret (T4 GPUs, 4GB memory)

### Railway Environment Variables
| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | Railway Postgres connection string |
| `R2_ENDPOINT` | Cloudflare R2 endpoint URL |
| `R2_ACCESS_KEY` | R2 access key |
| `R2_SECRET_KEY` | R2 secret key |
| `R2_BUCKET` | R2 bucket name |
| `USE_MODAL` | Set to `true` for GPU processing |
| `USE_WORKER` | Set to `true` for durable job queue processing |
| `MODAL_TOKEN_ID` | Modal API token ID |
| `MODAL_TOKEN_SECRET` | Modal API token secret |
| `CORS_ORIGINS` | Vercel frontend URL |
| `SCORING_PROFILE` | Risk scoring profile (default, fintech_high_risk, crypto_exchange, social_verification) |
| `PAD_PROFILE` | PAD detection profile (default, strict, lenient) |
| `BACKEND_API_KEY` | **REQUIRED**: API authentication key |
| `TRUSTED_PROXY_CIDRS` | Railway private network CIDRs (e.g., `10.0.0.0/8`) |
| `MAX_UPLOAD_SIZE_BYTES` | Max upload size (default: 50MB) |
| `GEOIP_DATABASE_PATH` | Path to MaxMind GeoLite2-Country.mmdb (optional) |
| `REDIS_URL` | Redis URL for distributed rate limiting (optional) |

### Security Checklist for Production Deployment

Before deploying, verify:

- [ ] `BACKEND_API_KEY` is set to a strong random value (or Basic Auth configured)
- [ ] `AUTH_DISABLED` is NOT set (or explicitly `false`)
- [ ] `TRUSTED_PROXY_CIDRS` is configured for your platform (Railway, Cloudflare, etc.)
- [ ] `CORS_ORIGINS` only includes your frontend domain
- [ ] Database uses SSL/TLS connection
- [ ] R2 bucket has proper CORS policy (frontend origin only)
- [ ] `REDIS_URL` configured for multi-instance rate limiting (or single-instance deployment)
- [ ] `GEOIP_DATABASE_PATH` configured for server-side IP geolocation (recommended)
- [ ] pgvector extension enabled in PostgreSQL (run `alembic upgrade head`)

## Dependencies to Watch

| Dependency | Note |
|------------|------|
| PaddleOCR | v2.8+ uses new `predict()` API. Supports 50+ languages via `lang` parameter. |
| InsightFace | Uses ONNX Runtime. GPU requires `onnxruntime-gpu`. Age estimation requires buffalo_l model. |
| pgvector | Requires PostgreSQL extension enabled in production. Used for face similarity search. |
| Next.js | Currently on v16.x. App Router is stable. |
| Modal | API changes frequently. Use `modal.Function.from_name()` for deployed functions. |
| OpenCV | Used for PAD heuristics, ELA analysis, and lip sync detection. |

## Keyboard Shortcuts

| Shortcut | Context | Action |
|----------|---------|--------|
| `⌘K` / `Ctrl+K` | Global | Open command palette |
| `+` / `-` | Lightbox | Zoom in/out |
| `R` | Lightbox | Rotate image |
| `0` | Lightbox | Reset zoom/rotation |
| `Esc` | Lightbox/Dialogs | Close |

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

### GeoIP lookup returns None
1. Download GeoLite2-Country.mmdb from MaxMind (free registration required)
2. Set `GEOIP_DATABASE_PATH=/path/to/GeoLite2-Country.mmdb`
3. Or place database in common location: `data/GeoLite2-Country.mmdb` or `/usr/share/GeoIP/`

### Redis rate limiting not working
1. Verify `REDIS_URL` is set correctly (e.g., `redis://localhost:6379/0`)
2. Check Redis server is running and accessible
3. Check logs for connection errors (system falls back to in-memory if Redis unavailable)

### pgvector extension not found
1. Railway: Enable in Dashboard → Postgres → Data → Extensions → vector
2. Supabase: Database → Extensions → Search "vector" → Enable
3. Self-hosted: `apt install postgresql-16-pgvector` then `CREATE EXTENSION vector;`
4. Run `alembic upgrade head` to verify extension and create indexes
