# KYC Sentinel Lab

**Red-team your remote identity flow with safe, synthetic attacks and explainable detection.**

A KYC Deepfake Red-Team Simulator with an Explainable Detection Dashboard. This tool lets fraud teams simulate modern KYC attack patterns (replay, injection artifacts, face-swap artifacts, ID tampering) and evaluate a rules + ML scoring engine that returns clear, auditable reason codes with evidence.

## Features

- 🎭 **Attack Simulation**: Generate synthetic KYC attack patterns including replay attacks, injection artifacts, face-swap artifacts, and document tampering
- 🔍 **Explainable Detection**: Clear reason codes with evidence for every detection decision
- 📊 **Metrics Dashboard**: Confusion matrices, score distributions, and attack family breakdowns
- 🔄 **Processing Pipeline**: Modular backend designed for easy scaling (BackgroundTasks → Modal)
- 🗃️ **Face Similarity Search**: pgvector-powered embedding search to find similar sessions

## Tech Stack

### Frontend
- Next.js 14 (App Router)
- TypeScript (strict mode)
- Tailwind CSS
- shadcn/ui components
- TanStack Query (React Query)
- Recharts

### Backend
- FastAPI (Python 3.11+)
- SQLAlchemy 2.0 + Alembic
- Pydantic v2
- PostgreSQL 15+ with pgvector

### ML/CV Stack
- InsightFace (face detection, embeddings, PAD)
- PaddleOCR (document OCR + layout)
- OpenCV + NumPy
- ffmpeg-python

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- pnpm (recommended) or npm

### 1. Clone and Setup

```bash
cd kyc-sentinel-lab

# Copy environment file
cp env.example .env
# Edit .env with your configuration
```

### 2. Start Database

```bash
docker-compose up -d
```

### 3. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --port 8000
```

### 4. Frontend Setup

```bash
cd frontend

# Install dependencies
pnpm install

# Start development server
pnpm dev
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Project Structure

```
kyc-sentinel-lab/
├── frontend/           # Next.js 14 application
├── backend/            # FastAPI application
├── simulator/          # Synthetic artifact generator
├── docs/               # Documentation
└── docker-compose.yml  # Local development services
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sessions` | Create session, get presigned upload URLs |
| POST | `/api/sessions/{id}/finalize` | Mark uploads complete, start processing |
| GET | `/api/sessions` | List sessions with filters |
| GET | `/api/sessions/{id}` | Full session detail with results + reasons |
| DELETE | `/api/sessions/{id}` | Delete session |
| POST | `/api/simulate` | Generate synthetic session |
| GET | `/api/metrics/summary` | Aggregate metrics |

## Reason Codes

The detection engine returns explainable reason codes:

| Category | Code | Description |
|----------|------|-------------|
| Face | `FACE_MISMATCH` | Selfie does not match ID photo |
| PAD | `PAD_SUSPECT_REPLAY` | Screen capture artifacts detected |
| PAD | `PAD_LOW_MOTION_ENTROPY` | Insufficient natural movement |
| Document | `DOC_TEMPLATE_MISMATCH` | Layout doesn't match template |
| Document | `DOC_FONT_INCONSISTENT` | Font variations detected |

See [docs/reason-codes.md](docs/reason-codes.md) for the complete taxonomy.

## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
pnpm test
```

### Database Migrations

```bash
cd backend

# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Deployment

See [`docs/deployment.md`](docs/deployment.md) for the Vercel + Railway + R2 + Modal setup.

## License

MIT






