# KYC Sentinel Lab Architecture

## Overview

KYC Sentinel Lab is a red-team simulator for KYC (Know Your Customer) identity verification flows. It allows fraud teams to:

1. **Upload** real KYC sessions for analysis
2. **Simulate** synthetic attack patterns
3. **Evaluate** detection capabilities
4. **Review** explainable reason codes

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                 │
│                    (Next.js 14 App Router)                      │
├─────────────────────────────────────────────────────────────────┤
│  Dashboard  │  Sessions  │  Upload  │  Simulate  │  Metrics    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP/REST
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

## Component Details

### Frontend (Next.js 14)

- **App Router**: File-based routing with React Server Components
- **shadcn/ui**: Accessible, customizable component library
- **TanStack Query**: Server state management with caching
- **Recharts**: Data visualization for metrics

### Backend (FastAPI)

- **API Layer**: RESTful endpoints with automatic OpenAPI docs
- **Services**: Business logic (storage, scoring, processing)
- **Detection**: ML/CV modules (face, document, PAD analysis)
- **Processing Backend**: Abstraction for local vs. serverless processing

### Database (PostgreSQL + pgvector)

- **Session data**: KYC verification sessions
- **Results**: Scoring and decision outcomes
- **Reasons**: Explainable reason codes with evidence
- **Embeddings**: Face embeddings for similarity search

### Storage (R2/S3)

- **Media assets**: Selfie images, ID documents
- **Processed outputs**: Frame extracts, cropped faces

## Processing Pipeline

```
┌─────────────┐
│ Create      │
│ Session     │
└─────┬───────┘
      │
┌─────▼───────┐
│ Upload      │
│ Assets      │───► R2 Storage
└─────┬───────┘
      │
┌─────▼───────┐
│ Finalize    │
│ Session     │
└─────┬───────┘
      │ (Background)
┌─────▼───────────────────────────────────────────┐
│ Processing Pipeline                              │
├──────────────────────────────────────────────────┤
│ 1. Frame Extraction (video → frames)            │
│ 2. Face Analysis (detection, embedding, match)  │
│ 3. PAD Analysis (liveness, artifact detection)  │
│ 4. Document Analysis (OCR, tampering)           │
│ 5. Scoring (risk score, decision)               │
│ 6. Reason Generation (explainable codes)        │
└──────────────────────────────────────────────────┘
```

## Reason Code System

Each detection generates structured reason codes:

```json
{
  "code": "PAD_SUSPECT_REPLAY",
  "severity": "high",
  "message": "Detected screen capture artifacts suggesting replay attack",
  "evidence": {
    "frame_indices": [12, 13, 14],
    "moire_score": 0.85,
    "affected_area": "face_region"
  }
}
```

Severity levels:
- `info`: Informational, no action needed
- `warn`: Requires review
- `high`: Strong indicator of fraud

## Scalability

The processing backend is designed for migration:

1. **MVP**: FastAPI BackgroundTasks (in-process)
2. **Scale**: Modal serverless (distributed GPU)

The `ProcessingBackend` protocol ensures consistent interfaces.

## Security Considerations

- Presigned URLs for media upload/download (no direct storage access)
- Session-scoped asset keys
- No PII stored in logs
- Synthetic data mode for safe testing



