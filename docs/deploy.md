# Deployment Guide (Vercel + Railway + Railway Postgres + Cloudflare R2 + Modal)

This repo is designed to deploy as:

- **Frontend**: Vercel (Next.js)
- **Backend API**: Railway (FastAPI)
- **Database**: Railway Postgres (with pgvector)
- **Object storage**: Cloudflare R2 (S3-compatible)
- **GPU processing**: Modal

## 1) Configure environment variables

Use [`env.example`](../env.example) as the source of truth.

### Vercel (Frontend)

- `NEXT_PUBLIC_API_URL`: your Railway backend base URL (example: `https://kyc-api-production.up.railway.app`)
- `BACKEND_API_KEY` (optional but recommended): shared secret used by the Next.js server-side proxy route to call the backend without exposing the secret to the browser.
- `NEXT_PUBLIC_R2_PUBLIC_HOSTNAME` (optional): if you use a custom domain for R2 assets, set the hostname (example: `media.example.com`).

### Railway (Backend API)

- `DATABASE_URL`: Railway Postgres URL
- `CORS_ORIGINS`: Vercel origin(s). Supported formats:
  - JSON array: `["https://your-app.vercel.app","https://yourdomain.com"]`
  - Comma-separated: `https://your-app.vercel.app,https://yourdomain.com`
- `R2_ENDPOINT`, `R2_ACCESS_KEY`, `R2_SECRET_KEY`, `R2_BUCKET`
- `USE_MODAL=true`
- `BACKEND_API_KEY` (optional): if set, backend endpoints require `X-API-Key: <value>`

## 2) Railway Postgres: enable pgvector and run migrations

The initial Alembic migration executes:

- `CREATE EXTENSION IF NOT EXISTS vector`

Make sure the Railway database role used by `DATABASE_URL` is allowed to create extensions. If not, enable `vector` manually once, then rerun migrations.

You should run migrations during deploy/startup:

```bash
alembic upgrade head
```

### Railway start command (recommended)

This repo includes a Railway-friendly startup script that runs migrations then starts the API:

- If your Railway service root is the repo root:

```bash
bash backend/start.sh
```

- If your Railway service root is `backend/`, set the start command to:

```bash
bash start.sh
```

## 3) Cloudflare R2: bucket + CORS

Create the bucket named by `R2_BUCKET`.

You must configure **bucket CORS** to allow browser uploads via presigned PUTs from your Vercel origin.

At minimum, allow:

- **Origins**: `https://your-app.vercel.app` (and custom domain if any)
- **Methods**: `PUT`, `GET`, `HEAD`
- **Headers**: `Content-Type`
- **Expose headers**: `ETag`

## 4) Modal: deploy GPU functions + secrets

The Modal app is defined in [`backend/modal_app.py`](../backend/modal_app.py).

Create a Modal secret named **`r2-credentials`** with:

- `R2_ENDPOINT`
- `R2_ACCESS_KEY`
- `R2_SECRET_KEY`
- `R2_BUCKET`

Then deploy the app via Modal CLI (example):

```bash
modal deploy backend/modal_app.py
```

## 5) Recommended: enable backend auth safely

If you set `BACKEND_API_KEY` on Railway, the browser should not call Railway directly. Instead, call the Next.js `/api/*` proxy route which injects the key server-side.


