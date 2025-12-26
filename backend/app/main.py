"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import sessions, upload, simulate, metrics
from app.api.security import require_auth


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    # Note: Database tables are managed by Alembic migrations
    yield
    # Shutdown
    # Add cleanup logic here if needed


app = FastAPI(
    title=settings.app_name,
    description="KYC Deepfake Red-Team Simulator with Explainable Detection",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check
@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


# Include routers
api_auth = [Depends(require_auth)]
app.include_router(
    sessions.router, prefix="/api/sessions", tags=["sessions"], dependencies=api_auth
)
app.include_router(
    upload.router, prefix="/api/upload", tags=["upload"], dependencies=api_auth
)
app.include_router(
    simulate.router, prefix="/api/simulate", tags=["simulate"], dependencies=api_auth
)
app.include_router(
    metrics.router, prefix="/api/metrics", tags=["metrics"], dependencies=api_auth
)






