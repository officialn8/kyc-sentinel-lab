"""API dependencies."""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.storage import StorageService, get_storage_service
from app.services.processing import ProcessingBackend, get_processing_backend


# Database session dependency
DbSession = Annotated[AsyncSession, Depends(get_db)]

# Storage service dependency
Storage = Annotated[StorageService, Depends(get_storage_service)]

# Processing backend dependency
Processing = Annotated[ProcessingBackend, Depends(get_processing_backend)]



