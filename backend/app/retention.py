"""Run data retention purge once.

Usage:
  python -m app.retention
"""

import asyncio

from app.database import async_session_maker
from app.services.data_lifecycle import (
    cleanup_stale_pending_uploads,
    purge_expired_media,
    purge_expired_metadata,
)
from app.services.storage import get_storage_service


async def run_once() -> None:
    storage = get_storage_service()
    async with async_session_maker() as db:
        await cleanup_stale_pending_uploads(db, storage)
        await purge_expired_media(db, storage)
        await purge_expired_metadata(db, storage)


def main() -> None:
    asyncio.run(run_once())


if __name__ == "__main__":
    main()
