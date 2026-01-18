"""Background task utilities with improved error handling.

Wraps FastAPI BackgroundTasks to ensure exceptions are logged instead of
silently swallowed, preventing sessions from being stuck in "processing" state.
"""

import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)


def logged_background_task(func: Callable) -> Callable:
    """Decorator that wraps async functions to log exceptions.
    
    FastAPI's BackgroundTasks silently swallows exceptions, which can leave
    sessions stuck in "processing" state. This wrapper ensures all exceptions
    are logged for debugging and alerting.
    
    Usage:
        @logged_background_task
        async def process_session(session_id: str) -> None:
            ...
        
        background_tasks.add_task(process_session, session_id)
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Log full traceback for debugging
            logger.exception(
                f"Background task {func.__name__} failed: {e!r}",
                extra={
                    "task_name": func.__name__,
                    "args": str(args)[:200],  # Truncate to prevent log bloat
                    "kwargs": str(kwargs)[:200],
                }
            )
            # Re-raise so caller can optionally handle
            raise
    return wrapper


async def run_with_error_handling(
    task_name: str,
    func: Callable,
    *args: Any,
    **kwargs: Any,
) -> tuple[bool, Any | Exception]:
    """Run an async function with structured error handling.
    
    Returns:
        Tuple of (success: bool, result_or_exception)
    
    Usage:
        success, result = await run_with_error_handling(
            "process_session",
            backend.process_session,
            session_id
        )
        if not success:
            logger.error(f"Task failed: {result}")
    """
    try:
        result = await func(*args, **kwargs)
        return True, result
    except Exception as e:
        logger.exception(
            f"Task {task_name} failed: {e!r}",
            extra={"task_name": task_name}
        )
        return False, e
