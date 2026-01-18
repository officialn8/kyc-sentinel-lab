"""WebSocket endpoint for real-time session processing progress.

Provides live updates to clients during session processing, including
step-by-step progress and completion/failure events.

Key features:
- State replay: Late-connecting clients receive the current progress state
- Auto-cleanup: Session state is cleaned up after terminal events or TTL expiry
"""

import asyncio
import logging
import time
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# TTL for session state (5 minutes) - cleanup old states
SESSION_STATE_TTL_SECONDS = 300


class ConnectionManager:
    """Manages WebSocket connections per session with state replay.

    Single-instance implementation. For multi-instance deployments,
    use Redis pub/sub (see Phase 2 in implementation plan).

    Features:
    - Stores last progress state per session
    - Replays current state to late-connecting clients
    - Auto-cleanup of stale session states
    """

    def __init__(self) -> None:
        # session_id -> list of active WebSocket connections
        self._connections: dict[str, list[WebSocket]] = {}
        # session_id -> (last_state, timestamp) for state replay
        self._session_states: dict[str, tuple[dict, float]] = {}
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

    def _start_cleanup_task(self) -> None:
        """Start background cleanup task if not already running."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_stale_states())

    async def _cleanup_stale_states(self) -> None:
        """Periodically clean up stale session states."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                now = time.time()
                stale_sessions = [
                    sid for sid, (_, ts) in self._session_states.items()
                    if now - ts > SESSION_STATE_TTL_SECONDS
                ]
                for sid in stale_sessions:
                    del self._session_states[sid]
                    logger.debug(f"Cleaned up stale state for session {sid}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    async def connect(self, session_id: str, websocket: WebSocket) -> Optional[dict]:
        """Accept and register a new WebSocket connection.

        Returns the current session state if available (for replay).
        """
        await websocket.accept()
        if session_id not in self._connections:
            self._connections[session_id] = []
        self._connections[session_id].append(websocket)
        logger.info(f"WebSocket connected for session {session_id}")

        # Start cleanup task if needed
        self._start_cleanup_task()

        # Return current state for replay (if exists and not stale)
        if session_id in self._session_states:
            state, timestamp = self._session_states[session_id]
            if time.time() - timestamp < SESSION_STATE_TTL_SECONDS:
                return state
        return None
    
    async def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        """Remove a WebSocket connection from the registry."""
        if session_id in self._connections:
            try:
                self._connections[session_id].remove(websocket)
                if not self._connections[session_id]:
                    del self._connections[session_id]
            except ValueError:
                pass  # Already removed
        logger.info(f"WebSocket disconnected for session {session_id}")

    def store_state(self, session_id: str, message: dict) -> None:
        """Store the current progress state for a session.

        This allows late-connecting clients to receive the current state.
        """
        self._session_states[session_id] = (message, time.time())

        # Clean up state after terminal events (with small delay for final delivery)
        if message.get("event") in ("completed", "failed"):
            # Keep state for a short time so late clients can still see it
            pass  # State will be cleaned up by TTL

    def clear_state(self, session_id: str) -> None:
        """Clear stored state for a session."""
        self._session_states.pop(session_id, None)

    async def broadcast(self, session_id: str, message: dict) -> None:
        """Send a message to all connections for a session.

        Also stores the message as the current state for late-connecting clients.
        """
        # Always store state (even if no clients connected yet)
        self.store_state(session_id, message)

        if session_id not in self._connections:
            logger.debug(f"No WebSocket clients for session {session_id}, state stored for replay")
            return

        dead_connections = []
        for websocket in self._connections[session_id]:
            try:
                await websocket.send_json(message)
            except Exception:
                dead_connections.append(websocket)

        # Clean up dead connections
        for ws in dead_connections:
            try:
                self._connections[session_id].remove(ws)
            except ValueError:
                pass

    def has_connections(self, session_id: str) -> bool:
        """Check if a session has active connections."""
        return session_id in self._connections and len(self._connections[session_id]) > 0

    def get_current_state(self, session_id: str) -> Optional[dict]:
        """Get the current stored state for a session."""
        if session_id in self._session_states:
            state, timestamp = self._session_states[session_id]
            if time.time() - timestamp < SESSION_STATE_TTL_SECONDS:
                return state
        return None


# Singleton connection manager
manager = ConnectionManager()


async def emit_progress(
    session_id: str,
    event: str,
    step: Optional[str] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Emit a progress event to all connected clients for a session.
    
    Args:
        session_id: The session to broadcast to
        event: Event type (started, progress, completed, failed)
        step: Processing step name (face_analysis, document_analysis, etc.)
        progress: Progress percentage (0.0 - 1.0)
        message: Human-readable status message
        error: Error message (for failed events)
    """
    payload = {"event": event}
    
    if step is not None:
        payload["step"] = step
    if progress is not None:
        payload["progress"] = progress
    if message is not None:
        payload["message"] = message
    if error is not None:
        payload["error"] = error
    
    await manager.broadcast(session_id, payload)


@router.websocket("/ws/sessions/{session_id}")
async def session_progress_websocket(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time session processing updates.

    Connect to receive progress events during session processing:

    Events:
    - connected: Initial connection established
    - started: Processing has begun
    - progress: Step progress update
    - completed: Processing finished successfully
    - failed: Processing encountered an error

    Message format:
    {
        "event": "progress",
        "step": "face_analysis",
        "progress": 0.5,
        "message": "Analyzing faces..."
    }

    State Replay:
    If the client connects after processing has started, the current
    progress state will be sent immediately after the connection message.
    """
    # Validate session_id format
    try:
        UUID(session_id)
    except ValueError:
        await websocket.close(code=4000, reason="Invalid session ID format")
        return

    # Connect and get current state for replay
    current_state = await manager.connect(session_id, websocket)

    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "event": "connected",
            "session_id": session_id,
            "message": "Connected to session progress stream"
        })

        # Replay current state if available (for late-connecting clients)
        if current_state:
            logger.info(f"Replaying state for session {session_id}: {current_state.get('event')}")
            await websocket.send_json(current_state)

            # If the session is already completed/failed, close after sending state
            if current_state.get("event") in ("completed", "failed"):
                logger.info(f"Session {session_id} already {current_state.get('event')}, closing connection")
                return

        # Keep connection alive and wait for disconnect
        while True:
            # Wait for ping or client messages
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(session_id, websocket)
