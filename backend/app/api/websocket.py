"""WebSocket endpoint for real-time session processing progress.

Provides live updates to clients during session processing, including
step-by-step progress and completion/failure events.
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections per session.
    
    Single-instance implementation. For multi-instance deployments,
    use Redis pub/sub (see Phase 2 in implementation plan).
    """
    
    def __init__(self) -> None:
        # session_id -> list of active WebSocket connections
        self._connections: dict[str, list[WebSocket]] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        if session_id not in self._connections:
            self._connections[session_id] = []
        self._connections[session_id].append(websocket)
        logger.info(f"WebSocket connected for session {session_id}")
    
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
    
    async def broadcast(self, session_id: str, message: dict) -> None:
        """Send a message to all connections for a session."""
        if session_id not in self._connections:
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
    """
    # Validate session_id format
    try:
        UUID(session_id)
    except ValueError:
        await websocket.close(code=4000, reason="Invalid session ID format")
        return
    
    await manager.connect(session_id, websocket)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "event": "connected",
            "session_id": session_id,
            "message": "Connected to session progress stream"
        })
        
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
