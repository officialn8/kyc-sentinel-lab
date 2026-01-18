/**
 * WebSocket types for real-time session processing updates
 */

export type ConnectionState =
  | "idle"        // Not connected
  | "connecting"  // WebSocket handshake in progress
  | "connected"   // Active connection receiving events
  | "reconnecting" // Lost connection, attempting to reconnect
  | "fallback";   // WebSocket failed, using polling

export type ProcessingStep =
  | "face_analysis"
  | "document_analysis"
  | "pad_analysis"
  | "scoring";

export interface ConnectedMessage {
  event: "connected";
  session_id: string;
  message: string;
}

export interface StartedMessage {
  event: "started";
  message: string;
}

export interface ProgressMessage {
  event: "progress";
  step: ProcessingStep;
  progress: number;
  message: string;
}

export interface CompletedMessage {
  event: "completed";
  progress: number;
  message: string;
}

export interface FailedMessage {
  event: "failed";
  error: string;
}

export type WebSocketMessage =
  | ConnectedMessage
  | StartedMessage
  | ProgressMessage
  | CompletedMessage
  | FailedMessage;

export interface WebSocketState {
  connectionState: ConnectionState;
  currentStep: ProcessingStep | null;
  progress: number;
  message: string;
  isComplete: boolean;
  isFailed: boolean;
  error: string | null;
}

// Step display metadata
export const PROCESSING_STEPS: {
  key: ProcessingStep;
  label: string;
  progress: number;
}[] = [
  { key: "face_analysis", label: "Face Analysis", progress: 20 },
  { key: "document_analysis", label: "Document", progress: 40 },
  { key: "pad_analysis", label: "PAD", progress: 60 },
  { key: "scoring", label: "Scoring", progress: 80 },
];
