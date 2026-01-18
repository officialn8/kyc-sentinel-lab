import { useState, useEffect, useCallback, useRef } from "react";
import type {
  ConnectionState,
  ProcessingStep,
  WebSocketMessage,
  WebSocketState,
} from "@/lib/types/websocket";

/**
 * Debounce a value with the specified delay
 */
export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(timer);
    };
  }, [value, delay]);

  return debouncedValue;
}

/**
 * Debounce a callback function
 */
export function useDebouncedCallback<T extends (...args: any[]) => any>(
  callback: T,
  delay: number
): (...args: Parameters<T>) => void {
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const debouncedCallback = useCallback(
    (...args: Parameters<T>) => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      timeoutRef.current = setTimeout(() => {
        callback(...args);
      }, delay);
    },
    [callback, delay]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return debouncedCallback;
}

/**
 * Track if a component is mounted
 */
export function useIsMounted(): boolean {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
    return () => setIsMounted(false);
  }, []);

  return isMounted;
}

/**
 * Get previous value of a state
 */
export function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T>();

  useEffect(() => {
    ref.current = value;
  }, [value]);

  return ref.current;
}

/**
 * WebSocket hook for real-time session processing updates
 *
 * Features:
 * - Auto-reconnect with exponential backoff
 * - Fallback to polling mode after max retries
 * - Clean disconnect on unmount
 */
interface UseSessionWebSocketOptions {
  enabled?: boolean;
  onComplete?: () => void;
  onFailed?: (error: string) => void;
}

const MAX_RECONNECT_ATTEMPTS = 5;
const INITIAL_RECONNECT_DELAY = 1000;

function getWebSocketUrl(sessionId: string): string {
  // Use environment variable or derive from window location
  const wsBase =
    typeof window !== "undefined"
      ? process.env.NEXT_PUBLIC_BACKEND_WS_URL ||
        `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}`
      : "ws://localhost:8000";

  return `${wsBase}/ws/sessions/${sessionId}`;
}

export function useSessionWebSocket(
  sessionId: string,
  options: UseSessionWebSocketOptions = {}
): WebSocketState {
  const { enabled = true, onComplete, onFailed } = options;

  const [state, setState] = useState<WebSocketState>({
    connectionState: "idle",
    currentStep: null,
    progress: 0,
    message: "",
    isComplete: false,
    isFailed: false,
    error: null,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  // Use refs to avoid stale closures in WebSocket handlers
  const stateRef = useRef(state);
  stateRef.current = state;

  // Callbacks refs to avoid stale closures
  const onCompleteRef = useRef(onComplete);
  const onFailedRef = useRef(onFailed);
  onCompleteRef.current = onComplete;
  onFailedRef.current = onFailed;

  const connect = useCallback(() => {
    if (!enabled || !sessionId || !mountedRef.current) return;

    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setState((prev) => ({ ...prev, connectionState: "connecting" }));

    try {
      const url = getWebSocketUrl(sessionId);
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        reconnectAttemptRef.current = 0;
        setState((prev) => ({
          ...prev,
          connectionState: "connected",
          error: null,
        }));
      };

      ws.onmessage = (event) => {
        if (!mountedRef.current) return;

        try {
          const message: WebSocketMessage = JSON.parse(event.data);

          switch (message.event) {
            case "connected":
              // Initial connection confirmation
              break;

            case "started":
              setState((prev) => ({
                ...prev,
                progress: 0,
                message: message.message || "Processing started",
                currentStep: null,
              }));
              break;

            case "progress":
              setState((prev) => ({
                ...prev,
                progress: (message.progress || 0) * 100,
                message: message.message || "",
                currentStep: message.step || null,
              }));
              break;

            case "completed":
              setState((prev) => ({
                ...prev,
                progress: 100,
                message: message.message || "Processing complete",
                isComplete: true,
                currentStep: null,
              }));
              onCompleteRef.current?.();
              // Close connection after completion
              ws.close();
              break;

            case "failed":
              setState((prev) => ({
                ...prev,
                isFailed: true,
                error: message.error || "Processing failed",
                currentStep: null,
              }));
              onFailedRef.current?.(message.error || "Processing failed");
              ws.close();
              break;
          }
        } catch (e) {
          console.error("Failed to parse WebSocket message:", e);
        }
      };

      ws.onerror = () => {
        if (!mountedRef.current) return;
        console.warn("WebSocket error occurred");
      };

      ws.onclose = (event) => {
        if (!mountedRef.current) return;

        wsRef.current = null;

        // Don't reconnect if closed cleanly or completed/failed
        if (event.wasClean) return;

        // Use ref to get current state (avoids stale closure)
        const currentState = stateRef.current;
        if (currentState.isComplete || currentState.isFailed) return;

        // Attempt reconnection
        if (reconnectAttemptRef.current < MAX_RECONNECT_ATTEMPTS) {
          const delay =
            INITIAL_RECONNECT_DELAY * Math.pow(2, reconnectAttemptRef.current);
          reconnectAttemptRef.current++;

          setState((prev) => ({ ...prev, connectionState: "reconnecting" }));

          reconnectTimeoutRef.current = setTimeout(() => {
            if (mountedRef.current) {
              connect();
            }
          }, delay);
        } else {
          // Fallback to polling
          setState((prev) => ({
            ...prev,
            connectionState: "fallback",
            error: "WebSocket connection failed, using polling",
          }));
        }
      };
    } catch (e) {
      console.error("Failed to create WebSocket:", e);
      setState((prev) => ({
        ...prev,
        connectionState: "fallback",
        error: "WebSocket not supported",
      }));
    }
  }, [enabled, sessionId]); // Removed state dependencies - using stateRef instead

  // Connect when enabled
  useEffect(() => {
    mountedRef.current = true;

    if (enabled && sessionId) {
      connect();
    }

    return () => {
      mountedRef.current = false;

      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }

      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [enabled, sessionId, connect]);

  // Reset state when session changes
  useEffect(() => {
    setState({
      connectionState: "idle",
      currentStep: null,
      progress: 0,
      message: "",
      isComplete: false,
      isFailed: false,
      error: null,
    });
    reconnectAttemptRef.current = 0;
  }, [sessionId]);

  return state;
}
