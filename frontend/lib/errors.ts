/**
 * Error handling utilities for user-friendly error messages
 */

// Map common error patterns to user-friendly messages
const ERROR_MESSAGES: Record<string, string> = {
  // Network errors
  "Failed to fetch": "Unable to connect to the server. Please check your internet connection.",
  "Network request failed": "Network error. Please check your connection and try again.",
  "NetworkError": "Network error. Please check your connection and try again.",
  "TypeError: Failed to fetch": "Unable to connect to the server. Please check your internet connection.",
  
  // HTTP status errors
  "HTTP 400": "Invalid request. Please check your input and try again.",
  "HTTP 401": "You need to sign in to access this resource.",
  "HTTP 403": "You don't have permission to access this resource.",
  "HTTP 404": "The requested resource was not found.",
  "HTTP 409": "This action conflicts with the current state. Please refresh and try again.",
  "HTTP 413": "The file is too large. Please try a smaller file.",
  "HTTP 422": "Invalid data provided. Please check your input.",
  "HTTP 429": "Too many requests. Please wait a moment and try again.",
  "HTTP 500": "Something went wrong on our end. Please try again later.",
  "HTTP 502": "Service temporarily unavailable. Please try again later.",
  "HTTP 503": "Service temporarily unavailable. Please try again later.",
  "HTTP 504": "The server took too long to respond. Please try again.",
  
  // Domain-specific errors
  "Session not found": "This session doesn't exist or has been deleted.",
  "Processing failed": "Session processing failed. Please try uploading again.",
  "Upload failed": "Failed to upload the file. Please try again.",
  "Invalid file type": "This file type is not supported. Please use JPEG, PNG, or WebP.",
  "File too large": "The file is too large. Maximum size is 10MB.",
  
  // Default
  "Unknown error": "An unexpected error occurred. Please try again.",
};

// Error codes for programmatic handling
export type ErrorCode = 
  | "NETWORK_ERROR"
  | "AUTH_ERROR"
  | "NOT_FOUND"
  | "VALIDATION_ERROR"
  | "RATE_LIMITED"
  | "SERVER_ERROR"
  | "TIMEOUT"
  | "UNKNOWN";

export interface ParsedError {
  message: string;
  code: ErrorCode;
  retryable: boolean;
  originalError?: Error;
}

/**
 * Parse an error and return a user-friendly message
 */
export function parseError(error: unknown): ParsedError {
  // Handle null/undefined
  if (!error) {
    return {
      message: ERROR_MESSAGES["Unknown error"],
      code: "UNKNOWN",
      retryable: true,
    };
  }

  // Extract message from various error types
  let message = "Unknown error";
  let originalError: Error | undefined;

  if (error instanceof Error) {
    message = error.message;
    originalError = error;
  } else if (typeof error === "string") {
    message = error;
  } else if (typeof error === "object" && "message" in error) {
    message = String((error as { message: unknown }).message);
  } else if (typeof error === "object" && "detail" in error) {
    message = String((error as { detail: unknown }).detail);
  }

  // Determine error code and retryability
  let code: ErrorCode = "UNKNOWN";
  let retryable = true;

  if (message.includes("Failed to fetch") || message.includes("Network")) {
    code = "NETWORK_ERROR";
    retryable = true;
  } else if (message.includes("401") || message.includes("403")) {
    code = "AUTH_ERROR";
    retryable = false;
  } else if (message.includes("404") || message.includes("not found")) {
    code = "NOT_FOUND";
    retryable = false;
  } else if (message.includes("400") || message.includes("422") || message.includes("Invalid")) {
    code = "VALIDATION_ERROR";
    retryable = false;
  } else if (message.includes("429")) {
    code = "RATE_LIMITED";
    retryable = true;
  } else if (message.includes("500") || message.includes("502") || message.includes("503")) {
    code = "SERVER_ERROR";
    retryable = true;
  } else if (message.includes("504") || message.includes("timeout")) {
    code = "TIMEOUT";
    retryable = true;
  }

  // Find matching user-friendly message
  let friendlyMessage = ERROR_MESSAGES["Unknown error"];
  for (const [pattern, msg] of Object.entries(ERROR_MESSAGES)) {
    if (message.toLowerCase().includes(pattern.toLowerCase())) {
      friendlyMessage = msg;
      break;
    }
  }

  // Check for HTTP status code pattern
  const httpMatch = message.match(/HTTP\s*(\d{3})/i);
  if (httpMatch) {
    const statusKey = `HTTP ${httpMatch[1]}`;
    if (ERROR_MESSAGES[statusKey]) {
      friendlyMessage = ERROR_MESSAGES[statusKey];
    }
  }

  return {
    message: friendlyMessage,
    code,
    retryable,
    originalError,
  };
}

/**
 * Get a short error title based on error code
 */
export function getErrorTitle(code: ErrorCode): string {
  switch (code) {
    case "NETWORK_ERROR":
      return "Connection Error";
    case "AUTH_ERROR":
      return "Authentication Error";
    case "NOT_FOUND":
      return "Not Found";
    case "VALIDATION_ERROR":
      return "Invalid Input";
    case "RATE_LIMITED":
      return "Too Many Requests";
    case "SERVER_ERROR":
      return "Server Error";
    case "TIMEOUT":
      return "Request Timeout";
    default:
      return "Error";
  }
}

/**
 * Check if the browser is online
 */
export function isOnline(): boolean {
  return typeof navigator !== "undefined" ? navigator.onLine : true;
}

/**
 * Create a hook to monitor online status
 */
export function useOnlineStatus() {
  if (typeof window === "undefined") {
    return true;
  }
  return navigator.onLine;
}
