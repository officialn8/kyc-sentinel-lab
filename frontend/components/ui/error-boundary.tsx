"use client";

import * as React from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("Error boundary caught an error:", error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <ErrorFallback
          error={this.state.error}
          onRetry={this.handleRetry}
        />
      );
    }

    return this.props.children;
  }
}

interface ErrorFallbackProps {
  error: Error | null;
  onRetry?: () => void;
  title?: string;
  description?: string;
}

export function ErrorFallback({
  error,
  onRetry,
  title = "Something went wrong",
  description = "An unexpected error occurred. Please try again.",
}: ErrorFallbackProps) {
  return (
    <div className="flex min-h-[400px] items-center justify-center p-6">
      <Card className="glass max-w-md w-full">
        <CardHeader className="text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-danger/10">
            <AlertTriangle className="h-8 w-8 text-danger" />
          </div>
          <CardTitle className="text-xl">{title}</CardTitle>
        </CardHeader>
        <CardContent className="text-center space-y-4">
          <p className="text-muted-foreground">{description}</p>
          
          {error && process.env.NODE_ENV === "development" && (
            <details className="text-left">
              <summary className="text-sm text-muted-foreground cursor-pointer hover:text-foreground">
                Technical details
              </summary>
              <pre className="mt-2 text-xs bg-muted p-3 rounded overflow-auto max-h-32">
                {error.message}
                {error.stack && `\n\n${error.stack}`}
              </pre>
            </details>
          )}

          <div className="flex gap-3 justify-center pt-2">
            {onRetry && (
              <Button onClick={onRetry} className="gap-2">
                <RefreshCw className="h-4 w-4" />
                Try Again
              </Button>
            )}
            <Button
              variant="outline"
              onClick={() => window.location.reload()}
            >
              Refresh Page
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Inline error state for cards/sections
interface InlineErrorProps {
  error: Error | unknown;
  onRetry?: () => void;
  className?: string;
  compact?: boolean;
}

export function InlineError({
  error,
  onRetry,
  className,
  compact = false,
}: InlineErrorProps) {
  const message =
    error instanceof Error
      ? error.message
      : typeof error === "string"
      ? error
      : "An error occurred";

  if (compact) {
    return (
      <div className={`flex items-center gap-2 text-danger text-sm ${className}`}>
        <AlertTriangle className="h-4 w-4 flex-shrink-0" />
        <span className="flex-1 truncate">{message}</span>
        {onRetry && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onRetry}
            className="h-7 px-2"
          >
            <RefreshCw className="h-3 w-3" />
          </Button>
        )}
      </div>
    );
  }

  return (
    <div
      className={`flex flex-col items-center justify-center py-8 px-4 text-center ${className}`}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-danger/10 mb-3">
        <AlertTriangle className="h-6 w-6 text-danger" />
      </div>
      <p className="text-sm font-medium text-danger mb-1">Error</p>
      <p className="text-sm text-muted-foreground mb-4 max-w-xs">{message}</p>
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry} className="gap-2">
          <RefreshCw className="h-4 w-4" />
          Retry
        </Button>
      )}
    </div>
  );
}
