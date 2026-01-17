"use client";

import * as React from "react";
import { Wifi, WifiOff } from "lucide-react";
import { cn } from "@/lib/utils";

export function ConnectionStatus() {
  const [isOnline, setIsOnline] = React.useState(true);
  const [showBanner, setShowBanner] = React.useState(false);

  React.useEffect(() => {
    // Set initial state
    setIsOnline(navigator.onLine);

    const handleOnline = () => {
      setIsOnline(true);
      // Show "back online" briefly
      setShowBanner(true);
      setTimeout(() => setShowBanner(false), 3000);
    };

    const handleOffline = () => {
      setIsOnline(false);
      setShowBanner(true);
    };

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  // Only show when offline or briefly when coming back online
  if (isOnline && !showBanner) {
    return null;
  }

  return (
    <div
      className={cn(
        "fixed top-0 left-0 right-0 z-50 flex items-center justify-center gap-2 py-2 px-4 text-sm font-medium transition-all",
        isOnline
          ? "bg-success text-success-foreground"
          : "bg-danger text-danger-foreground"
      )}
    >
      {isOnline ? (
        <>
          <Wifi className="h-4 w-4" />
          <span>Back online</span>
        </>
      ) : (
        <>
          <WifiOff className="h-4 w-4" />
          <span>You're offline. Some features may be unavailable.</span>
        </>
      )}
    </div>
  );
}

// Small indicator for navbar
export function ConnectionIndicator() {
  const [isOnline, setIsOnline] = React.useState(true);

  React.useEffect(() => {
    setIsOnline(navigator.onLine);

    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  if (isOnline) {
    return null;
  }

  return (
    <div className="flex items-center gap-1.5 text-danger text-sm">
      <WifiOff className="h-4 w-4" />
      <span className="hidden sm:inline">Offline</span>
    </div>
  );
}
