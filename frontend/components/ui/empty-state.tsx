"use client";

import * as React from "react";
import Link from "next/link";
import { LucideIcon, FileSearch, Database, Zap, Upload, BarChart3 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description: string;
  action?: {
    label: string;
    href?: string;
    onClick?: () => void;
  };
  secondaryAction?: {
    label: string;
    href?: string;
    onClick?: () => void;
  };
  className?: string;
  variant?: "default" | "compact";
}

export function EmptyState({
  icon: Icon = FileSearch,
  title,
  description,
  action,
  secondaryAction,
  className,
  variant = "default",
}: EmptyStateProps) {
  const isCompact = variant === "compact";

  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center text-center",
        isCompact ? "py-8 px-4" : "py-16 px-6",
        className
      )}
    >
      {/* Illustrated icon with glow effect */}
      <div
        className={cn(
          "relative mb-4",
          isCompact ? "mb-3" : "mb-6"
        )}
      >
        <div className="absolute inset-0 bg-primary/20 blur-2xl rounded-full scale-150" />
        <div
          className={cn(
            "relative flex items-center justify-center rounded-2xl bg-gradient-to-br from-primary/20 to-primary/5 border border-primary/20",
            isCompact ? "h-14 w-14" : "h-20 w-20"
          )}
        >
          <Icon
            className={cn(
              "text-primary",
              isCompact ? "h-7 w-7" : "h-10 w-10"
            )}
          />
        </div>
      </div>

      {/* Title */}
      <h3
        className={cn(
          "font-semibold text-foreground",
          isCompact ? "text-base mb-1" : "text-xl mb-2"
        )}
      >
        {title}
      </h3>

      {/* Description */}
      <p
        className={cn(
          "text-muted-foreground max-w-sm",
          isCompact ? "text-sm mb-4" : "text-base mb-6"
        )}
      >
        {description}
      </p>

      {/* Actions */}
      {(action || secondaryAction) && (
        <div className="flex flex-col sm:flex-row gap-3">
          {action && (
            action.href ? (
              <Button asChild size={isCompact ? "sm" : "default"}>
                <Link href={action.href}>{action.label}</Link>
              </Button>
            ) : (
              <Button size={isCompact ? "sm" : "default"} onClick={action.onClick}>
                {action.label}
              </Button>
            )
          )}
          {secondaryAction && (
            secondaryAction.href ? (
              <Button variant="outline" size={isCompact ? "sm" : "default"} asChild>
                <Link href={secondaryAction.href}>{secondaryAction.label}</Link>
              </Button>
            ) : (
              <Button variant="outline" size={isCompact ? "sm" : "default"} onClick={secondaryAction.onClick}>
                {secondaryAction.label}
              </Button>
            )
          )}
        </div>
      )}
    </div>
  );
}

// Pre-configured empty states for common use cases
export function EmptySessionsState() {
  return (
    <EmptyState
      icon={Database}
      title="No sessions found"
      description="Get started by uploading a KYC session for analysis or generate synthetic attack simulations to test your detection capabilities."
      action={{
        label: "Upload Session",
        href: "/upload",
      }}
      secondaryAction={{
        label: "Generate Attack",
        href: "/simulate",
      }}
    />
  );
}

export function EmptyMetricsState() {
  return (
    <EmptyState
      icon={BarChart3}
      title="No metrics available"
      description="Process some sessions to see detection metrics, confusion matrices, and score distributions."
      action={{
        label: "View Sessions",
        href: "/sessions",
      }}
    />
  );
}

export function EmptySearchState({ query }: { query?: string }) {
  return (
    <EmptyState
      icon={FileSearch}
      title="No results found"
      description={
        query
          ? `No sessions matching "${query}". Try adjusting your search or filters.`
          : "Try adjusting your search criteria or filters."
      }
      variant="compact"
    />
  );
}
