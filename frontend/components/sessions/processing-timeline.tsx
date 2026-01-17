"use client";

import * as React from "react";
import {
  Upload,
  Loader2,
  User,
  Shield,
  FileText,
  CheckCircle2,
  XCircle,
  Clock,
} from "lucide-react";
import { cn } from "@/lib/utils";

type SessionStatus = "pending" | "processing" | "completed" | "failed";

interface ProcessingTimelineProps {
  status: SessionStatus;
  hasResult?: boolean;
  hasFaceMatch?: boolean;
  hasPadAnalysis?: boolean;
  hasDocAnalysis?: boolean;
  className?: string;
}

interface TimelineStep {
  id: string;
  label: string;
  description: string;
  icon: React.ElementType;
  status: "completed" | "current" | "pending" | "failed";
}

export function ProcessingTimeline({
  status,
  hasResult,
  hasFaceMatch,
  hasPadAnalysis,
  hasDocAnalysis,
  className,
}: ProcessingTimelineProps) {
  // Determine step statuses based on session status and available data
  const getStepStatus = (stepId: string): TimelineStep["status"] => {
    if (status === "failed") {
      // Show which step failed (simplified logic)
      if (stepId === "upload") return "completed";
      if (stepId === "processing") return "failed";
      return "pending";
    }

    if (status === "pending") {
      if (stepId === "upload") return "completed";
      return "pending";
    }

    if (status === "processing") {
      if (stepId === "upload") return "completed";
      if (stepId === "processing") return "current";
      // Check individual analysis steps
      if (stepId === "face" && hasFaceMatch) return "completed";
      if (stepId === "pad" && hasPadAnalysis) return "completed";
      if (stepId === "doc" && hasDocAnalysis) return "completed";
      if (stepId === "face" && !hasFaceMatch) return "current";
      return "pending";
    }

    if (status === "completed") {
      return "completed";
    }

    return "pending";
  };

  const steps: TimelineStep[] = [
    {
      id: "upload",
      label: "Upload",
      description: "Media files uploaded",
      icon: Upload,
      status: getStepStatus("upload"),
    },
    {
      id: "processing",
      label: "Processing",
      description: "Extracting frames and features",
      icon: Loader2,
      status: getStepStatus("processing"),
    },
    {
      id: "face",
      label: "Face Match",
      description: "Comparing selfie to ID document",
      icon: User,
      status: getStepStatus("face"),
    },
    {
      id: "pad",
      label: "PAD Check",
      description: "Presentation attack detection",
      icon: Shield,
      status: getStepStatus("pad"),
    },
    {
      id: "doc",
      label: "Doc Check",
      description: "Document analysis & OCR",
      icon: FileText,
      status: getStepStatus("doc"),
    },
    {
      id: "complete",
      label: status === "failed" ? "Failed" : "Complete",
      description:
        status === "failed"
          ? "Processing encountered an error"
          : "Analysis complete",
      icon: status === "failed" ? XCircle : CheckCircle2,
      status: getStepStatus("complete"),
    },
  ];

  return (
    <div className={cn("relative", className)}>
      {/* Timeline line */}
      <div className="absolute left-4 top-6 bottom-6 w-0.5 bg-border" />

      {/* Steps */}
      <div className="space-y-0">
        {steps.map((step, index) => (
          <TimelineStepItem
            key={step.id}
            step={step}
            isLast={index === steps.length - 1}
          />
        ))}
      </div>
    </div>
  );
}

interface TimelineStepItemProps {
  step: TimelineStep;
  isLast: boolean;
}

function TimelineStepItem({ step, isLast }: TimelineStepItemProps) {
  const statusStyles = {
    completed: {
      icon: "bg-success text-success-foreground",
      text: "text-foreground",
      line: "bg-success",
    },
    current: {
      icon: "bg-primary text-primary-foreground animate-pulse",
      text: "text-foreground",
      line: "bg-primary",
    },
    pending: {
      icon: "bg-muted text-muted-foreground",
      text: "text-muted-foreground",
      line: "bg-border",
    },
    failed: {
      icon: "bg-danger text-danger-foreground",
      text: "text-danger",
      line: "bg-danger",
    },
  };

  const styles = statusStyles[step.status];
  const Icon = step.status === "completed" ? CheckCircle2 : step.icon;

  return (
    <div className="relative flex gap-4 pb-6 last:pb-0">
      {/* Icon */}
      <div
        className={cn(
          "relative z-10 flex h-8 w-8 items-center justify-center rounded-full",
          styles.icon
        )}
      >
        <Icon
          className={cn(
            "h-4 w-4",
            step.status === "current" && step.id === "processing" && "animate-spin"
          )}
        />
      </div>

      {/* Content */}
      <div className="flex-1 pt-1">
        <div className="flex items-center gap-2">
          <p className={cn("font-medium text-sm", styles.text)}>{step.label}</p>
          {step.status === "current" && (
            <span className="text-xs text-primary animate-pulse">In progress...</span>
          )}
        </div>
        <p className="text-xs text-muted-foreground mt-0.5">{step.description}</p>
      </div>

      {/* Status indicator */}
      <div className="pt-1">
        {step.status === "completed" && (
          <CheckCircle2 className="h-4 w-4 text-success" />
        )}
        {step.status === "current" && (
          <Clock className="h-4 w-4 text-primary animate-pulse" />
        )}
        {step.status === "failed" && (
          <XCircle className="h-4 w-4 text-danger" />
        )}
      </div>
    </div>
  );
}

// Compact horizontal version
export function ProcessingTimelineCompact({
  status,
  className,
}: Pick<ProcessingTimelineProps, "status" | "className">) {
  const steps = ["Upload", "Process", "Face", "PAD", "Doc", "Done"];
  
  const getStepIndex = () => {
    switch (status) {
      case "pending":
        return 1;
      case "processing":
        return 2;
      case "completed":
        return 6;
      case "failed":
        return -1;
      default:
        return 0;
    }
  };

  const currentStep = getStepIndex();

  return (
    <div className={cn("flex items-center gap-1", className)}>
      {steps.map((step, index) => {
        const stepNum = index + 1;
        const isCompleted = currentStep > stepNum;
        const isCurrent = currentStep === stepNum;
        const isFailed = status === "failed" && isCurrent;

        return (
          <React.Fragment key={step}>
            <div
              className={cn(
                "flex items-center justify-center w-6 h-6 rounded-full text-xs font-medium transition-colors",
                isCompleted && "bg-success text-success-foreground",
                isCurrent && !isFailed && "bg-primary text-primary-foreground",
                isFailed && "bg-danger text-danger-foreground",
                !isCompleted && !isCurrent && "bg-muted text-muted-foreground"
              )}
              title={step}
            >
              {isCompleted ? (
                <CheckCircle2 className="h-3.5 w-3.5" />
              ) : isFailed ? (
                <XCircle className="h-3.5 w-3.5" />
              ) : (
                stepNum
              )}
            </div>
            {index < steps.length - 1 && (
              <div
                className={cn(
                  "flex-1 h-0.5 min-w-[12px]",
                  isCompleted ? "bg-success" : "bg-border"
                )}
              />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}
