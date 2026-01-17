"use client";

import Link from "next/link";
import {
  ArrowRight,
  CheckCircle2,
  FileUp,
  Shield,
  Zap,
  BarChart3,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface Step {
  id: number;
  title: string;
  description: string;
  icon: React.ElementType;
  href: string;
  action: string;
  completed?: boolean;
}

const steps: Step[] = [
  {
    id: 1,
    title: "Upload a Session",
    description: "Upload a real KYC session with selfie and ID document for analysis",
    icon: FileUp,
    href: "/upload",
    action: "Upload",
  },
  {
    id: 2,
    title: "Generate Attack Simulations",
    description: "Create synthetic attacks to test your detection capabilities",
    icon: Zap,
    href: "/simulate",
    action: "Simulate",
  },
  {
    id: 3,
    title: "Review Detection Results",
    description: "Analyze results with explainable reason codes and risk scores",
    icon: Shield,
    href: "/sessions",
    action: "View Sessions",
  },
  {
    id: 4,
    title: "Explore Metrics",
    description: "View confusion matrices and score distributions",
    icon: BarChart3,
    href: "/metrics",
    action: "View Metrics",
  },
];

interface GettingStartedProps {
  completedSteps?: number[];
  className?: string;
  onDismiss?: () => void;
}

export function GettingStarted({
  completedSteps = [],
  className,
  onDismiss,
}: GettingStartedProps) {
  const stepsWithStatus = steps.map((step) => ({
    ...step,
    completed: completedSteps.includes(step.id),
  }));

  const progress = (completedSteps.length / steps.length) * 100;

  return (
    <Card className={cn("glass overflow-hidden", className)}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/20">
              <Shield className="h-4 w-4 text-primary" />
            </div>
            Getting Started
          </CardTitle>
          {onDismiss && (
            <Button variant="ghost" size="sm" onClick={onDismiss}>
              Dismiss
            </Button>
          )}
        </div>
        <p className="text-sm text-muted-foreground">
          Follow these steps to get the most out of KYC Sentinel Lab
        </p>
        {/* Progress bar */}
        <div className="mt-3 h-2 w-full rounded-full bg-muted overflow-hidden">
          <div
            className="h-full bg-primary transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          {completedSteps.length} of {steps.length} steps completed
        </p>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 sm:grid-cols-2">
          {stepsWithStatus.map((step) => (
            <Link
              key={step.id}
              href={step.href}
              className={cn(
                "group relative flex items-start gap-4 rounded-lg border p-4 transition-all",
                step.completed
                  ? "border-success/20 bg-success/5"
                  : "border-border/50 hover:border-primary/50 hover:bg-primary/5"
              )}
            >
              {/* Step number / check */}
              <div
                className={cn(
                  "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg",
                  step.completed
                    ? "bg-success/20 text-success"
                    : "bg-primary/10 text-primary"
                )}
              >
                {step.completed ? (
                  <CheckCircle2 className="h-5 w-5" />
                ) : (
                  <step.icon className="h-5 w-5" />
                )}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <h4 className="font-medium text-sm">{step.title}</h4>
                  {step.completed && (
                    <span className="text-xs text-success font-medium">Done</span>
                  )}
                </div>
                <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                  {step.description}
                </p>
              </div>

              {/* Arrow */}
              <ArrowRight
                className={cn(
                  "h-4 w-4 shrink-0 transition-transform",
                  step.completed
                    ? "text-success"
                    : "text-muted-foreground group-hover:translate-x-1 group-hover:text-primary"
                )}
              />
            </Link>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
