"use client";

import { motion, AnimatePresence } from "framer-motion";
import {
  CheckCircle2,
  Circle,
  Loader2,
  Radio,
  Upload,
  User,
  FileText,
  Shield,
  Calculator,
  Wifi,
  WifiOff,
} from "lucide-react";
import { ProgressRing } from "@/components/ui/animations";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { ConnectionState, ProcessingStep } from "@/lib/types/websocket";

interface LiveProgressBannerProps {
  progress: number;
  currentStep: ProcessingStep | null;
  message: string;
  connectionState: ConnectionState;
  className?: string;
}

interface StepConfig {
  key: ProcessingStep | "upload" | "complete";
  label: string;
  icon: typeof User;
  targetProgress: number;
}

const STEPS: StepConfig[] = [
  { key: "upload", label: "Upload", icon: Upload, targetProgress: 0 },
  { key: "face_analysis", label: "Face", icon: User, targetProgress: 20 },
  { key: "document_analysis", label: "Doc", icon: FileText, targetProgress: 40 },
  { key: "pad_analysis", label: "PAD", icon: Shield, targetProgress: 60 },
  { key: "scoring", label: "Score", icon: Calculator, targetProgress: 80 },
  { key: "complete", label: "Done", icon: CheckCircle2, targetProgress: 100 },
];

function getStepStatus(
  step: StepConfig,
  progress: number,
  currentStep: ProcessingStep | null
): "completed" | "current" | "pending" {
  // Upload is always completed when we're processing
  if (step.key === "upload") return "completed";

  // Complete step
  if (step.key === "complete") {
    return progress >= 100 ? "completed" : "pending";
  }

  // Current step based on WebSocket event
  if (currentStep === step.key) return "current";

  // Determine by progress threshold
  if (progress >= step.targetProgress + 20) return "completed";
  if (progress >= step.targetProgress) return "current";

  return "pending";
}

export function LiveProgressBanner({
  progress,
  currentStep,
  message,
  connectionState,
  className,
}: LiveProgressBannerProps) {
  const isConnected = connectionState === "connected";
  const isReconnecting = connectionState === "reconnecting";
  const isFallback = connectionState === "fallback";

  return (
    <Card className={cn("glass border-l-4 border-l-primary bg-primary/5", className)}>
      <CardContent className="py-6">
        <div className="flex flex-col md:flex-row md:items-center gap-6">
          {/* Progress Ring */}
          <div className="flex items-center gap-4">
            <ProgressRing
              progress={progress}
              size={80}
              strokeWidth={6}
              showPercentage
              className="flex-shrink-0"
            />
            <div className="md:hidden">
              <p className="font-semibold text-primary">Processing</p>
              <AnimatePresence mode="wait">
                <motion.p
                  key={message}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -5 }}
                  className="text-sm text-muted-foreground"
                >
                  {message || "Analyzing session..."}
                </motion.p>
              </AnimatePresence>
            </div>
          </div>

          {/* Status and Steps */}
          <div className="flex-1 space-y-4">
            {/* Desktop status message */}
            <div className="hidden md:flex items-center justify-between">
              <div>
                <p className="font-semibold text-primary">Processing</p>
                <AnimatePresence mode="wait">
                  <motion.p
                    key={message}
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -5 }}
                    className="text-sm text-muted-foreground"
                  >
                    {message || "Analyzing session..."}
                  </motion.p>
                </AnimatePresence>
              </div>

              {/* Connection Status Badge */}
              <Badge
                variant={isConnected ? "success" : isFallback ? "secondary" : "warning"}
                className="gap-1"
              >
                {isConnected ? (
                  <>
                    <Radio className="h-3 w-3 animate-pulse" />
                    Live
                  </>
                ) : isReconnecting ? (
                  <>
                    <Loader2 className="h-3 w-3 animate-spin" />
                    Reconnecting
                  </>
                ) : isFallback ? (
                  <>
                    <WifiOff className="h-3 w-3" />
                    Polling
                  </>
                ) : (
                  <>
                    <Wifi className="h-3 w-3" />
                    Connecting
                  </>
                )}
              </Badge>
            </div>

            {/* Step Indicators */}
            <div className="flex items-center justify-between gap-1">
              {STEPS.map((step, index) => {
                const status = getStepStatus(step, progress, currentStep);
                const Icon = step.icon;
                const isLast = index === STEPS.length - 1;

                return (
                  <div key={step.key} className="flex items-center flex-1">
                    {/* Step Circle */}
                    <div className="flex flex-col items-center gap-1">
                      <motion.div
                        className={cn(
                          "relative flex items-center justify-center w-8 h-8 rounded-full border-2 transition-colors",
                          status === "completed" &&
                            "bg-success border-success text-success-foreground",
                          status === "current" &&
                            "bg-primary border-primary text-primary-foreground",
                          status === "pending" &&
                            "bg-muted border-border text-muted-foreground"
                        )}
                        animate={
                          status === "current"
                            ? { scale: [1, 1.1, 1] }
                            : { scale: 1 }
                        }
                        transition={{
                          duration: 1.5,
                          repeat: status === "current" ? Infinity : 0,
                          ease: "easeInOut",
                        }}
                      >
                        {status === "completed" ? (
                          <CheckCircle2 className="h-4 w-4" />
                        ) : status === "current" ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Icon className="h-4 w-4" />
                        )}
                      </motion.div>
                      <span
                        className={cn(
                          "text-[10px] font-medium whitespace-nowrap",
                          status === "completed" && "text-success",
                          status === "current" && "text-primary",
                          status === "pending" && "text-muted-foreground"
                        )}
                      >
                        {step.label}
                      </span>
                    </div>

                    {/* Connector Line */}
                    {!isLast && (
                      <div className="flex-1 h-0.5 mx-1 mt-[-12px]">
                        <motion.div
                          className={cn(
                            "h-full rounded-full",
                            status === "completed" ? "bg-success" : "bg-border"
                          )}
                          initial={{ scaleX: 0 }}
                          animate={{ scaleX: 1 }}
                          style={{ originX: 0 }}
                          transition={{ duration: 0.3 }}
                        />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Mobile Connection Badge */}
        <div className="mt-4 flex justify-center md:hidden">
          <Badge
            variant={isConnected ? "success" : isFallback ? "secondary" : "warning"}
            className="gap-1"
          >
            {isConnected ? (
              <>
                <Radio className="h-3 w-3 animate-pulse" />
                Live Updates
              </>
            ) : isReconnecting ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin" />
                Reconnecting
              </>
            ) : isFallback ? (
              <>
                <WifiOff className="h-3 w-3" />
                Polling Mode
              </>
            ) : (
              <>
                <Wifi className="h-3 w-3" />
                Connecting
              </>
            )}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}
