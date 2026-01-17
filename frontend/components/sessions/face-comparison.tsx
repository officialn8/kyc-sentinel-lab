"use client";

import * as React from "react";
import { ArrowLeftRight, CheckCircle2, XCircle, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import { Progress } from "@/components/ui/progress";
import { LightboxTrigger } from "@/components/ui/lightbox";

interface FaceComparisonProps {
  selfieUrl?: string | null;
  selfieCropUrl?: string | null;
  idUrl?: string | null;
  idCropUrl?: string | null;
  similarity?: number | null;
  className?: string;
}

export function FaceComparison({
  selfieUrl,
  selfieCropUrl,
  idUrl,
  idCropUrl,
  similarity,
  className,
}: FaceComparisonProps) {
  const hasFaceCrops = selfieCropUrl && idCropUrl;
  const similarityPercent = similarity ? similarity * 100 : null;

  // Determine match status
  const getMatchStatus = () => {
    if (similarityPercent === null) return null;
    if (similarityPercent >= 70) return "match";
    if (similarityPercent >= 45) return "uncertain";
    return "mismatch";
  };

  const matchStatus = getMatchStatus();

  const statusConfig = {
    match: {
      icon: CheckCircle2,
      label: "Face Match",
      color: "text-success",
      bgColor: "bg-success/10",
      borderColor: "border-success/30",
    },
    uncertain: {
      icon: AlertTriangle,
      label: "Uncertain Match",
      color: "text-warning",
      bgColor: "bg-warning/10",
      borderColor: "border-warning/30",
    },
    mismatch: {
      icon: XCircle,
      label: "Face Mismatch",
      color: "text-danger",
      bgColor: "bg-danger/10",
      borderColor: "border-danger/30",
    },
  };

  const status = matchStatus ? statusConfig[matchStatus] : null;

  if (!hasFaceCrops) {
    return (
      <div className={cn("rounded-lg border border-border/50 bg-muted/30 p-6", className)}>
        <div className="flex items-center justify-center gap-4 text-muted-foreground">
          <div className="w-20 h-20 rounded-full bg-muted flex items-center justify-center">
            <span className="text-2xl">?</span>
          </div>
          <ArrowLeftRight className="h-6 w-6" />
          <div className="w-20 h-20 rounded-full bg-muted flex items-center justify-center">
            <span className="text-2xl">?</span>
          </div>
        </div>
        <p className="text-center text-sm text-muted-foreground mt-4">
          Face crops not available
        </p>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "rounded-lg border p-4",
        status ? status.borderColor : "border-border/50",
        status ? status.bgColor : "bg-card/30",
        className
      )}
    >
      {/* Status header */}
      {status && (
        <div className={cn("flex items-center gap-2 mb-4", status.color)}>
          <status.icon className="h-5 w-5" />
          <span className="font-medium">{status.label}</span>
        </div>
      )}

      {/* Face comparison */}
      <div className="flex items-center justify-center gap-4">
        {/* Selfie face */}
        <div className="text-center">
          <LightboxTrigger
            src={selfieCropUrl!}
            alt="Selfie face crop"
            title="Selfie Face"
          >
            <div className="relative group">
              <img
                src={selfieCropUrl!}
                alt="Selfie face"
                className="w-24 h-24 md:w-28 md:h-28 rounded-full object-cover border-2 border-border shadow-lg group-hover:border-primary transition-colors"
              />
              <div className="absolute inset-0 rounded-full bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
                <span className="text-white opacity-0 group-hover:opacity-100 text-xs font-medium">
                  Click to zoom
                </span>
              </div>
            </div>
          </LightboxTrigger>
          <p className="text-xs text-muted-foreground mt-2">Selfie</p>
        </div>

        {/* Similarity indicator */}
        <div className="flex flex-col items-center gap-2 px-4">
          <ArrowLeftRight className={cn("h-6 w-6", status?.color || "text-muted-foreground")} />
          {similarityPercent !== null && (
            <div className="text-center">
              <span className={cn("text-2xl font-bold", status?.color)}>
                {similarityPercent.toFixed(0)}%
              </span>
              <p className="text-xs text-muted-foreground">Similarity</p>
            </div>
          )}
        </div>

        {/* ID face */}
        <div className="text-center">
          <LightboxTrigger
            src={idCropUrl!}
            alt="ID document face crop"
            title="ID Document Face"
          >
            <div className="relative group">
              <img
                src={idCropUrl!}
                alt="ID face"
                className="w-24 h-24 md:w-28 md:h-28 rounded-full object-cover border-2 border-border shadow-lg group-hover:border-primary transition-colors"
              />
              <div className="absolute inset-0 rounded-full bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
                <span className="text-white opacity-0 group-hover:opacity-100 text-xs font-medium">
                  Click to zoom
                </span>
              </div>
            </div>
          </LightboxTrigger>
          <p className="text-xs text-muted-foreground mt-2">ID Document</p>
        </div>
      </div>

      {/* Similarity progress bar */}
      {similarityPercent !== null && (
        <div className="mt-4 space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Mismatch</span>
            <span>Match</span>
          </div>
          <div className="relative">
            <Progress
              value={similarityPercent}
              className={cn(
                "h-2",
                matchStatus === "match" && "[&>div]:bg-success",
                matchStatus === "uncertain" && "[&>div]:bg-warning",
                matchStatus === "mismatch" && "[&>div]:bg-danger"
              )}
            />
            {/* Threshold markers */}
            <div className="absolute top-0 left-[45%] w-px h-2 bg-border" title="Threshold: 45%" />
            <div className="absolute top-0 left-[70%] w-px h-2 bg-border" title="High confidence: 70%" />
          </div>
          <div className="flex justify-between text-[10px] text-muted-foreground/70">
            <span>0%</span>
            <span>45%</span>
            <span>70%</span>
            <span>100%</span>
          </div>
        </div>
      )}
    </div>
  );
}
