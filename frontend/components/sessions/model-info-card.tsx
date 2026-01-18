"use client";

import { Cpu, User, FileText, Scale, Info } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

interface ModelInfoCardProps {
  modelVersion: string;
  rulesVersion: string;
  className?: string;
}

interface ParsedModels {
  face: { engine: string; model: string };
  doc: { engine: string; model: string };
}

interface ScoringProfile {
  name: string;
  description: string;
  weights: { face: number; pad: number; doc: number };
  thresholds: { fail: number; review: number };
}

const SCORING_PROFILES: Record<string, ScoringProfile> = {
  default: {
    name: "Default",
    description: "Balanced scoring for general KYC verification",
    weights: { face: 45, pad: 35, doc: 20 },
    thresholds: { fail: 70, review: 40 },
  },
  fintech_high_risk: {
    name: "Fintech High Risk",
    description: "Stricter scoring for high-value financial services",
    weights: { face: 50, pad: 35, doc: 15 },
    thresholds: { fail: 60, review: 35 },
  },
  crypto_exchange: {
    name: "Crypto Exchange",
    description: "Enhanced fraud detection for cryptocurrency platforms",
    weights: { face: 40, pad: 40, doc: 20 },
    thresholds: { fail: 55, review: 30 },
  },
  social_verification: {
    name: "Social Verification",
    description: "Lightweight verification for social platforms",
    weights: { face: 60, pad: 25, doc: 15 },
    thresholds: { fail: 70, review: 45 },
  },
};

function parseModelVersion(version: string): ParsedModels {
  // Input: "insightface-buffalo_l,paddleocr-en"
  const parts = version.split(",");

  const parsePart = (part: string) => {
    const segments = part.trim().split("-");
    return {
      engine: segments[0] || "unknown",
      model: segments.slice(1).join("-") || "unknown",
    };
  };

  return {
    face: parts[0] ? parsePart(parts[0]) : { engine: "insightface", model: "unknown" },
    doc: parts[1] ? parsePart(parts[1]) : { engine: "paddleocr", model: "unknown" },
  };
}

function formatEngineName(engine: string): string {
  const names: Record<string, string> = {
    insightface: "InsightFace",
    paddleocr: "PaddleOCR",
  };
  return names[engine.toLowerCase()] || engine;
}

function formatModelName(model: string): string {
  // buffalo_l -> Buffalo L
  // en -> English
  const languageNames: Record<string, string> = {
    en: "English",
    ch: "Chinese",
    fr: "French",
    de: "German",
    es: "Spanish",
    pt: "Portuguese",
    ar: "Arabic",
    ru: "Russian",
    ja: "Japanese",
    ko: "Korean",
  };

  if (languageNames[model]) return languageNames[model];

  return model
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export function ModelInfoCard({
  modelVersion,
  rulesVersion,
  className,
}: ModelInfoCardProps) {
  const models = parseModelVersion(modelVersion);
  const profile = SCORING_PROFILES[rulesVersion] || SCORING_PROFILES.default;

  return (
    <TooltipProvider>
      <Card className={cn("glass", className)}>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Cpu className="h-4 w-4" />
            Detection Models
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Model Cards */}
          <div className="grid grid-cols-2 gap-3">
            {/* Face Analysis */}
            <div className="p-3 rounded-lg bg-muted/50 border border-border/30">
              <div className="flex items-center gap-2 mb-2">
                <User className="h-4 w-4 text-muted-foreground" />
                <span className="text-xs font-medium text-muted-foreground">
                  Face
                </span>
              </div>
              <p className="text-sm font-medium mb-1">
                {formatEngineName(models.face.engine)}
              </p>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge variant="secondary" className="font-mono text-xs cursor-help">
                    {models.face.model}
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Model: {formatModelName(models.face.model)}</p>
                  <p className="text-muted-foreground text-xs mt-1">
                    Face detection and embedding model
                  </p>
                </TooltipContent>
              </Tooltip>
            </div>

            {/* Document Analysis */}
            <div className="p-3 rounded-lg bg-muted/50 border border-border/30">
              <div className="flex items-center gap-2 mb-2">
                <FileText className="h-4 w-4 text-muted-foreground" />
                <span className="text-xs font-medium text-muted-foreground">
                  Document
                </span>
              </div>
              <p className="text-sm font-medium mb-1">
                {formatEngineName(models.doc.engine)}
              </p>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge variant="secondary" className="font-mono text-xs cursor-help">
                    {models.doc.model}
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Language: {formatModelName(models.doc.model)}</p>
                  <p className="text-muted-foreground text-xs mt-1">
                    OCR language model for document text extraction
                  </p>
                </TooltipContent>
              </Tooltip>
            </div>
          </div>

          {/* Scoring Profile */}
          <div className="p-3 rounded-lg bg-muted/50 border border-border/30">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Scale className="h-4 w-4 text-muted-foreground" />
                <span className="text-xs font-medium text-muted-foreground">
                  Scoring Profile
                </span>
              </div>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge className="cursor-help">{profile.name}</Badge>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  <p className="font-medium">{profile.name}</p>
                  <p className="text-muted-foreground text-xs mt-1">
                    {profile.description}
                  </p>
                </TooltipContent>
              </Tooltip>
            </div>
            <p className="text-xs text-muted-foreground mb-3">
              {profile.description}
            </p>

            {/* Weights */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Weights</span>
                <span className="font-mono">
                  Face {profile.weights.face}% · PAD {profile.weights.pad}% · Doc{" "}
                  {profile.weights.doc}%
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Thresholds</span>
                <span className="font-mono">
                  <span className="text-danger">Fail ≥{profile.thresholds.fail}</span>
                  {" · "}
                  <span className="text-warning">Review ≥{profile.thresholds.review}</span>
                </span>
              </div>
            </div>
          </div>

          {/* Raw Version Info */}
          <div className="pt-2 border-t border-border/30">
            <Tooltip>
              <TooltipTrigger asChild>
                <button className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors">
                  <Info className="h-3 w-3" />
                  Version details
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom" className="font-mono text-xs">
                <p>Model: {modelVersion}</p>
                <p>Rules: {rulesVersion}</p>
              </TooltipContent>
            </Tooltip>
          </div>
        </CardContent>
      </Card>
    </TooltipProvider>
  );
}
