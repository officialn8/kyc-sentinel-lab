"use client";

import * as React from "react";
import { ChevronDown, ChevronRight, Hash, Type, ToggleLeft, List, Braces } from "lucide-react";
import { cn } from "@/lib/utils";
import { Progress } from "@/components/ui/progress";

interface EvidenceViewerProps {
  evidence: Record<string, unknown>;
  className?: string;
}

export function EvidenceViewer({ evidence, className }: EvidenceViewerProps) {
  if (Object.keys(evidence).length === 0) {
    return (
      <p className="text-sm text-muted-foreground italic">No evidence data</p>
    );
  }

  return (
    <div className={cn("space-y-2", className)}>
      {Object.entries(evidence).map(([key, value]) => (
        <EvidenceItem key={key} name={key} value={value} />
      ))}
    </div>
  );
}

interface EvidenceItemProps {
  name: string;
  value: unknown;
  depth?: number;
}

function EvidenceItem({ name, value, depth = 0 }: EvidenceItemProps) {
  const [expanded, setExpanded] = React.useState(depth < 2);

  // Format the key for display
  const displayName = name
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(" ");

  // Determine the type and render accordingly
  const valueType = getValueType(value);

  // Render based on type
  if (valueType === "object" && value !== null) {
    const entries = Object.entries(value as Record<string, unknown>);
    return (
      <div className={cn("rounded-lg", depth > 0 && "ml-4")}>
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-2 text-sm font-medium hover:text-primary transition-colors w-full text-left py-1"
        >
          {expanded ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
          )}
          <Braces className="h-3.5 w-3.5 text-muted-foreground" />
          <span>{displayName}</span>
          <span className="text-muted-foreground font-normal">
            ({entries.length} {entries.length === 1 ? "field" : "fields"})
          </span>
        </button>
        {expanded && (
          <div className="pl-6 border-l border-border/50 ml-2 mt-1 space-y-1">
            {entries.map(([k, v]) => (
              <EvidenceItem key={k} name={k} value={v} depth={depth + 1} />
            ))}
          </div>
        )}
      </div>
    );
  }

  if (valueType === "array") {
    const arr = value as unknown[];
    return (
      <div className={cn("rounded-lg", depth > 0 && "ml-4")}>
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-2 text-sm font-medium hover:text-primary transition-colors w-full text-left py-1"
        >
          {expanded ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
          )}
          <List className="h-3.5 w-3.5 text-muted-foreground" />
          <span>{displayName}</span>
          <span className="text-muted-foreground font-normal">
            ({arr.length} {arr.length === 1 ? "item" : "items"})
          </span>
        </button>
        {expanded && (
          <div className="pl-6 border-l border-border/50 ml-2 mt-1 space-y-1">
            {arr.map((item, idx) => (
              <EvidenceItem key={idx} name={`[${idx}]`} value={item} depth={depth + 1} />
            ))}
          </div>
        )}
      </div>
    );
  }

  // Primitive values
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-4 py-1.5 px-2 rounded hover:bg-muted/50 transition-colors",
        depth > 0 && "ml-4"
      )}
    >
      <div className="flex items-center gap-2 text-sm min-w-0">
        <TypeIcon type={valueType} />
        <span className="text-muted-foreground truncate">{displayName}</span>
      </div>
      <ValueDisplay value={value} type={valueType} name={name} />
    </div>
  );
}

function getValueType(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value === "number") return "number";
  if (typeof value === "boolean") return "boolean";
  if (typeof value === "string") return "string";
  if (typeof value === "object") return "object";
  return "unknown";
}

function TypeIcon({ type }: { type: string }) {
  switch (type) {
    case "number":
      return <Hash className="h-3.5 w-3.5 text-blue-500 flex-shrink-0" />;
    case "string":
      return <Type className="h-3.5 w-3.5 text-green-500 flex-shrink-0" />;
    case "boolean":
      return <ToggleLeft className="h-3.5 w-3.5 text-purple-500 flex-shrink-0" />;
    default:
      return <div className="w-3.5" />;
  }
}

function ValueDisplay({
  value,
  type,
  name,
}: {
  value: unknown;
  type: string;
  name: string;
}) {
  // Special handling for confidence/score/similarity values (show as progress bar)
  if (
    type === "number" &&
    (name.includes("score") ||
      name.includes("confidence") ||
      name.includes("similarity") ||
      name.includes("rate") ||
      name.includes("ratio"))
  ) {
    const numValue = value as number;
    // Assume values 0-1 are percentages, >1 might be actual percentages or raw scores
    const percentage = numValue <= 1 ? numValue * 100 : numValue;
    const clampedPercentage = Math.min(100, Math.max(0, percentage));

    return (
      <div className="flex items-center gap-2 min-w-[120px]">
        <Progress value={clampedPercentage} className="h-2 flex-1" />
        <span className="text-sm font-mono w-12 text-right">
          {numValue <= 1 ? (numValue * 100).toFixed(1) : numValue.toFixed(1)}%
        </span>
      </div>
    );
  }

  // Number formatting
  if (type === "number") {
    const num = value as number;
    return (
      <span className="text-sm font-mono text-blue-400">
        {Number.isInteger(num) ? num : num.toFixed(4)}
      </span>
    );
  }

  // Boolean
  if (type === "boolean") {
    return (
      <span
        className={cn(
          "text-sm font-medium",
          value ? "text-success" : "text-muted-foreground"
        )}
      >
        {value ? "true" : "false"}
      </span>
    );
  }

  // String
  if (type === "string") {
    const str = value as string;
    // Truncate long strings
    const displayStr = str.length > 50 ? str.slice(0, 50) + "..." : str;
    return (
      <span className="text-sm text-green-400 font-mono truncate max-w-[200px]">
        "{displayStr}"
      </span>
    );
  }

  // Null
  if (type === "null") {
    return <span className="text-sm text-muted-foreground italic">null</span>;
  }

  return <span className="text-sm">{String(value)}</span>;
}
