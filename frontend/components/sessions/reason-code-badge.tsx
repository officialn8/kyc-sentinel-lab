"use client";

import { AlertTriangle, Info } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Severity } from "@/lib/types";

interface ReasonCodeBadgeProps {
  code: string;
  severity: Severity;
  showIcon?: boolean;
}

const severityConfig = {
  info: {
    variant: "secondary" as const,
    icon: Info,
    iconColor: "text-blue-500",
  },
  warn: {
    variant: "warning" as const,
    icon: AlertTriangle,
    iconColor: "text-warning",
  },
  high: {
    variant: "danger" as const,
    icon: AlertTriangle,
    iconColor: "text-danger",
  },
};

export function ReasonCodeBadge({
  code,
  severity,
  showIcon = true,
}: ReasonCodeBadgeProps) {
  const config = severityConfig[severity];
  const Icon = config.icon;

  return (
    <Badge variant={config.variant} className="gap-1">
      {showIcon && <Icon className={cn("h-3 w-3", config.iconColor)} />}
      {code}
    </Badge>
  );
}






