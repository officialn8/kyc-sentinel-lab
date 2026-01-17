"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  ExternalLink,
} from "lucide-react";
import { Session } from "@/lib/api";
import { formatRelativeTime, truncateId, cn } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface SessionCardProps {
  session: Session;
}

const statusConfig = {
  pending: { icon: Clock, color: "text-muted-foreground", badge: "secondary" },
  processing: { icon: Clock, color: "text-warning", badge: "warning" },
  completed: { icon: CheckCircle2, color: "text-success", badge: "success" },
  failed: { icon: AlertTriangle, color: "text-danger", badge: "danger" },
} as const;

export function SessionCard({ session }: SessionCardProps) {
  const router = useRouter();
  const status = statusConfig[session.status];
  const StatusIcon = status.icon;

  // Prefetch on hover for faster navigation
  const handleMouseEnter = useCallback(() => {
    router.prefetch(`/sessions/${session.id}`);
  }, [router, session.id]);

  return (
    <Link href={`/sessions/${session.id}`} onMouseEnter={handleMouseEnter}>
      <Card className="glass-interactive cursor-pointer">
        <CardContent className="p-4">
          <div className="flex items-start justify-between">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <code className="text-sm font-mono">
                  {truncateId(session.id)}
                </code>
                <Badge variant={status.badge as any}>{session.status}</Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                {session.source}
                {session.attack_family && ` • ${session.attack_family}`}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">
                {formatRelativeTime(session.created_at)}
              </span>
              <ExternalLink className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}












