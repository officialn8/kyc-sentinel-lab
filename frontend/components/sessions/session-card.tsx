"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useRef, useState } from "react";
import { motion, useMotionValue, useTransform } from "framer-motion";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  ExternalLink,
  Loader2,
  ShieldAlert,
} from "lucide-react";
import { Session } from "@/lib/api";
import { formatRelativeTime, truncateId, cn, getRiskColor } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { animations } from "@/lib/animations";

interface SessionCardProps {
  session: Session & { result?: { risk_score?: number; decision?: string } };
}

const statusConfig = {
  pending: { icon: Clock, color: "text-muted-foreground", badge: "secondary", glow: false },
  processing: { icon: Loader2, color: "text-warning", badge: "warning", glow: true },
  completed: { icon: CheckCircle2, color: "text-success", badge: "success", glow: false },
  failed: { icon: ShieldAlert, color: "text-danger", badge: "danger", glow: true },
} as const;

export function SessionCard({ session }: SessionCardProps) {
  const router = useRouter();
  const cardRef = useRef<HTMLDivElement>(null);
  const status = statusConfig[session.status];
  const StatusIcon = status.icon;

  // 3D tilt effect
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const rotateX = useTransform(y, [-100, 100], [15, -15]);
  const rotateY = useTransform(x, [-100, 100], [-15, 15]);

  // Handle mouse move for 3D effect
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return;

    const rect = cardRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    x.set((e.clientX - centerX) / 5);
    y.set((e.clientY - centerY) / 5);
  }, [x, y]);

  const handleMouseLeave = useCallback(() => {
    x.set(0);
    y.set(0);
  }, [x, y]);

  // Prefetch on hover for faster navigation
  const handleMouseEnter = useCallback(() => {
    router.prefetch(`/sessions/${session.id}`);
  }, [router, session.id]);

  const riskScore = session.result?.risk_score;
  const decision = session.result?.decision;

  return (
    <Link href={`/sessions/${session.id}`} onMouseEnter={handleMouseEnter}>
      <motion.div
        ref={cardRef}
        className="relative cursor-pointer"
        style={{
          rotateX,
          rotateY,
          transformStyle: "preserve-3d",
        }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        whileHover="hover"
        whileTap="tap"
        variants={animations.cardHoverVariants}
        initial="idle"
      >
        <Card className="glass-soft shadow-elevation-2 hover:shadow-elevation-4 transition-all duration-300 overflow-hidden group">
          {/* Animated gradient overlay */}
          <motion.div
            className="absolute inset-0 opacity-0 group-hover:opacity-100 pointer-events-none"
            style={{
              background: `radial-gradient(circle at 50% 50%, rgba(var(--primary-rgb), 0.1) 0%, transparent 50%)`,
            }}
            transition={{ duration: 0.3 }}
          />

          <CardContent className="p-5 relative z-10">
            <div className="flex items-start justify-between gap-3">
              <div className="space-y-2 flex-1">
                <div className="flex items-center gap-3">
                  {/* Animated ID with monospace font */}
                  <code className="text-sm font-mono bg-muted/30 px-2 py-1 rounded">
                    {truncateId(session.id)}
                  </code>

                  {/* Status badge with conditional animations */}
                  <motion.div
                    animate={session.status === "processing" ? { scale: [1, 1.1, 1] } : {}}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    <Badge
                      variant={status.badge as any}
                      className={cn(
                        "gap-1.5",
                        status.glow && "animate-glow-pulse"
                      )}
                    >
                      <StatusIcon className={cn(
                        "h-3 w-3",
                        session.status === "processing" && "animate-spin"
                      )} />
                      {session.status}
                    </Badge>
                  </motion.div>
                </div>

                {/* Session details */}
                <div className="flex items-center gap-3 text-sm text-muted-foreground">
                  <span className="font-medium">{session.source}</span>
                  {session.attack_family && (
                    <>
                      <span className="text-muted-foreground/50">•</span>
                      <span className="text-accent font-medium">{session.attack_family}</span>
                    </>
                  )}
                </div>

                {/* Risk score and decision (if available) */}
                {session.result && (
                  <motion.div
                    className="flex items-center gap-3"
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={animations.springs.smooth}
                  >
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs text-muted-foreground">Risk:</span>
                      <motion.span
                        className={cn(
                          "font-semibold text-sm",
                          getRiskColor(riskScore || 0)
                        )}
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={animations.springs.bouncy}
                      >
                        {riskScore}%
                      </motion.span>
                    </div>
                    {decision && (
                      <Badge
                        variant={
                          decision === "pass" ? "success" :
                          decision === "review" ? "warning" : "danger"
                        }
                        className="text-xs"
                      >
                        {decision.toUpperCase()}
                      </Badge>
                    )}
                  </motion.div>
                )}
              </div>

              {/* Right side: Time and arrow */}
              <div className="flex flex-col items-end gap-2">
                <span className="text-xs text-muted-foreground">
                  {formatRelativeTime(session.created_at)}
                </span>
                <motion.div
                  className="opacity-0 group-hover:opacity-100"
                  initial={{ x: -10, opacity: 0 }}
                  whileHover={{ x: 0, opacity: 1 }}
                  transition={animations.springs.snappy}
                >
                  <ExternalLink className="h-4 w-4 text-primary" />
                </motion.div>
              </div>
            </div>
          </CardContent>

          {/* Bottom gradient fade for depth */}
          <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-border to-transparent" />
        </Card>
      </motion.div>
    </Link>
  );
}












