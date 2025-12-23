"use client";

import { useQuery } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import Image from "next/image";
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  Clock,
  FileText,
  Info,
  Loader2,
  Scan,
  User,
} from "lucide-react";
import Link from "next/link";
import { api, Reason } from "@/lib/api";
import { formatDate, getRiskColor, formatPercentage, cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const severityIcons = {
  info: Info,
  warn: AlertTriangle,
  high: AlertTriangle,
};

const severityColors = {
  info: "text-blue-500",
  warn: "text-warning",
  high: "text-danger",
};

export default function SessionDetailPage() {
  const params = useParams();
  const sessionId = params.id as string;

  const { data: session, isLoading } = useQuery({
    queryKey: ["session", sessionId],
    queryFn: () => api.getSession(sessionId),
    enabled: !!sessionId,
    // Poll every 2 seconds while processing
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.status === "processing" ? 2000 : false;
    },
  });

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="h-8 w-48 animate-pulse rounded bg-muted" />
        <div className="grid gap-6 md:grid-cols-2">
          <div className="h-64 animate-pulse rounded-lg bg-muted" />
          <div className="h-64 animate-pulse rounded-lg bg-muted" />
        </div>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">Session not found</p>
        <Button asChild className="mt-4">
          <Link href="/sessions">Back to Sessions</Link>
        </Button>
      </div>
    );
  }

  const result = session.result;
  const decisionConfig = {
    pass: { color: "text-success", bg: "bg-success/10", icon: CheckCircle2 },
    review: { color: "text-warning", bg: "bg-warning/10", icon: Clock },
    fail: { color: "text-danger", bg: "bg-danger/10", icon: AlertTriangle },
  };

  const decision = result?.decision
    ? decisionConfig[result.decision]
    : decisionConfig.review;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" asChild>
          <Link href="/sessions">
            <ArrowLeft className="h-5 w-5" />
          </Link>
        </Button>
        <div>
          <h1 className="text-2xl font-bold">Session Details</h1>
          <p className="text-sm text-muted-foreground font-mono">{session.id}</p>
        </div>
      </div>

      {/* Processing Banner */}
      {session.status === "processing" && (
        <Card className="glass border-l-4 border-l-primary bg-primary/5">
          <CardContent className="flex items-center gap-4 py-4">
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
            <div>
              <p className="font-semibold text-primary">Processing</p>
              <p className="text-sm text-muted-foreground">
                Running face matching, document analysis, and PAD checks...
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Status Banner */}
      {result && (
        <Card className={cn("glass border-l-4", decision.bg, decision.color.replace("text-", "border-l-"))}>
          <CardContent className="flex items-center justify-between py-4">
            <div className="flex items-center gap-3">
              <decision.icon className={cn("h-6 w-6", decision.color)} />
              <div>
                <p className={cn("font-semibold capitalize", decision.color)}>
                  {result.decision}
                </p>
                <p className="text-sm text-muted-foreground">
                  Risk Score: {result.risk_score}/100
                </p>
              </div>
            </div>
            <div className="flex items-center gap-6">
              <div className="text-right">
                <p className="text-xs text-muted-foreground">Model</p>
                <p className="text-sm font-mono">{result.model_version}</p>
              </div>
              <div className="text-right">
                <p className="text-xs text-muted-foreground">Rules</p>
                <p className="text-sm font-mono">{result.rules_version}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Left: Media Preview */}
        <div className="lg:col-span-2 space-y-6">
          <Card className="glass">
            <CardHeader>
              <CardTitle>Media Assets</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <p className="text-sm font-medium flex items-center gap-2">
                    <User className="h-4 w-4" />
                    Selfie
                  </p>
                  <div className="aspect-[3/4] rounded-lg bg-muted overflow-hidden">
                    {session.selfie_url ? (
                      <img
                        src={session.selfie_url}
                        alt="Selfie"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full text-muted-foreground">
                        No selfie uploaded
                      </div>
                    )}
                  </div>
                  {session.selfie_crop_url && (
                    <div className="mt-2">
                      <p className="text-xs text-muted-foreground mb-1">Face Crop</p>
                      <img
                        src={session.selfie_crop_url}
                        alt="Selfie face crop"
                        className="h-20 w-20 object-cover rounded border border-border"
                      />
                    </div>
                  )}
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium flex items-center gap-2">
                    <FileText className="h-4 w-4" />
                    ID Document
                  </p>
                  <div className="aspect-[3/4] rounded-lg bg-muted overflow-hidden">
                    {session.id_url ? (
                      <img
                        src={session.id_url}
                        alt="ID Document"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full text-muted-foreground">
                        No ID uploaded
                      </div>
                    )}
                  </div>
                  {session.id_crop_url && (
                    <div className="mt-2">
                      <p className="text-xs text-muted-foreground mb-1">Face Crop</p>
                      <img
                        src={session.id_crop_url}
                        alt="ID face crop"
                        className="h-20 w-20 object-cover rounded border border-border"
                      />
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Reason Codes */}
          {session.reasons.length > 0 && (
            <Card className="glass">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Scan className="h-5 w-5" />
                  Detection Reasons ({session.reasons.length})
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {session.reasons.map((reason: Reason) => {
                  const Icon = severityIcons[reason.severity];
                  return (
                    <div
                      key={reason.id}
                      className="flex items-start gap-3 p-3 rounded-lg bg-muted/50"
                    >
                      <Icon
                        className={cn(
                          "h-5 w-5 mt-0.5",
                          severityColors[reason.severity]
                        )}
                      />
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <code className="text-sm font-medium">
                            {reason.code}
                          </code>
                          <Badge
                            variant={
                              reason.severity === "high"
                                ? "danger"
                                : reason.severity === "warn"
                                ? "warning"
                                : "secondary"
                            }
                          >
                            {reason.severity}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">
                          {reason.message}
                        </p>
                        {Object.keys(reason.evidence).length > 0 && (
                          <pre className="text-xs text-muted-foreground mt-2 p-2 bg-muted rounded">
                            {JSON.stringify(reason.evidence, null, 2)}
                          </pre>
                        )}
                      </div>
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right: Details */}
        <div className="space-y-6">
          {/* Scores */}
          {result && (
            <Card className="glass">
              <CardHeader>
                <CardTitle>Component Scores</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Face Similarity</span>
                    <span className={getRiskColor(100 - (result.face_similarity || 0) * 100)}>
                      {formatPercentage(result.face_similarity || 0)}
                    </span>
                  </div>
                  <Progress value={(result.face_similarity || 0) * 100} />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>PAD Score (Suspicion)</span>
                    <span className={getRiskColor((result.pad_score || 0) * 100)}>
                      {formatPercentage(result.pad_score || 0)}
                    </span>
                  </div>
                  <Progress value={(result.pad_score || 0) * 100} className="[&>div]:bg-warning" />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Doc Score (Suspicion)</span>
                    <span className={getRiskColor((result.doc_score || 0) * 100)}>
                      {formatPercentage(result.doc_score || 0)}
                    </span>
                  </div>
                  <Progress value={(result.doc_score || 0) * 100} className="[&>div]:bg-warning" />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Metadata */}
          <Card className="glass">
            <CardHeader>
              <CardTitle>Session Info</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Status</span>
                <Badge
                  variant={
                    session.status === "completed"
                      ? "success"
                      : session.status === "failed"
                      ? "danger"
                      : "secondary"
                  }
                >
                  {session.status}
                </Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Source</span>
                <span>{session.source}</span>
              </div>
              {session.attack_family && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Attack Family</span>
                  <span className="capitalize">{session.attack_family}</span>
                </div>
              )}
              {session.attack_severity && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Severity</span>
                  <span className="capitalize">{session.attack_severity}</span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-muted-foreground">Created</span>
                <span className="text-sm">{formatDate(session.created_at)}</span>
              </div>
              {session.device_os && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Device OS</span>
                  <span>{session.device_os}</span>
                </div>
              )}
              {session.device_model && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Device</span>
                  <span>{session.device_model}</span>
                </div>
              )}
              {session.ip_country && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Country</span>
                  <span>{session.ip_country}</span>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}



