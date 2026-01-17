"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  FileText,
  Loader2,
  Monitor,
  Zap,
  Shield,
} from "lucide-react";
import { api, Session, SessionListResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "@/components/ui/use-toast";

const attackIcons: Record<string, React.ElementType> = {
  replay: Monitor,
  injection: Camera,
  face_swap: AlertTriangle,
  doc_tamper: FileText,
  benign: Shield,
};

const attackColors: Record<string, string> = {
  replay: "border-orange-500/50 hover:border-orange-500",
  injection: "border-purple-500/50 hover:border-purple-500",
  face_swap: "border-red-500/50 hover:border-red-500",
  doc_tamper: "border-yellow-500/50 hover:border-yellow-500",
  benign: "border-green-500/50 hover:border-green-500",
};

const attackBg: Record<string, string> = {
  replay: "bg-orange-500/10",
  injection: "bg-purple-500/10",
  face_swap: "bg-red-500/10",
  doc_tamper: "bg-yellow-500/10",
  benign: "bg-green-500/10",
};

// Generate a temporary ID for optimistic updates
function generateTempId() {
  return `temp-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// Create optimistic session placeholder
function createOptimisticSession(
  attackFamily: string,
  attackSeverity: string
): Session {
  const now = new Date().toISOString();
  return {
    id: generateTempId(),
    created_at: now,
    updated_at: now,
    status: "processing",
    source: "synthetic",
    attack_family: attackFamily,
    attack_severity: attackSeverity,
  };
}

export default function SimulatePage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [selectedFamily, setSelectedFamily] = useState<string>("");
  const [severity, setSeverity] = useState("medium");
  const [count, setCount] = useState("1");
  const [generationProgress, setGenerationProgress] = useState<{
    current: number;
    total: number;
    status: "idle" | "generating" | "processing" | "complete";
  }>({ current: 0, total: 0, status: "idle" });

  const { data: families, isLoading: loadingFamilies } = useQuery({
    queryKey: ["attack-families"],
    queryFn: () => api.listAttackFamilies(),
  });

  const generateMutation = useMutation({
    mutationFn: async () => {
      const requestCount = parseInt(count);
      setGenerationProgress({
        current: 0,
        total: requestCount,
        status: "generating",
      });

      // Call the API
      const sessions = await api.generateSyntheticSessions({
        attack_family: selectedFamily,
        attack_severity: severity,
        count: requestCount,
      });

      // Update progress to processing
      setGenerationProgress({
        current: requestCount,
        total: requestCount,
        status: "processing",
      });

      return sessions;
    },
    onMutate: async () => {
      // Cancel any outgoing refetches
      await queryClient.cancelQueries({ queryKey: ["sessions"] });

      // Snapshot the previous value
      const previousSessions = queryClient.getQueryData<SessionListResponse>([
        "sessions",
        1,
        { status: "", source: "", attack_family: "" },
      ]);

      // Create optimistic sessions
      const requestCount = parseInt(count);
      const optimisticSessions: Session[] = Array.from({
        length: requestCount,
      }).map(() => createOptimisticSession(selectedFamily, severity));

      // Optimistically update the cache
      queryClient.setQueryData<SessionListResponse>(
        ["sessions", 1, { status: "", source: "", attack_family: "" }],
        (old) => {
          if (!old) {
            return {
              items: optimisticSessions,
              total: requestCount,
              page: 1,
              page_size: 20,
              pages: 1,
            };
          }
          return {
            ...old,
            items: [...optimisticSessions, ...old.items].slice(0, 20),
            total: old.total + requestCount,
          };
        }
      );

      return { previousSessions, optimisticSessions };
    },
    onSuccess: (sessions, _, context) => {
      // Update progress to complete
      setGenerationProgress((prev) => ({
        ...prev,
        status: "complete",
      }));

      // Replace optimistic sessions with real ones
      if (context?.optimisticSessions) {
        queryClient.setQueryData<SessionListResponse>(
          ["sessions", 1, { status: "", source: "", attack_family: "" }],
          (old) => {
            if (!old) return old;

            // Remove optimistic sessions and add real ones
            const filteredItems = old.items.filter(
              (item) => !item.id.startsWith("temp-")
            );
            return {
              ...old,
              items: [...sessions, ...filteredItems].slice(0, 20),
            };
          }
        );
      }

      // Invalidate to ensure fresh data
      queryClient.invalidateQueries({ queryKey: ["sessions"] });

      toast({
        title: "Sessions generated",
        description: `Created ${sessions.length} synthetic session(s). Processing has started.`,
      });

      // Navigate after a brief delay to show the complete state
      setTimeout(() => {
        setGenerationProgress({ current: 0, total: 0, status: "idle" });
        if (sessions.length === 1) {
          router.push(`/sessions/${sessions[0].id}`);
        } else {
          router.push("/sessions");
        }
      }, 500);
    },
    onError: (error, _, context) => {
      // Rollback to the previous value
      if (context?.previousSessions) {
        queryClient.setQueryData(
          ["sessions", 1, { status: "", source: "", attack_family: "" }],
          context.previousSessions
        );
      }

      setGenerationProgress({ current: 0, total: 0, status: "idle" });

      toast({
        title: "Generation failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const isGenerating = generationProgress.status !== "idle";
  const progressPercent =
    generationProgress.total > 0
      ? generationProgress.status === "complete"
        ? 100
        : generationProgress.status === "processing"
          ? 75
          : (generationProgress.current / generationProgress.total) * 50
      : 0;

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Zap className="h-6 w-6 text-primary" />
          Attack Simulator
        </h1>
        <p className="text-muted-foreground">
          Generate synthetic KYC sessions with various attack patterns
        </p>
      </div>

      {/* Attack Family Selection */}
      <div className="space-y-4">
        <Label className="text-base">Select Attack Family</Label>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {loadingFamilies
            ? Array.from({ length: 5 }).map((_, i) => (
                <div
                  key={i}
                  className="h-32 animate-pulse rounded-lg bg-muted"
                />
              ))
            : families?.map((family) => {
                const Icon = attackIcons[family.id] || AlertTriangle;
                const isSelected = selectedFamily === family.id;
                return (
                  <Card
                    key={family.id}
                    className={cn(
                      "cursor-pointer transition-all glass border-2",
                      attackColors[family.id],
                      isSelected && attackBg[family.id],
                      isSelected && "ring-2 ring-primary",
                      isGenerating && "pointer-events-none opacity-50"
                    )}
                    onClick={() => !isGenerating && setSelectedFamily(family.id)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-start gap-3">
                        <div
                          className={cn(
                            "p-2 rounded-lg",
                            attackBg[family.id]
                          )}
                        >
                          <Icon className="h-5 w-5" />
                        </div>
                        <div>
                          <h3 className="font-medium">{family.name}</h3>
                          <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                            {family.description}
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
        </div>
      </div>

      {/* Configuration */}
      <Card className={cn("glass", isGenerating && "opacity-50")}>
        <CardHeader>
          <CardTitle className="text-base">Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-6 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="severity">Artifact Severity</Label>
              <Select
                value={severity}
                onValueChange={setSeverity}
                disabled={isGenerating}
              >
                <SelectTrigger id="severity">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="low">
                    Low - Subtle artifacts, harder to detect
                  </SelectItem>
                  <SelectItem value="medium">
                    Medium - Moderate artifacts
                  </SelectItem>
                  <SelectItem value="high">
                    High - Obvious artifacts, easier to detect
                  </SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Higher severity creates more obvious attack artifacts
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="count">Number of Sessions</Label>
              <Select
                value={count}
                onValueChange={setCount}
                disabled={isGenerating}
              >
                <SelectTrigger id="count">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">1 session</SelectItem>
                  <SelectItem value="3">3 sessions</SelectItem>
                  <SelectItem value="5">5 sessions</SelectItem>
                  <SelectItem value="10">10 sessions</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Generate multiple sessions for testing
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Generation Progress */}
      {isGenerating && (
        <Card className="glass border-primary/50 bg-primary/5">
          <CardContent className="py-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {generationProgress.status === "complete" ? (
                    <CheckCircle2 className="h-5 w-5 text-success" />
                  ) : (
                    <Loader2 className="h-5 w-5 animate-spin text-primary" />
                  )}
                  <div>
                    <p className="font-medium">
                      {generationProgress.status === "generating"
                        ? "Generating synthetic sessions..."
                        : generationProgress.status === "processing"
                          ? "Processing detection pipeline..."
                          : "Complete!"}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {generationProgress.status === "generating"
                        ? "Creating synthetic images with attack artifacts"
                        : generationProgress.status === "processing"
                          ? "Running face matching, PAD, and document analysis"
                          : "Redirecting to sessions..."}
                    </p>
                  </div>
                </div>
                <span className="text-sm font-mono text-muted-foreground">
                  {generationProgress.current}/{generationProgress.total}
                </span>
              </div>
              <Progress value={progressPercent} className="h-2" />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Info Box */}
      <Card className="glass border-primary/30 bg-primary/5">
        <CardContent className="p-4">
          <div className="flex gap-3">
            <Shield className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium">Safe Synthetic Data</p>
              <p className="text-xs text-muted-foreground mt-1">
                These sessions use synthetic artifacts that simulate attack
                patterns without using actual deepfakes or personal data. Perfect
                for red-team testing your detection capabilities.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Submit */}
      <div className="flex justify-end gap-4">
        <Button
          variant="outline"
          onClick={() => router.push("/sessions")}
          disabled={isGenerating}
        >
          Cancel
        </Button>
        <Button
          size="lg"
          disabled={!selectedFamily || isGenerating}
          onClick={() => generateMutation.mutate()}
        >
          {isGenerating ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              {generationProgress.status === "generating"
                ? "Generating..."
                : generationProgress.status === "processing"
                  ? "Processing..."
                  : "Complete!"}
            </>
          ) : (
            <>
              <Zap className="mr-2 h-4 w-4" />
              Generate {count} Session{parseInt(count) > 1 ? "s" : ""}
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
