"use client";

import { useQuery } from "@tanstack/react-query";
import {
  BarChart3,
  CheckCircle2,
  Clock,
  FileWarning,
  Target,
  TrendingUp,
} from "lucide-react";
import {
  Bar,
  BarChart,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

const COLORS = {
  pass: "#22c55e",
  review: "#f59e0b",
  fail: "#ef4444",
  benign: "#22c55e",
  replay: "#f97316",
  injection: "#a855f7",
  face_swap: "#ef4444",
  doc_tamper: "#eab308",
};

export default function MetricsPage() {
  const { data: summary, isLoading: loadingSummary } = useQuery({
    queryKey: ["metrics", "summary"],
    queryFn: () => api.getMetricsSummary(),
  });

  const { data: breakdown, isLoading: loadingBreakdown } = useQuery({
    queryKey: ["metrics", "breakdown"],
    queryFn: () => api.getAttackFamilyBreakdown(),
  });

  const { data: confusion, isLoading: loadingConfusion } = useQuery({
    queryKey: ["metrics", "confusion"],
    queryFn: () => api.getConfusionMatrix(),
  });

  // Prepare pie chart data
  const decisionData = summary
    ? [
        { name: "Pass", value: summary.pass_count, color: COLORS.pass },
        { name: "Review", value: summary.review_count, color: COLORS.review },
        { name: "Fail", value: summary.fail_count, color: COLORS.fail },
      ]
    : [];

  // Prepare bar chart data
  const familyData =
    breakdown?.families.map((f) => ({
      name: f.family.replace("_", " "),
      detected: f.detected,
      missed: f.missed,
      rate: f.detection_rate,
    })) ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <BarChart3 className="h-6 w-6 text-primary" />
          Detection Metrics
        </h1>
        <p className="text-muted-foreground">
          Aggregate performance metrics and detection rates
        </p>
      </div>

      {/* Summary Stats */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="glass">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total Sessions
            </CardTitle>
            <Target className="h-4 w-4 text-primary" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {loadingSummary ? (
                <div className="h-8 w-16 animate-pulse rounded bg-muted" />
              ) : (
                summary?.total_sessions ?? 0
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              {summary?.completed_sessions ?? 0} completed
            </p>
          </CardContent>
        </Card>

        <Card className="glass">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Detection Rate
            </CardTitle>
            <TrendingUp className="h-4 w-4 text-success" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-success">
              {loadingSummary ? (
                <div className="h-8 w-16 animate-pulse rounded bg-muted" />
              ) : (
                `${summary?.detection_rate ?? 0}%`
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Attacks correctly flagged
            </p>
          </CardContent>
        </Card>

        <Card className="glass">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Avg Risk Score
            </CardTitle>
            <BarChart3 className="h-4 w-4 text-warning" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {loadingSummary ? (
                <div className="h-8 w-16 animate-pulse rounded bg-muted" />
              ) : (
                summary?.avg_risk_score ?? 0
              )}
            </div>
            <p className="text-xs text-muted-foreground">Out of 100</p>
          </CardContent>
        </Card>

        <Card className="glass">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Failed Sessions
            </CardTitle>
            <FileWarning className="h-4 w-4 text-danger" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-danger">
              {loadingSummary ? (
                <div className="h-8 w-16 animate-pulse rounded bg-muted" />
              ) : (
                summary?.fail_count ?? 0
              )}
            </div>
            <p className="text-xs text-muted-foreground">High risk flagged</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Decision Distribution Pie */}
        <Card className="glass">
          <CardHeader>
            <CardTitle>Decision Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={decisionData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {decisionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Attack Family Detection Rates */}
        <Card className="glass">
          <CardHeader>
            <CardTitle>Detection by Attack Family</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={familyData} layout="vertical">
                  <XAxis type="number" domain={[0, 100]} />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={100}
                    tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                    formatter={(value: number) => [`${value}%`, "Detection Rate"]}
                  />
                  <Bar dataKey="rate" radius={[0, 4, 4, 0]}>
                    {familyData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={
                          COLORS[
                            entry.name.replace(" ", "_") as keyof typeof COLORS
                          ] || "#8884d8"
                        }
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Attack Family Details */}
      <Card className="glass">
        <CardHeader>
          <CardTitle>Attack Family Breakdown</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {loadingBreakdown
              ? Array.from({ length: 5 }).map((_, i) => (
                  <div key={i} className="h-16 animate-pulse rounded bg-muted" />
                ))
              : breakdown?.families.map((family) => (
                  <div key={family.family} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{
                            backgroundColor:
                              COLORS[family.family as keyof typeof COLORS] ||
                              "#8884d8",
                          }}
                        />
                        <span className="font-medium capitalize">
                          {family.family.replace("_", " ")}
                        </span>
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {family.detected}/{family.total} detected •{" "}
                        {family.detection_rate}%
                      </div>
                    </div>
                    <Progress value={family.detection_rate} />
                    <div className="flex gap-6 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <CheckCircle2 className="h-3 w-3 text-success" />
                        {family.detected} detected
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3 text-warning" />
                        {family.missed} missed
                      </span>
                      <span>Avg score: {family.avg_risk_score}</span>
                    </div>
                  </div>
                ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}






