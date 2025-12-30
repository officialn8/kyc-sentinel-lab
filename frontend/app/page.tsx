"use client";

import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  FileWarning,
  Shield,
  TrendingUp,
  Users,
} from "lucide-react";
import Link from "next/link";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function DashboardPage() {
  const { data: metrics, isLoading } = useQuery({
    queryKey: ["metrics", "summary"],
    queryFn: () => api.getMetricsSummary(),
  });

  const stats = [
    {
      title: "Total Sessions",
      value: metrics?.total_sessions ?? 0,
      icon: Users,
      description: "All time",
      color: "text-primary",
    },
    {
      title: "Detection Rate",
      value: metrics ? `${metrics.detection_rate}%` : "—",
      icon: Shield,
      description: "Attacks detected",
      color: "text-success",
    },
    {
      title: "Avg Risk Score",
      value: metrics?.avg_risk_score ?? 0,
      icon: TrendingUp,
      description: "Out of 100",
      color: "text-warning",
    },
    {
      title: "Failed Sessions",
      value: metrics?.fail_count ?? 0,
      icon: AlertTriangle,
      description: "High risk",
      color: "text-danger",
    },
  ];

  const decisionStats = [
    {
      label: "Pass",
      count: metrics?.pass_count ?? 0,
      color: "bg-success",
      icon: CheckCircle2,
    },
    {
      label: "Review",
      count: metrics?.review_count ?? 0,
      color: "bg-warning",
      icon: Clock,
    },
    {
      label: "Fail",
      count: metrics?.fail_count ?? 0,
      color: "bg-danger",
      icon: FileWarning,
    },
  ];

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-2xl border border-border/50 bg-card/50 p-8 glass">
        <div className="absolute inset-0 bg-gradient-to-r from-primary/10 via-transparent to-transparent" />
        <div className="relative">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-primary/20">
              <Shield className="h-6 w-6 text-primary" />
            </div>
            <h1 className="text-3xl font-bold tracking-tight">
              KYC Sentinel Lab
            </h1>
          </div>
          <p className="text-muted-foreground max-w-2xl mb-6">
            Red-team your remote identity verification flow with safe, synthetic
            attacks and explainable detection. Simulate modern KYC attack
            patterns and evaluate your detection capabilities.
          </p>
          <div className="flex gap-4">
            <Button asChild>
              <Link href="/upload">Upload Session</Link>
            </Button>
            <Button variant="outline" asChild>
              <Link href="/simulate">Generate Attacks</Link>
            </Button>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <Card key={stat.title} className="glass">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                {stat.title}
              </CardTitle>
              <stat.icon className={`h-4 w-4 ${stat.color}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {isLoading ? (
                  <div className="h-8 w-20 animate-pulse rounded bg-muted" />
                ) : (
                  stat.value
                )}
              </div>
              <p className="text-xs text-muted-foreground">{stat.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Decision Distribution */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card className="glass">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              Decision Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {decisionStats.map((stat) => {
                const total = metrics?.completed_sessions || 1;
                const percentage = Math.round((stat.count / total) * 100);
                return (
                  <div key={stat.label} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <stat.icon className="h-4 w-4" />
                        <span>{stat.label}</span>
                      </div>
                      <span className="font-medium">
                        {stat.count} ({percentage}%)
                      </span>
                    </div>
                    <div className="h-2 rounded-full bg-muted overflow-hidden">
                      <div
                        className={`h-full ${stat.color} transition-all duration-500`}
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        <Card className="glass">
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Button variant="outline" className="w-full justify-start" asChild>
              <Link href="/sessions">
                <Users className="mr-2 h-4 w-4" />
                View All Sessions
              </Link>
            </Button>
            <Button variant="outline" className="w-full justify-start" asChild>
              <Link href="/simulate">
                <Shield className="mr-2 h-4 w-4" />
                Run Attack Simulation
              </Link>
            </Button>
            <Button variant="outline" className="w-full justify-start" asChild>
              <Link href="/metrics">
                <TrendingUp className="mr-2 h-4 w-4" />
                View Detailed Metrics
              </Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}











