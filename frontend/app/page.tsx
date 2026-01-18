"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  FileWarning,
  Shield,
  TrendingUp,
  Users,
  Upload,
  Zap,
} from "lucide-react";
import Link from "next/link";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EnhancedButton } from "@/components/ui/enhanced-button";
import { SkeletonStatCard } from "@/components/ui/skeleton";
import { GettingStarted } from "@/components/layout/getting-started";
import { animations } from "@/lib/animations";

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

  // Determine onboarding progress
  const hasNoSessions = !isLoading && metrics?.total_sessions === 0;
  const completedSteps: number[] = [];
  if (metrics?.total_sessions && metrics.total_sessions > 0) {
    completedSteps.push(1, 2, 3); // Has sessions, likely completed first 3 steps
  }
  if (metrics?.completed_sessions && metrics.completed_sessions > 0) {
    completedSteps.push(4); // Has viewed results
  }

  return (
    <motion.div
      initial="initial"
      animate="animate"
      variants={animations.pageTransitionVariants}
      className="space-y-8"
    >
      {/* Hero Section */}
      <motion.div
        variants={animations.variants.scaleIn}
        className="relative overflow-hidden rounded-xl border border-border/50 bg-card/30 p-8 md:p-12 glass-strong gradient-mesh-animated shadow-elevation-3"
      >
        {/* Animated gradient danger overlay */}
        <div className="absolute inset-0 gradient-danger opacity-50 pointer-events-none" />

        {/* Floating security particles */}
        <motion.div
          className="absolute top-10 right-10 opacity-20 pointer-events-none"
          animate={{
            y: [-10, 10, -10],
            transition: {
              duration: 6,
              ease: "easeInOut",
              repeat: Infinity,
            },
          }}
        >
          <Shield className="h-24 w-24 text-primary" />
        </motion.div>
        <motion.div
          className="absolute bottom-10 left-10 opacity-10 pointer-events-none"
          animate={{
            y: [-10, 10, -10],
            transition: {
              duration: 6,
              ease: "easeInOut",
              repeat: Infinity,
              delay: 2,
            },
          }}
        >
          <AlertTriangle className="h-32 w-32 text-accent" />
        </motion.div>

        <div className="relative z-10">
          <motion.div
            className="flex items-center gap-4 mb-6"
            variants={animations.variants.slideRight}
          >
            <motion.div
              className="p-3 rounded-lg bg-primary/20 backdrop-blur-xl shadow-glow-primary"
              whileHover={{ scale: 1.1, rotate: 360 }}
              transition={animations.springs.bouncy}
            >
              <Shield className="h-8 w-8 text-primary" />
            </motion.div>
            <h1 className="text-3xl md:text-5xl font-bold tracking-tighter">
              KYC Sentinel Lab
            </h1>
          </motion.div>

          <motion.p
            variants={animations.variants.slideUp}
            className="text-muted-foreground max-w-3xl mb-8 text-base md:text-lg leading-relaxed"
          >
            <span className="text-primary font-semibold">Red-team</span> your remote identity verification flow with{" "}
            <span className="text-danger font-semibold">synthetic attacks</span> and{" "}
            <span className="text-accent font-semibold">explainable detection</span>.
            Simulate modern KYC attack patterns and evaluate your fraud detection capabilities.
          </motion.p>

          <motion.div
            variants={animations.variants.slideUp}
            className="flex flex-col sm:flex-row gap-4"
          >
            <EnhancedButton size="lg" glow asChild>
              <Link href="/upload">
                <Upload className="mr-2 h-5 w-5" />
                Upload Session
              </Link>
            </EnhancedButton>
            <EnhancedButton variant="outline" size="lg" asChild>
              <Link href="/simulate">
                <Zap className="mr-2 h-5 w-5 text-accent" />
                Generate Attacks
              </Link>
            </EnhancedButton>
          </motion.div>
        </div>
      </motion.div>

      {/* Getting Started Card - Show when no sessions */}
      {hasNoSessions && (
        <GettingStarted completedSteps={completedSteps} />
      )}

      {/* Stats Grid */}
      <motion.div
        variants={animations.variants.container}
        className="grid gap-4 grid-cols-2 lg:grid-cols-4"
      >
        {isLoading ? (
          // Skeleton loading state
          Array.from({ length: 4 }).map((_, i) => (
            <motion.div
              key={i}
              variants={animations.listItemVariants}
              custom={i}
            >
              <SkeletonStatCard />
            </motion.div>
          ))
        ) : (
          stats.map((stat, index) => (
            <motion.div
              key={stat.title}
              variants={animations.listItemVariants}
              custom={index}
              whileHover="hover"
              whileTap="tap"
            >
              <Card className="glass-soft shadow-elevation-1 hover:shadow-elevation-3 transition-all duration-300 group">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-xs md:text-sm font-medium text-muted-foreground">
                    {stat.title}
                  </CardTitle>
                  <motion.div
                    whileHover={{ scale: 1.2, rotate: 15 }}
                    transition={animations.springs.bouncy}
                  >
                    <stat.icon className={`h-4 w-4 ${stat.color} transition-all group-hover:drop-shadow-glow`} />
                  </motion.div>
                </CardHeader>
                <CardContent>
                  <motion.div
                    className="text-xl md:text-2xl font-bold"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 + 0.3 }}
                  >
                    {stat.value}
                  </motion.div>
                  <p className="text-xs text-muted-foreground">{stat.description}</p>
                </CardContent>
              </Card>
            </motion.div>
          ))
        )}
      </motion.div>

      {/* Decision Distribution & Quick Actions */}
      <div className="grid gap-4 md:grid-cols-2">
        <motion.div
          variants={animations.variants.slideRight}
          whileHover={{ scale: 1.01 }}
          transition={animations.springs.smooth}
        >
          <Card className="glass-soft shadow-elevation-2 h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                >
                  <Activity className="h-5 w-5 text-primary" />
                </motion.div>
                Decision Distribution
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {decisionStats.map((stat, index) => {
                  const total = metrics?.completed_sessions || 1;
                  const percentage = Math.round((stat.count / total) * 100);
                  return (
                    <motion.div
                      key={stat.label}
                      className="space-y-2"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                    >
                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-2">
                          <motion.div
                            whileHover={{ scale: 1.2 }}
                            transition={animations.springs.bouncy}
                          >
                            <stat.icon className="h-4 w-4" />
                          </motion.div>
                          <span>{stat.label}</span>
                        </div>
                        <span className="font-medium">
                          <motion.span
                            key={stat.count}
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={animations.springs.snappy}
                          >
                            {stat.count}
                          </motion.span>{" "}
                          ({percentage}%)
                        </span>
                      </div>
                      <div className="h-3 rounded-full bg-muted/50 overflow-hidden shadow-inner">
                        <motion.div
                          className={`h-full ${stat.color} shadow-elevation-1`}
                          initial={{ width: 0 }}
                          animate={{ width: `${percentage}%` }}
                          transition={{
                            duration: 1,
                            delay: index * 0.1,
                            ease: [0.4, 0, 0.2, 1],
                          }}
                        />
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          variants={animations.variants.slideLeft}
          whileHover={{ scale: 1.01 }}
          transition={animations.springs.smooth}
        >
          <Card className="glass-soft shadow-elevation-2 h-full">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5 text-accent animate-pulse" />
                Quick Actions
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <EnhancedButton
                variant="outline"
                className="w-full justify-start group"
                asChild
              >
                <Link href="/sessions">
                  <Users className="mr-2 h-4 w-4 transition-transform group-hover:scale-110" />
                  View All Sessions
                </Link>
              </EnhancedButton>
              <EnhancedButton
                variant="outline"
                className="w-full justify-start group"
                asChild
              >
                <Link href="/simulate">
                  <Shield className="mr-2 h-4 w-4 transition-transform group-hover:rotate-12" />
                  Run Attack Simulation
                </Link>
              </EnhancedButton>
              <EnhancedButton
                variant="outline"
                className="w-full justify-start group"
                asChild
              >
                <Link href="/metrics">
                  <TrendingUp className="mr-2 h-4 w-4 transition-transform group-hover:translate-x-1 group-hover:-translate-y-1" />
                  View Detailed Metrics
                </Link>
              </EnhancedButton>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </motion.div>
  );
}












