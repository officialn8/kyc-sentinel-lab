"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import {
  ChevronLeft,
  ChevronRight,
  Filter,
  RefreshCw,
} from "lucide-react";
import { api, Session } from "@/lib/api";
import { formatRelativeTime, truncateId } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const statusColors: Record<string, "default" | "secondary" | "success" | "warning" | "danger"> = {
  pending: "secondary",
  processing: "warning",
  completed: "success",
  failed: "danger",
};

export default function SessionsPage() {
  const [page, setPage] = useState(1);
  const [filters, setFilters] = useState({
    status: "",
    source: "",
    attack_family: "",
  });

  const { data, isLoading, refetch } = useQuery({
    queryKey: ["sessions", page, filters],
    queryFn: () =>
      api.listSessions({
        page,
        page_size: 20,
        ...filters,
      }),
  });

  const sessions = data?.items ?? [];
  const totalPages = data?.pages ?? 1;

  return (
    <div className="space-y-6">
      {/* Filters */}
      <Card className="glass">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Filter className="h-5 w-5" />
              Filters
            </CardTitle>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => refetch()}
              className="gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4">
            <Select
              value={filters.status}
              onValueChange={(v) =>
                setFilters({ ...filters, status: v === "all" ? "" : v })
              }
            >
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="processing">Processing</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
              </SelectContent>
            </Select>

            <Select
              value={filters.source}
              onValueChange={(v) =>
                setFilters({ ...filters, source: v === "all" ? "" : v })
              }
            >
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="Source" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Sources</SelectItem>
                <SelectItem value="upload">Upload</SelectItem>
                <SelectItem value="synthetic">Synthetic</SelectItem>
              </SelectContent>
            </Select>

            <Select
              value={filters.attack_family}
              onValueChange={(v) =>
                setFilters({ ...filters, attack_family: v === "all" ? "" : v })
              }
            >
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Attack Family" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Families</SelectItem>
                <SelectItem value="benign">Benign</SelectItem>
                <SelectItem value="replay">Replay</SelectItem>
                <SelectItem value="injection">Injection</SelectItem>
                <SelectItem value="face_swap">Face Swap</SelectItem>
                <SelectItem value="doc_tamper">Doc Tamper</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Sessions Table */}
      <Card className="glass">
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border/50">
                  <th className="px-6 py-4 text-left text-sm font-medium text-muted-foreground">
                    ID
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-medium text-muted-foreground">
                    Status
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-medium text-muted-foreground">
                    Source
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-medium text-muted-foreground">
                    Attack Family
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-medium text-muted-foreground">
                    Created
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-medium text-muted-foreground">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {isLoading ? (
                  Array.from({ length: 5 }).map((_, i) => (
                    <tr key={i} className="border-b border-border/50">
                      <td colSpan={6} className="px-6 py-4">
                        <div className="h-6 animate-pulse rounded bg-muted" />
                      </td>
                    </tr>
                  ))
                ) : sessions.length === 0 ? (
                  <tr>
                    <td
                      colSpan={6}
                      className="px-6 py-12 text-center text-muted-foreground"
                    >
                      No sessions found
                    </td>
                  </tr>
                ) : (
                  sessions.map((session: Session) => (
                    <tr
                      key={session.id}
                      className="border-b border-border/50 hover:bg-muted/50 transition-colors"
                    >
                      <td className="px-6 py-4">
                        <code className="text-sm font-mono">
                          {truncateId(session.id)}
                        </code>
                      </td>
                      <td className="px-6 py-4">
                        <Badge variant={statusColors[session.status]}>
                          {session.status}
                        </Badge>
                      </td>
                      <td className="px-6 py-4 text-sm">{session.source}</td>
                      <td className="px-6 py-4 text-sm">
                        {session.attack_family || "—"}
                      </td>
                      <td className="px-6 py-4 text-sm text-muted-foreground">
                        {formatRelativeTime(session.created_at)}
                      </td>
                      <td className="px-6 py-4">
                        <Button variant="ghost" size="sm" asChild>
                          <Link href={`/sessions/${session.id}`}>View</Link>
                        </Button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between border-t border-border/50 px-6 py-4">
            <p className="text-sm text-muted-foreground">
              Page {page} of {totalPages} • {data?.total ?? 0} total sessions
            </p>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page >= totalPages}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}




