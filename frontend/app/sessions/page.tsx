"use client";

import { useState, useRef, useCallback, useEffect, Suspense } from "react";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  ChevronLeft,
  ChevronRight,
  Filter,
  Loader2,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import { api, Session } from "@/lib/api";
import { formatRelativeTime, truncateId } from "@/lib/utils";
import { useDebounce } from "@/lib/hooks";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SkeletonSessionRow } from "@/components/ui/skeleton";
import { EmptySessionsState, EmptySearchState } from "@/components/ui/empty-state";

const statusColors: Record<string, "default" | "secondary" | "success" | "warning" | "danger"> = {
  pending: "secondary",
  processing: "warning",
  completed: "success",
  failed: "danger",
};

// Row height for virtualization
const ROW_HEIGHT = 65;
// Threshold for enabling virtualization
const VIRTUALIZATION_THRESHOLD = 50;

interface SessionRowProps {
  session: Session;
  onHover?: () => void;
}

function SessionRow({ session, onHover }: SessionRowProps) {
  const router = useRouter();

  // Prefetch on hover
  const handleMouseEnter = useCallback(() => {
    router.prefetch(`/sessions/${session.id}`);
    onHover?.();
  }, [router, session.id, onHover]);

  return (
    <tr
      className="border-b border-border/50 hover:bg-muted/50 transition-colors"
      onMouseEnter={handleMouseEnter}
    >
      <td className="px-6 py-4">
        <code className="text-sm font-mono">{truncateId(session.id)}</code>
      </td>
      <td className="px-6 py-4">
        <Badge variant={statusColors[session.status]}>{session.status}</Badge>
      </td>
      <td className="px-6 py-4 text-sm">{session.source}</td>
      <td className="px-6 py-4 text-sm">{session.attack_family || "—"}</td>
      <td className="px-6 py-4 text-sm text-muted-foreground">
        {formatRelativeTime(session.created_at)}
      </td>
      <td className="px-6 py-4">
        <Button variant="ghost" size="sm" asChild>
          <Link href={`/sessions/${session.id}`} prefetch={true}>
            View
          </Link>
        </Button>
      </td>
    </tr>
  );
}

interface VirtualizedSessionTableProps {
  sessions: Session[];
}

function VirtualizedSessionTable({ sessions }: VirtualizedSessionTableProps) {
  const parentRef = useRef<HTMLDivElement>(null);

  const rowVirtualizer = useVirtualizer({
    count: sessions.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 10,
  });

  const virtualItems = rowVirtualizer.getVirtualItems();

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead className="sticky top-0 bg-card z-10">
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
      </table>
      <div
        ref={parentRef}
        className="overflow-auto"
        style={{ height: Math.min(sessions.length * ROW_HEIGHT, 600) }}
      >
        <table className="w-full">
          <tbody
            style={{
              height: `${rowVirtualizer.getTotalSize()}px`,
              width: "100%",
              position: "relative",
            }}
          >
            {virtualItems.map((virtualRow) => {
              const session = sessions[virtualRow.index];
              return (
                <tr
                  key={session.id}
                  className="border-b border-border/50 hover:bg-muted/50 transition-colors absolute w-full"
                  style={{
                    height: `${virtualRow.size}px`,
                    transform: `translateY(${virtualRow.start}px)`,
                  }}
                >
                  <td className="px-6 py-4 w-[15%]">
                    <code className="text-sm font-mono">
                      {truncateId(session.id)}
                    </code>
                  </td>
                  <td className="px-6 py-4 w-[12%]">
                    <Badge variant={statusColors[session.status]}>
                      {session.status}
                    </Badge>
                  </td>
                  <td className="px-6 py-4 text-sm w-[12%]">{session.source}</td>
                  <td className="px-6 py-4 text-sm w-[15%]">
                    {session.attack_family || "—"}
                  </td>
                  <td className="px-6 py-4 text-sm text-muted-foreground w-[20%]">
                    {formatRelativeTime(session.created_at)}
                  </td>
                  <td className="px-6 py-4 w-[10%]">
                    <Button variant="ghost" size="sm" asChild>
                      <Link href={`/sessions/${session.id}`} prefetch={true}>
                        View
                      </Link>
                    </Button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StandardSessionTable({ sessions }: VirtualizedSessionTableProps) {
  return (
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
          {sessions.map((session) => (
            <SessionRow key={session.id} session={session} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SessionsContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [page, setPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState(searchParams.get("q") || "");
  const debouncedSearch = useDebounce(searchQuery, 300);
  const [filters, setFilters] = useState({
    status: searchParams.get("status") || "",
    source: searchParams.get("source") || "",
    attack_family: searchParams.get("attack_family") || "",
  });

  // Reset page when filters or search changes
  useEffect(() => {
    setPage(1);
  }, [debouncedSearch, filters]);

  const { data, isLoading, isFetching, refetch } = useQuery({
    queryKey: ["sessions", page, filters, debouncedSearch],
    queryFn: () =>
      api.listSessions({
        page,
        page_size: 20,
        search: debouncedSearch || undefined,
        ...filters,
      }),
  });

  const isSearching = searchQuery !== debouncedSearch || isFetching;

  const sessions = data?.items ?? [];
  const totalPages = data?.pages ?? 1;
  const totalSessions = data?.total ?? 0;

  // Use virtualization for large lists
  const useVirtualization = sessions.length >= VIRTUALIZATION_THRESHOLD;

  return (
    <div className="space-y-6">
      {/* Search & Filters */}
      <Card className="glass">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Filter className="h-5 w-5" />
              Search & Filters
            </CardTitle>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => refetch()}
              className="gap-2"
              disabled={isFetching}
            >
              <RefreshCw className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Search input */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search by session ID, attack family..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 pr-9"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
              >
                {isSearching ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <X className="h-4 w-4" />
                )}
              </button>
            )}
          </div>

          {/* Filter dropdowns */}
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

            {/* Clear all filters */}
            {(searchQuery || filters.status || filters.source || filters.attack_family) && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setSearchQuery("");
                  setFilters({ status: "", source: "", attack_family: "" });
                }}
                className="text-muted-foreground"
              >
                Clear all
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Sessions Table */}
      <Card className="glass">
        <CardContent className="p-0">
          {isLoading ? (
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
                  {Array.from({ length: 5 }).map((_, i) => (
                    <SkeletonSessionRow key={i} />
                  ))}
                </tbody>
              </table>
            </div>
          ) : sessions.length === 0 ? (
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
                  <tr>
                    <td colSpan={6}>
                      {searchQuery || filters.status || filters.source || filters.attack_family ? (
                        <EmptySearchState />
                      ) : (
                        <EmptySessionsState />
                      )}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          ) : useVirtualization ? (
            <VirtualizedSessionTable sessions={sessions} />
          ) : (
            <StandardSessionTable sessions={sessions} />
          )}

          {/* Pagination */}
          <div className="flex items-center justify-between border-t border-border/50 px-6 py-4">
            <p className="text-sm text-muted-foreground">
              Page {page} of {totalPages} • {totalSessions} total sessions
              {useVirtualization && (
                <span className="ml-2 text-xs text-primary">
                  (virtualized)
                </span>
              )}
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

function SessionsLoadingState() {
  return (
    <div className="space-y-6">
      <Card className="glass">
        <CardContent className="p-6">
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default function SessionsPage() {
  return (
    <Suspense fallback={<SessionsLoadingState />}>
      <SessionsContent />
    </Suspense>
  );
}
