import { cn } from "@/lib/utils";

interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  shimmer?: boolean;
}

function Skeleton({ className, shimmer = true, ...props }: SkeletonProps) {
  return (
    <div
      className={cn(
        "rounded-md bg-muted",
        shimmer && "skeleton-shimmer",
        className
      )}
      {...props}
    />
  );
}

// Pre-built skeleton variants for common use cases
function SkeletonText({ lines = 3, className }: { lines?: number; className?: string }) {
  return (
    <div className={cn("space-y-2", className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          className={cn(
            "h-4",
            i === lines - 1 ? "w-3/4" : "w-full"
          )}
        />
      ))}
    </div>
  );
}

function SkeletonCard({ className }: { className?: string }) {
  return (
    <div className={cn("rounded-lg border border-border/50 bg-card/30 p-4", className)}>
      <div className="flex items-center gap-4">
        <Skeleton className="h-12 w-12 rounded-full" />
        <div className="flex-1 space-y-2">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-3 w-1/2" />
        </div>
      </div>
    </div>
  );
}

function SkeletonTable({ rows = 5, cols = 4 }: { rows?: number; cols?: number }) {
  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex gap-4 pb-2 border-b border-border/50">
        {Array.from({ length: cols }).map((_, i) => (
          <Skeleton key={i} className="h-4 flex-1" />
        ))}
      </div>
      {/* Rows */}
      {Array.from({ length: rows }).map((_, rowIdx) => (
        <div key={rowIdx} className="flex gap-4 py-2">
          {Array.from({ length: cols }).map((_, colIdx) => (
            <Skeleton key={colIdx} className="h-5 flex-1" />
          ))}
        </div>
      ))}
    </div>
  );
}

function SkeletonSessionRow() {
  return (
    <tr className="border-b border-border/50">
      <td className="px-6 py-4">
        <Skeleton className="h-5 w-24" />
      </td>
      <td className="px-6 py-4">
        <Skeleton className="h-6 w-20 rounded-full" />
      </td>
      <td className="px-6 py-4">
        <Skeleton className="h-5 w-16" />
      </td>
      <td className="px-6 py-4">
        <Skeleton className="h-5 w-20" />
      </td>
      <td className="px-6 py-4">
        <Skeleton className="h-5 w-16" />
      </td>
      <td className="px-6 py-4">
        <Skeleton className="h-8 w-14 rounded" />
      </td>
    </tr>
  );
}

function SkeletonStatCard() {
  return (
    <div className="rounded-lg border border-border/50 bg-card/30 p-6 glass">
      <div className="flex items-center justify-between mb-2">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-4 w-4 rounded" />
      </div>
      <Skeleton className="h-8 w-20 mb-1" />
      <Skeleton className="h-3 w-16" />
    </div>
  );
}

export {
  Skeleton,
  SkeletonText,
  SkeletonCard,
  SkeletonTable,
  SkeletonSessionRow,
  SkeletonStatCard,
};
