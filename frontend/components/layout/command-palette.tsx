"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart3,
  FileUp,
  Home,
  List,
  Search,
  Shield,
  Zap,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  Clock,
} from "lucide-react";
import { api, Session } from "@/lib/api";
import { truncateId } from "@/lib/utils";
import { useDebounce } from "@/lib/hooks";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
} from "@/components/ui/command";

const navigation = [
  {
    name: "Dashboard",
    href: "/",
    icon: Home,
    shortcut: "D",
  },
  {
    name: "Sessions",
    href: "/sessions",
    icon: List,
    shortcut: "S",
  },
  {
    name: "Upload Session",
    href: "/upload",
    icon: FileUp,
    shortcut: "U",
  },
  {
    name: "Attack Simulator",
    href: "/simulate",
    icon: Zap,
    shortcut: "A",
  },
  {
    name: "Metrics",
    href: "/metrics",
    icon: BarChart3,
    shortcut: "M",
  },
];

const statusIcons = {
  pending: Clock,
  processing: Loader2,
  completed: CheckCircle2,
  failed: AlertTriangle,
} as const;

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  const router = useRouter();
  const [search, setSearch] = React.useState("");
  const debouncedSearch = useDebounce(search, 300);

  // Search sessions when query is entered
  const { data: sessionResults, isLoading: isSearching } = useQuery({
    queryKey: ["command-palette-search", debouncedSearch],
    queryFn: () =>
      api.listSessions({
        search: debouncedSearch,
        page_size: 5,
      }),
    enabled: debouncedSearch.length >= 2,
    staleTime: 30000,
  });

  // Reset search when dialog closes
  React.useEffect(() => {
    if (!open) {
      setSearch("");
    }
  }, [open]);

  const runCommand = React.useCallback(
    (command: () => void) => {
      onOpenChange(false);
      command();
    },
    [onOpenChange]
  );

  const sessions = sessionResults?.items ?? [];
  const showSessionResults = debouncedSearch.length >= 2;

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput
        placeholder="Search sessions, navigate, or run a command..."
        value={search}
        onValueChange={setSearch}
      />
      <CommandList>
        <CommandEmpty>
          {isSearching ? (
            <div className="flex items-center justify-center gap-2 py-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Searching...</span>
            </div>
          ) : (
            "No results found."
          )}
        </CommandEmpty>

        {/* Session search results */}
        {showSessionResults && sessions.length > 0 && (
          <>
            <CommandGroup heading="Sessions">
              {sessions.map((session) => {
                const StatusIcon = statusIcons[session.status];
                return (
                  <CommandItem
                    key={session.id}
                    value={`session-${session.id}`}
                    onSelect={() =>
                      runCommand(() => router.push(`/sessions/${session.id}`))
                    }
                  >
                    <StatusIcon className="mr-2 h-4 w-4" />
                    <span className="font-mono text-sm">
                      {truncateId(session.id)}
                    </span>
                    <span className="ml-2 text-muted-foreground text-sm">
                      {session.source}
                      {session.attack_family && ` • ${session.attack_family}`}
                    </span>
                  </CommandItem>
                );
              })}
              {sessionResults && sessionResults.total > 5 && (
                <CommandItem
                  value="view-all-sessions"
                  onSelect={() =>
                    runCommand(() =>
                      router.push(`/sessions?q=${encodeURIComponent(debouncedSearch)}`)
                    )
                  }
                >
                  <Search className="mr-2 h-4 w-4" />
                  <span>
                    View all {sessionResults.total} results for "{debouncedSearch}"
                  </span>
                </CommandItem>
              )}
            </CommandGroup>
            <CommandSeparator />
          </>
        )}

        {/* Navigation */}
        <CommandGroup heading="Navigation">
          {navigation.map((item) => (
            <CommandItem
              key={item.href}
              value={item.name}
              onSelect={() => runCommand(() => router.push(item.href))}
            >
              <item.icon className="mr-2 h-4 w-4" />
              <span>{item.name}</span>
              <CommandShortcut>⌘{item.shortcut}</CommandShortcut>
            </CommandItem>
          ))}
        </CommandGroup>
        <CommandSeparator />
        <CommandGroup heading="Quick Actions">
          <CommandItem
            value="Generate Synthetic Attack"
            onSelect={() => runCommand(() => router.push("/simulate"))}
          >
            <Shield className="mr-2 h-4 w-4" />
            <span>Generate Synthetic Attack</span>
          </CommandItem>
          <CommandItem
            value="Upload KYC Session"
            onSelect={() => runCommand(() => router.push("/upload"))}
          >
            <FileUp className="mr-2 h-4 w-4" />
            <span>Upload KYC Session</span>
          </CommandItem>
          <CommandItem
            value="Search all sessions"
            onSelect={() =>
              runCommand(() =>
                router.push(
                  search
                    ? `/sessions?q=${encodeURIComponent(search)}`
                    : "/sessions"
                )
              )
            }
          >
            <Search className="mr-2 h-4 w-4" />
            <span>Search all sessions</span>
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}

// Hook to manage command palette state globally
export function useCommandPalette() {
  const [open, setOpen] = React.useState(false);

  React.useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((open) => !open);
      }
    };

    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);

  return { open, setOpen };
}
