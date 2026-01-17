"use client";

import * as React from "react";
import { useTheme } from "next-themes";
import { Moon, Sun, Monitor } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

interface ThemeToggleProps {
  variant?: "icon" | "select" | "compact";
  className?: string;
}

export function ThemeToggle({ variant = "icon", className }: ThemeToggleProps) {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);

  // Avoid hydration mismatch
  React.useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    // Return placeholder with same dimensions to avoid layout shift
    if (variant === "icon") {
      return (
        <Button variant="ghost" size="icon" className={cn("h-9 w-9", className)}>
          <div className="h-5 w-5" />
        </Button>
      );
    }
    return <div className={cn("h-9 w-[120px]", className)} />;
  }

  // Icon toggle - cycles through dark -> light -> system
  if (variant === "icon") {
    const cycleTheme = () => {
      if (theme === "dark") {
        setTheme("light");
      } else if (theme === "light") {
        setTheme("system");
      } else {
        setTheme("dark");
      }
    };

    return (
      <Button
        variant="ghost"
        size="icon"
        onClick={cycleTheme}
        className={cn("h-9 w-9", className)}
        title={`Theme: ${theme} (click to change)`}
      >
        {resolvedTheme === "dark" ? (
          <Moon className="h-5 w-5" />
        ) : (
          <Sun className="h-5 w-5" />
        )}
        {theme === "system" && (
          <span className="absolute -bottom-0.5 -right-0.5 h-2 w-2 rounded-full bg-primary" />
        )}
        <span className="sr-only">Toggle theme</span>
      </Button>
    );
  }

  // Compact toggle - just dark/light icons in a pill
  if (variant === "compact") {
    return (
      <div
        className={cn(
          "flex items-center gap-1 rounded-full bg-muted p-1",
          className
        )}
      >
        <button
          onClick={() => setTheme("light")}
          className={cn(
            "rounded-full p-1.5 transition-colors",
            resolvedTheme === "light"
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          )}
          title="Light mode"
        >
          <Sun className="h-4 w-4" />
        </button>
        <button
          onClick={() => setTheme("dark")}
          className={cn(
            "rounded-full p-1.5 transition-colors",
            resolvedTheme === "dark"
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          )}
          title="Dark mode"
        >
          <Moon className="h-4 w-4" />
        </button>
        <button
          onClick={() => setTheme("system")}
          className={cn(
            "rounded-full p-1.5 transition-colors",
            theme === "system"
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          )}
          title="System preference"
        >
          <Monitor className="h-3.5 w-3.5" />
        </button>
      </div>
    );
  }

  // Select dropdown
  return (
    <Select value={theme} onValueChange={setTheme}>
      <SelectTrigger className={cn("w-[120px]", className)}>
        <SelectValue placeholder="Theme" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="light">
          <div className="flex items-center gap-2">
            <Sun className="h-4 w-4" />
            Light
          </div>
        </SelectItem>
        <SelectItem value="dark">
          <div className="flex items-center gap-2">
            <Moon className="h-4 w-4" />
            Dark
          </div>
        </SelectItem>
        <SelectItem value="system">
          <div className="flex items-center gap-2">
            <Monitor className="h-4 w-4" />
            System
          </div>
        </SelectItem>
      </SelectContent>
    </Select>
  );
}
