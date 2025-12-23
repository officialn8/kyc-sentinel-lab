"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Bell, Menu, Search, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const pageTitles: Record<string, string> = {
  "/": "Dashboard",
  "/sessions": "Sessions",
  "/upload": "Upload Session",
  "/simulate": "Attack Simulator",
  "/metrics": "Metrics",
};

export function Navbar() {
  const pathname = usePathname();
  
  // Get page title, handle dynamic routes
  let title = pageTitles[pathname];
  if (!title && pathname.startsWith("/sessions/")) {
    title = "Session Details";
  }
  title = title || "KYC Sentinel Lab";

  return (
    <header className="flex h-16 items-center justify-between border-b border-border/50 bg-card/30 px-6">
      {/* Left: Mobile menu + Title */}
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" className="lg:hidden">
          <Menu className="h-5 w-5" />
        </Button>
        
        {/* Mobile logo */}
        <Link href="/" className="flex items-center gap-2 lg:hidden">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
            <Shield className="h-5 w-5 text-primary-foreground" />
          </div>
        </Link>

        <h1 className="text-lg font-semibold">{title}</h1>
      </div>

      {/* Right: Search + Actions */}
      <div className="flex items-center gap-4">
        <div className="relative hidden md:block">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="search"
            placeholder="Search sessions..."
            className="w-64 pl-9"
          />
        </div>

        <Button variant="ghost" size="icon" className="relative">
          <Bell className="h-5 w-5" />
          <span className="absolute right-1 top-1 h-2 w-2 rounded-full bg-primary" />
        </Button>
      </div>
    </header>
  );
}




