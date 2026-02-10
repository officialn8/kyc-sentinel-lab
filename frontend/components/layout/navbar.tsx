"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { OrganizationSwitcher, SignedIn } from "@clerk/nextjs";
import { Bell, Menu, Search, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { MobileNav } from "@/components/layout/mobile-nav";
import { CommandPalette, useCommandPalette } from "@/components/layout/command-palette";
import { ThemeToggle } from "@/components/layout/theme-toggle";

const pageTitles: Record<string, string> = {
  "/": "Dashboard",
  "/sessions": "Sessions",
  "/upload": "Upload Session",
  "/simulate": "Attack Simulator",
  "/metrics": "Metrics",
  "/organization/create": "Create Organization",
};

export function Navbar() {
  const pathname = usePathname();
  const [mobileNavOpen, setMobileNavOpen] = React.useState(false);
  const { open: commandOpen, setOpen: setCommandOpen } = useCommandPalette();
  
  // Get page title, handle dynamic routes
  let title = pageTitles[pathname];
  if (!title && pathname.startsWith("/sessions/")) {
    title = "Session Details";
  }
  title = title || "KYC Sentinel Lab";

  return (
    <>
      <header className="flex h-16 items-center justify-between border-b border-border/50 bg-card/30 px-4 md:px-6">
        {/* Left: Mobile menu + Title */}
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            onClick={() => setMobileNavOpen(true)}
          >
            <Menu className="h-5 w-5" />
            <span className="sr-only">Open menu</span>
          </Button>
          
          {/* Mobile logo */}
          <Link href="/" className="flex items-center gap-2 lg:hidden">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
              <Shield className="h-5 w-5 text-primary-foreground" />
            </div>
          </Link>

          <h1 className="text-lg font-semibold hidden sm:block">{title}</h1>
        </div>

        {/* Right: Search + Actions */}
        <div className="flex items-center gap-2 md:gap-4">
          {/* Command palette trigger - works on all breakpoints */}
          <Button
            variant="outline"
            className="relative h-9 w-9 p-0 md:h-9 md:w-60 md:justify-start md:px-3 md:py-2"
            onClick={() => setCommandOpen(true)}
          >
            <Search className="h-4 w-4 md:mr-2" />
            <span className="hidden md:inline-flex text-sm text-muted-foreground">
              Search...
            </span>
            <kbd className="pointer-events-none absolute right-1.5 top-1.5 hidden h-6 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium opacity-100 md:flex">
              <span className="text-xs">⌘</span>K
            </kbd>
          </Button>

          <SignedIn>
            <OrganizationSwitcher
              hidePersonal
              createOrganizationMode="navigation"
              createOrganizationUrl="/organization/create"
              afterCreateOrganizationUrl="/"
              afterSelectOrganizationUrl="/"
            />
          </SignedIn>

          <ThemeToggle variant="icon" />

          <Button variant="ghost" size="icon" className="relative">
            <Bell className="h-5 w-5" />
            <span className="absolute right-1 top-1 h-2 w-2 rounded-full bg-primary" />
            <span className="sr-only">Notifications</span>
          </Button>
        </div>
      </header>

      {/* Mobile Navigation Drawer */}
      <MobileNav open={mobileNavOpen} onOpenChange={setMobileNavOpen} />

      {/* Command Palette */}
      <CommandPalette open={commandOpen} onOpenChange={setCommandOpen} />
    </>
  );
}











