"use client";

import * as React from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import * as VisuallyHiddenPrimitive from "@radix-ui/react-visually-hidden";
import { X, ZoomIn, ZoomOut, RotateCw, Download } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface LightboxProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  src: string;
  alt: string;
  title?: string;
}

export function Lightbox({ open, onOpenChange, src, alt, title }: LightboxProps) {
  const [zoom, setZoom] = React.useState(1);
  const [rotation, setRotation] = React.useState(0);
  const [position, setPosition] = React.useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = React.useState(false);
  const [dragStart, setDragStart] = React.useState({ x: 0, y: 0 });

  // Reset state when closing
  React.useEffect(() => {
    if (!open) {
      setZoom(1);
      setRotation(0);
      setPosition({ x: 0, y: 0 });
    }
  }, [open]);

  const handleZoomIn = () => setZoom((z) => Math.min(z + 0.25, 3));
  const handleZoomOut = () => setZoom((z) => Math.max(z - 0.25, 0.5));
  const handleRotate = () => setRotation((r) => (r + 90) % 360);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom > 1) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging && zoom > 1) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleDownload = async () => {
    try {
      const response = await fetch(src);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = alt || "image";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to download image:", err);
    }
  };

  // Handle keyboard shortcuts
  React.useEffect(() => {
    if (!open) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case "+":
        case "=":
          handleZoomIn();
          break;
        case "-":
          handleZoomOut();
          break;
        case "r":
          handleRotate();
          break;
        case "0":
          setZoom(1);
          setPosition({ x: 0, y: 0 });
          break;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open]);

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/90 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0" />
        <DialogPrimitive.Content
          className="fixed inset-0 z-50 flex flex-col"
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          aria-describedby={undefined}
        >
          {/* Visually hidden title for accessibility */}
          <VisuallyHiddenPrimitive.Root asChild>
            <DialogPrimitive.Title>{title || alt}</DialogPrimitive.Title>
          </VisuallyHiddenPrimitive.Root>

          {/* Header */}
          <div className="flex items-center justify-between p-4 bg-black/50">
            <div className="text-white">
              {title && <h3 className="font-medium">{title}</h3>}
              <p className="text-sm text-white/70">{alt}</p>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={handleZoomOut}
                className="text-white hover:bg-white/10"
                disabled={zoom <= 0.5}
              >
                <ZoomOut className="h-5 w-5" />
              </Button>
              <span className="text-white text-sm min-w-[3rem] text-center">
                {Math.round(zoom * 100)}%
              </span>
              <Button
                variant="ghost"
                size="icon"
                onClick={handleZoomIn}
                className="text-white hover:bg-white/10"
                disabled={zoom >= 3}
              >
                <ZoomIn className="h-5 w-5" />
              </Button>
              <div className="w-px h-6 bg-white/20 mx-2" />
              <Button
                variant="ghost"
                size="icon"
                onClick={handleRotate}
                className="text-white hover:bg-white/10"
              >
                <RotateCw className="h-5 w-5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={handleDownload}
                className="text-white hover:bg-white/10"
              >
                <Download className="h-5 w-5" />
              </Button>
              <div className="w-px h-6 bg-white/20 mx-2" />
              <DialogPrimitive.Close asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-white hover:bg-white/10"
                >
                  <X className="h-5 w-5" />
                </Button>
              </DialogPrimitive.Close>
            </div>
          </div>

          {/* Image container */}
          <div
            className="flex-1 flex items-center justify-center overflow-hidden cursor-move"
            onMouseDown={handleMouseDown}
          >
            <img
              src={src}
              alt={alt}
              className={cn(
                "max-h-full max-w-full object-contain transition-transform select-none",
                isDragging ? "cursor-grabbing" : zoom > 1 ? "cursor-grab" : "cursor-default"
              )}
              style={{
                transform: `translate(${position.x}px, ${position.y}px) scale(${zoom}) rotate(${rotation}deg)`,
              }}
              draggable={false}
            />
          </div>

          {/* Footer with keyboard hints */}
          <div className="p-4 bg-black/50 text-center text-white/50 text-xs">
            <kbd className="px-1.5 py-0.5 rounded bg-white/10 mx-1">+</kbd>/<kbd className="px-1.5 py-0.5 rounded bg-white/10 mx-1">-</kbd> Zoom
            <span className="mx-3">|</span>
            <kbd className="px-1.5 py-0.5 rounded bg-white/10 mx-1">R</kbd> Rotate
            <span className="mx-3">|</span>
            <kbd className="px-1.5 py-0.5 rounded bg-white/10 mx-1">0</kbd> Reset
            <span className="mx-3">|</span>
            <kbd className="px-1.5 py-0.5 rounded bg-white/10 mx-1">Esc</kbd> Close
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

// Trigger component for opening lightbox on click
interface LightboxTriggerProps {
  src: string;
  alt: string;
  title?: string;
  children: React.ReactNode;
  className?: string;
}

export function LightboxTrigger({
  src,
  alt,
  title,
  children,
  className,
}: LightboxTriggerProps) {
  const [open, setOpen] = React.useState(false);

  return (
    <>
      <div
        className={cn("cursor-pointer", className)}
        onClick={() => setOpen(true)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && setOpen(true)}
      >
        {children}
      </div>
      <Lightbox
        open={open}
        onOpenChange={setOpen}
        src={src}
        alt={alt}
        title={title}
      />
    </>
  );
}
