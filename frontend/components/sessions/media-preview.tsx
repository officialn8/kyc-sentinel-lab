"use client";

import { ZoomIn } from "lucide-react";
import { cn } from "@/lib/utils";
import { AspectImage } from "@/components/ui/optimized-image";

interface MediaPreviewProps {
  src?: string | null;
  alt: string;
  aspectRatio?: "square" | "portrait" | "landscape";
  className?: string;
  showZoomHint?: boolean;
  priority?: boolean;
}

export function MediaPreview({
  src,
  alt,
  aspectRatio = "portrait",
  className,
  showZoomHint = true,
  priority = false,
}: MediaPreviewProps) {
  const aspectMap = {
    square: "square" as const,
    portrait: "portrait" as const,
    landscape: "landscape" as const,
  };

  return (
    <div className={cn("relative group", className)}>
      <AspectImage
        src={src}
        alt={alt}
        aspectRatio={aspectMap[aspectRatio]}
        priority={priority}
        placeholderText="No image"
      />
      {showZoomHint && src && (
        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none rounded-lg">
          <ZoomIn className="h-8 w-8 text-white" />
        </div>
      )}
    </div>
  );
}












