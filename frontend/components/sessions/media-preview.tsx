"use client";

import { useState } from "react";
import { FileImage, Loader2, ZoomIn } from "lucide-react";
import { cn } from "@/lib/utils";

interface MediaPreviewProps {
  src?: string | null;
  alt: string;
  aspectRatio?: "square" | "portrait" | "landscape";
  className?: string;
}

export function MediaPreview({
  src,
  alt,
  aspectRatio = "portrait",
  className,
}: MediaPreviewProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  const aspectClasses = {
    square: "aspect-square",
    portrait: "aspect-[3/4]",
    landscape: "aspect-video",
  };

  if (!src || hasError) {
    return (
      <div
        className={cn(
          "bg-muted rounded-lg flex items-center justify-center",
          aspectClasses[aspectRatio],
          className
        )}
      >
        <div className="text-center text-muted-foreground">
          <FileImage className="h-8 w-8 mx-auto mb-2" />
          <p className="text-xs">No image</p>
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "relative bg-muted rounded-lg overflow-hidden group",
        aspectClasses[aspectRatio],
        className
      )}
    >
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      )}
      <img
        src={src}
        alt={alt}
        className={cn(
          "w-full h-full object-cover transition-opacity",
          isLoading ? "opacity-0" : "opacity-100"
        )}
        onLoad={() => setIsLoading(false)}
        onError={() => {
          setIsLoading(false);
          setHasError(true);
        }}
      />
      <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
        <ZoomIn className="h-8 w-8 text-white" />
      </div>
    </div>
  );
}






