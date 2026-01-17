"use client";

import * as React from "react";
import Image from "next/image";
import { FileImage, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface OptimizedImageProps {
  src?: string | null;
  alt: string;
  width?: number;
  height?: number;
  fill?: boolean;
  priority?: boolean;
  className?: string;
  containerClassName?: string;
  sizes?: string;
  quality?: number;
  onLoad?: () => void;
  onError?: () => void;
  showPlaceholder?: boolean;
  placeholderIcon?: React.ReactNode;
  placeholderText?: string;
}

// Simple blur placeholder data URL (gray)
const BLUR_DATA_URL =
  "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZTVlN2ViIi8+PC9zdmc+";

export function OptimizedImage({
  src,
  alt,
  width,
  height,
  fill = false,
  priority = false,
  className,
  containerClassName,
  sizes = "(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw",
  quality = 85,
  onLoad,
  onError,
  showPlaceholder = true,
  placeholderIcon,
  placeholderText = "No image",
}: OptimizedImageProps) {
  const [isLoading, setIsLoading] = React.useState(true);
  const [hasError, setHasError] = React.useState(false);

  // Reset state when src changes
  React.useEffect(() => {
    if (src) {
      setIsLoading(true);
      setHasError(false);
    }
  }, [src]);

  // Show placeholder for missing or errored images
  if (!src || hasError) {
    if (!showPlaceholder) return null;

    return (
      <div
        className={cn(
          "flex items-center justify-center bg-muted rounded-lg",
          containerClassName
        )}
      >
        <div className="text-center text-muted-foreground p-4">
          {placeholderIcon || <FileImage className="h-8 w-8 mx-auto mb-2" />}
          <p className="text-xs">{placeholderText}</p>
        </div>
      </div>
    );
  }

  // Check URL type
  const isDataUrl = src.startsWith("data:");
  const isBlob = src.startsWith("blob:");

  // For blob/data URLs or any URL, use regular img tag for reliability
  // next/image can be finicky with external URLs and CORS
  if (isBlob || isDataUrl) {
    return (
      <div className={cn("relative overflow-hidden", containerClassName)}>
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-muted">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        )}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={alt}
          className={cn(
            "transition-opacity duration-300",
            isLoading ? "opacity-0" : "opacity-100",
            className
          )}
          onLoad={() => {
            setIsLoading(false);
            onLoad?.();
          }}
          onError={() => {
            setIsLoading(false);
            setHasError(true);
            onError?.();
          }}
        />
      </div>
    );
  }

  // For external URLs, use native img for maximum compatibility
  // This avoids issues with next/image remotePatterns and CORS
  return (
    <div className={cn("relative overflow-hidden", containerClassName)}>
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-muted z-10">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      )}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={alt}
        loading={priority ? "eager" : "lazy"}
        decoding="async"
        className={cn(
          "transition-opacity duration-300",
          isLoading ? "opacity-0" : "opacity-100",
          fill && "absolute inset-0 w-full h-full",
          className
        )}
        style={!fill && width && height ? { width, height } : undefined}
        onLoad={() => {
          setIsLoading(false);
          onLoad?.();
        }}
        onError={() => {
          setIsLoading(false);
          setHasError(true);
          onError?.();
        }}
      />
    </div>
  );
}

// Card-sized image with aspect ratio
interface AspectImageProps
  extends Omit<OptimizedImageProps, "fill" | "width" | "height"> {
  aspectRatio?: "square" | "portrait" | "landscape" | "video";
}

export function AspectImage({
  aspectRatio = "portrait",
  className,
  containerClassName,
  ...props
}: AspectImageProps) {
  const aspectClasses = {
    square: "aspect-square",
    portrait: "aspect-[3/4]",
    landscape: "aspect-[4/3]",
    video: "aspect-video",
  };

  return (
    <OptimizedImage
      {...props}
      fill
      containerClassName={cn(
        "relative rounded-lg",
        aspectClasses[aspectRatio],
        containerClassName
      )}
      className={cn("object-cover", className)}
    />
  );
}
