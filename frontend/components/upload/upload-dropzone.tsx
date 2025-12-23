"use client";

import { useCallback } from "react";
import { useDropzone, Accept } from "react-dropzone";
import { Upload, FileImage, X, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface UploadDropzoneProps {
  accept?: Accept;
  maxSize?: number;
  onFileSelect: (file: File) => void;
  selectedFile?: File | null;
  onClear?: () => void;
  previewUrl?: string | null;
  label?: string;
  description?: string;
  className?: string;
}

export function UploadDropzone({
  accept = { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
  maxSize = 10 * 1024 * 1024, // 10MB
  onFileSelect,
  selectedFile,
  onClear,
  previewUrl,
  label = "Upload file",
  description = "Drag & drop or click to select",
  className,
}: UploadDropzoneProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles[0]) {
        onFileSelect(acceptedFiles[0]);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } =
    useDropzone({
      onDrop,
      accept,
      maxFiles: 1,
      maxSize,
    });

  const hasFile = selectedFile || previewUrl;

  return (
    <div className={cn("space-y-2", className)}>
      {label && <p className="text-sm font-medium">{label}</p>}
      <div
        {...getRootProps()}
        className={cn(
          "relative border-2 border-dashed rounded-lg cursor-pointer transition-all",
          isDragActive && "border-primary bg-primary/5",
          isDragReject && "border-danger bg-danger/5",
          !isDragActive &&
            !isDragReject &&
            "border-border hover:border-primary/50",
          hasFile && "border-success/50 bg-success/5"
        )}
      >
        <input {...getInputProps()} />

        {previewUrl ? (
          <div className="relative p-2">
            <img
              src={previewUrl}
              alt="Preview"
              className="w-full aspect-[3/4] object-cover rounded"
            />
            {onClear && (
              <Button
                variant="destructive"
                size="icon"
                className="absolute top-4 right-4"
                onClick={(e) => {
                  e.stopPropagation();
                  onClear();
                }}
              >
                <X className="h-4 w-4" />
              </Button>
            )}
            <div className="absolute bottom-4 left-4 flex items-center gap-1 bg-success/90 text-success-foreground text-xs px-2 py-1 rounded">
              <CheckCircle2 className="h-3 w-3" />
              Ready to upload
            </div>
          </div>
        ) : (
          <div className="p-8 text-center">
            <div
              className={cn(
                "mx-auto w-12 h-12 rounded-full flex items-center justify-center mb-4",
                isDragActive ? "bg-primary/20" : "bg-muted"
              )}
            >
              {isDragActive ? (
                <Upload className="h-6 w-6 text-primary" />
              ) : (
                <FileImage className="h-6 w-6 text-muted-foreground" />
              )}
            </div>
            <p className="text-sm text-muted-foreground">
              {isDragReject
                ? "File type not accepted"
                : isDragActive
                ? "Drop the file here"
                : description}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Max size: {Math.round(maxSize / 1024 / 1024)}MB
            </p>
          </div>
        )}
      </div>
    </div>
  );
}




