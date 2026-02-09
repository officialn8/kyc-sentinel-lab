"use client";

import { useState, useEffect, useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertCircle,
  CheckCircle2,
  FileImage,
  Info,
  Loader2,
  Upload,
  X,
} from "lucide-react";
import {
  api,
  uploadToPresignedPost,
  uploadToPresignedPut,
  PresignedUpload,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { toast } from "@/components/ui/use-toast";
import { SuccessAnimation, ProgressRing } from "@/components/ui/animations";

// Validation constants
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const MIN_IMAGE_WIDTH = 480;
const MIN_IMAGE_HEIGHT = 640;
const ACCEPTED_FORMATS = ["image/jpeg", "image/jpg", "image/png", "image/webp"];

interface UploadFile {
  file: File;
  preview: string;
  validation: FileValidation;
}

interface FileValidation {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  dimensions?: { width: number; height: number };
  isBlurry?: boolean;
}

interface UploadProgress {
  selfie: number;
  idDoc: number;
}

// Parse user agent for device info
function parseUserAgent(): { os: string; model: string } {
  if (typeof navigator === "undefined") {
    return { os: "", model: "" };
  }

  const ua = navigator.userAgent;
  let os = "";
  let model = "";

  // Detect OS
  if (/iPhone|iPad|iPod/.test(ua)) {
    const match = ua.match(/OS (\d+[_\.]\d+)/);
    os = `iOS ${match ? match[1].replace("_", ".") : ""}`.trim();
    if (/iPhone/.test(ua)) model = "iPhone";
    else if (/iPad/.test(ua)) model = "iPad";
  } else if (/Android/.test(ua)) {
    const match = ua.match(/Android (\d+\.?\d*)/);
    os = `Android ${match ? match[1] : ""}`.trim();
    // Try to get device model
    const modelMatch = ua.match(/;\s*([^;)]+)\s*Build/);
    if (modelMatch) model = modelMatch[1].trim();
  } else if (/Mac OS X/.test(ua)) {
    const match = ua.match(/Mac OS X (\d+[_\.]\d+)/);
    os = `macOS ${match ? match[1].replace(/_/g, ".") : ""}`.trim();
    model = "Mac";
  } else if (/Windows/.test(ua)) {
    const match = ua.match(/Windows NT (\d+\.\d+)/);
    const versions: Record<string, string> = {
      "10.0": "10/11",
      "6.3": "8.1",
      "6.2": "8",
      "6.1": "7",
    };
    os = `Windows ${match ? versions[match[1]] || match[1] : ""}`.trim();
    model = "PC";
  } else if (/Linux/.test(ua)) {
    os = "Linux";
    model = "PC";
  }

  return { os, model };
}

// Simple blur detection using canvas
async function checkImageQuality(file: File): Promise<{ isBlurry: boolean; sharpness: number }> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        resolve({ isBlurry: false, sharpness: 100 });
        return;
      }

      // Scale down for faster processing
      const scale = Math.min(1, 200 / Math.max(img.width, img.height));
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Get image data
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;

      // Calculate Laplacian variance (simple sharpness metric)
      let sum = 0;
      let sumSq = 0;
      let count = 0;

      for (let y = 1; y < canvas.height - 1; y++) {
        for (let x = 1; x < canvas.width - 1; x++) {
          const idx = (y * canvas.width + x) * 4;
          // Convert to grayscale
          const gray =
            0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];

          // Laplacian kernel approximation
          const idxTop = ((y - 1) * canvas.width + x) * 4;
          const idxBot = ((y + 1) * canvas.width + x) * 4;
          const idxLeft = (y * canvas.width + (x - 1)) * 4;
          const idxRight = (y * canvas.width + (x + 1)) * 4;

          const grayTop =
            0.299 * data[idxTop] +
            0.587 * data[idxTop + 1] +
            0.114 * data[idxTop + 2];
          const grayBot =
            0.299 * data[idxBot] +
            0.587 * data[idxBot + 1] +
            0.114 * data[idxBot + 2];
          const grayLeft =
            0.299 * data[idxLeft] +
            0.587 * data[idxLeft + 1] +
            0.114 * data[idxLeft + 2];
          const grayRight =
            0.299 * data[idxRight] +
            0.587 * data[idxRight + 1] +
            0.114 * data[idxRight + 2];

          const laplacian =
            grayTop + grayBot + grayLeft + grayRight - 4 * gray;
          sum += laplacian;
          sumSq += laplacian * laplacian;
          count++;
        }
      }

      const variance = count > 0 ? sumSq / count - (sum / count) ** 2 : 0;
      // Normalize to 0-100 scale (empirically determined thresholds)
      const sharpness = Math.min(100, variance / 100);
      const isBlurry = sharpness < 15; // Threshold for blurry image

      resolve({ isBlurry, sharpness });
    };
    img.onerror = () => resolve({ isBlurry: false, sharpness: 100 });
    img.src = URL.createObjectURL(file);
  });
}

// Validate image file
async function validateFile(file: File): Promise<FileValidation> {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Check file type
  if (!ACCEPTED_FORMATS.includes(file.type)) {
    errors.push(`Invalid format: ${file.type || "unknown"}. Use JPG, PNG, or WebP.`);
  }

  // Check file size
  if (file.size > MAX_FILE_SIZE) {
    errors.push(
      `File too large: ${(file.size / 1024 / 1024).toFixed(1)}MB. Max ${MAX_FILE_SIZE / 1024 / 1024}MB.`
    );
  }

  // Check dimensions
  const dimensions = await new Promise<{ width: number; height: number }>(
    (resolve) => {
      const img = new Image();
      img.onload = () => resolve({ width: img.width, height: img.height });
      img.onerror = () => resolve({ width: 0, height: 0 });
      img.src = URL.createObjectURL(file);
    }
  );

  if (dimensions.width < MIN_IMAGE_WIDTH || dimensions.height < MIN_IMAGE_HEIGHT) {
    warnings.push(
      `Image resolution (${dimensions.width}×${dimensions.height}) is below recommended minimum (${MIN_IMAGE_WIDTH}×${MIN_IMAGE_HEIGHT}).`
    );
  }

  // Check image quality (blur detection)
  const { isBlurry, sharpness } = await checkImageQuality(file);
  if (isBlurry) {
    warnings.push(
      `Image may be blurry (sharpness: ${Math.round(sharpness)}%). Consider using a clearer image.`
    );
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    dimensions,
    isBlurry,
  };
}

export default function UploadPage() {
  const router = useRouter();
  const [selfie, setSelfie] = useState<UploadFile | null>(null);
  const [idDoc, setIdDoc] = useState<UploadFile | null>(null);
  const [isValidating, setIsValidating] = useState({ selfie: false, idDoc: false });
  const [uploadProgress, setUploadProgress] = useState<UploadProgress>({ selfie: 0, idDoc: 0 });
  const [showSuccess, setShowSuccess] = useState(false);
  const [createdSessionId, setCreatedSessionId] = useState<string | null>(null);
  const [deviceInfo, setDeviceInfo] = useState({
    device_os: "",
    device_model: "",
    ip_country: "",
  });

  // Auto-detect device info on mount
  useEffect(() => {
    const { os, model } = parseUserAgent();
    setDeviceInfo((prev) => ({
      ...prev,
      device_os: os,
      device_model: model,
    }));
  }, []);

  const handleFileSelect = useCallback(
    async (file: File, type: "selfie" | "idDoc") => {
      setIsValidating((prev) => ({ ...prev, [type]: true }));

      const validation = await validateFile(file);
      const preview = URL.createObjectURL(file);

      const uploadFile: UploadFile = { file, preview, validation };

      if (type === "selfie") {
        setSelfie(uploadFile);
      } else {
        setIdDoc(uploadFile);
      }

      setIsValidating((prev) => ({ ...prev, [type]: false }));
    },
    []
  );

  const selfieDropzone = useDropzone({
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    maxFiles: 1,
    maxSize: MAX_FILE_SIZE,
    onDrop: (files) => {
      if (files[0]) handleFileSelect(files[0], "selfie");
    },
    onDropRejected: (rejections) => {
      const error = rejections[0]?.errors[0];
      if (error) {
        toast({
          title: "File rejected",
          description: error.message,
          variant: "destructive",
        });
      }
    },
  });

  const idDropzone = useDropzone({
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    maxFiles: 1,
    maxSize: MAX_FILE_SIZE,
    onDrop: (files) => {
      if (files[0]) handleFileSelect(files[0], "idDoc");
    },
    onDropRejected: (rejections) => {
      const error = rejections[0]?.errors[0];
      if (error) {
        toast({
          title: "File rejected",
          description: error.message,
          variant: "destructive",
        });
      }
    },
  });

  // Upload with progress tracking (presigned POST)
  const uploadWithProgress = async (
    upload: PresignedUpload,
    file: File,
    onProgress: (progress: number) => void
  ): Promise<void> => {
    if (upload.method === "PUT") {
      await uploadToPresignedPut(upload, file, onProgress);
      return;
    }
    await uploadToPresignedPost(upload, file, onProgress);
  };

  const uploadMutation = useMutation({
    mutationFn: async () => {
      if (!selfie || !idDoc) {
        throw new Error("Please upload both selfie and ID document");
      }

      if (!selfie.validation.isValid || !idDoc.validation.isValid) {
        throw new Error("Please fix validation errors before uploading");
      }

      setUploadProgress({ selfie: 0, idDoc: 0 });

      // Create session
      const { session, selfie_upload, id_upload } = await api.createSession({
        source: "upload",
        ...deviceInfo,
        selfie_filename: selfie.file.name,
        selfie_content_type: selfie.file.type,
        selfie_size_bytes: selfie.file.size,
        id_filename: idDoc.file.name,
        id_content_type: idDoc.file.type,
        id_size_bytes: idDoc.file.size,
      });

      // Upload files with progress
      await Promise.all([
        uploadWithProgress(selfie_upload, selfie.file, (p) =>
          setUploadProgress((prev) => ({ ...prev, selfie: p }))
        ),
        uploadWithProgress(id_upload, idDoc.file, (p) =>
          setUploadProgress((prev) => ({ ...prev, idDoc: p }))
        ),
      ]);

      if (!selfie_upload.ticket || !id_upload.ticket) {
        throw new Error("Upload ticket missing from session creation response");
      }

      // Finalize session
      await api.finalizeSession(session.id, {
        selfie_ticket: selfie_upload.ticket,
        id_ticket: id_upload.ticket,
      });

      return session;
    },
    onSuccess: (session) => {
      setCreatedSessionId(session.id);
      setShowSuccess(true);
      toast({
        title: "Session created",
        description: "Processing started. You will be redirected shortly.",
      });
      // Delay redirect to show success animation
      setTimeout(() => {
        router.push(`/sessions/${session.id}`);
      }, 1500);
    },
    onError: (error) => {
      toast({
        title: "Upload failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const canSubmit =
    selfie &&
    idDoc &&
    selfie.validation.isValid &&
    idDoc.validation.isValid &&
    !uploadMutation.isPending &&
    !isValidating.selfie &&
    !isValidating.idDoc;

  const hasValidationErrors =
    (selfie && !selfie.validation.isValid) ||
    (idDoc && !idDoc.validation.isValid);

  const hasWarnings =
    (selfie && selfie.validation.warnings.length > 0) ||
    (idDoc && idDoc.validation.warnings.length > 0);

  const isUploading = uploadMutation.isPending;
  const overallProgress = isUploading
    ? Math.round((uploadProgress.selfie + uploadProgress.idDoc) / 2)
    : 0;

  return (
    <div className="max-w-4xl mx-auto space-y-6 relative">
      {/* Success overlay */}
      <AnimatePresence>
        {showSuccess && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm"
          >
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="flex flex-col items-center gap-4 p-8 rounded-xl bg-card border shadow-xl"
            >
              <SuccessAnimation show={showSuccess} size="lg" />
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="text-center"
              >
                <h3 className="text-xl font-semibold text-success">Upload Complete!</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Redirecting to session details...
                </p>
              </motion.div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <div>
        <h1 className="text-2xl font-bold">Upload KYC Session</h1>
        <p className="text-muted-foreground">
          Upload a selfie image and ID document for analysis
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Selfie Upload */}
        <Card className="glass">
          <CardHeader>
            <CardTitle className="text-base">Selfie Image</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div
              {...selfieDropzone.getRootProps()}
              className={cn(
                "relative border-2 border-dashed rounded-lg p-6 cursor-pointer transition-colors",
                selfieDropzone.isDragActive
                  ? "border-primary bg-primary/5"
                  : selfie && !selfie.validation.isValid
                    ? "border-danger bg-danger/5"
                    : selfie && selfie.validation.warnings.length > 0
                      ? "border-warning bg-warning/5"
                      : selfie
                        ? "border-success bg-success/5"
                        : "border-border hover:border-primary/50"
              )}
            >
              <input {...selfieDropzone.getInputProps()} />
              {isValidating.selfie ? (
                <div className="text-center py-8">
                  <Loader2 className="h-12 w-12 mx-auto text-primary animate-spin mb-4" />
                  <p className="text-sm text-muted-foreground">
                    Validating image...
                  </p>
                </div>
              ) : selfie ? (
                <div className="relative">
                  <img
                    src={selfie.preview}
                    alt="Selfie preview"
                    className="w-full aspect-[3/4] object-cover rounded"
                  />
                  <Button
                    variant="destructive"
                    size="icon"
                    className="absolute top-2 right-2"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelfie(null);
                    }}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                  <div
                    className={cn(
                      "absolute bottom-2 left-2 flex items-center gap-1 text-xs px-2 py-1 rounded",
                      selfie.validation.isValid
                        ? "bg-success/90 text-success-foreground"
                        : "bg-danger/90 text-danger-foreground"
                    )}
                  >
                    {selfie.validation.isValid ? (
                      <CheckCircle2 className="h-3 w-3" />
                    ) : (
                      <AlertCircle className="h-3 w-3" />
                    )}
                    {selfie.validation.isValid ? "Ready" : "Has errors"}
                  </div>
                  {selfie.validation.dimensions && (
                    <div className="absolute bottom-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                      {selfie.validation.dimensions.width}×
                      {selfie.validation.dimensions.height}
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8">
                  <FileImage className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <p className="text-sm text-muted-foreground">
                    Drag & drop or click to upload
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    JPG, PNG or WebP • Max 10MB
                  </p>
                </div>
              )}
            </div>

            {/* Validation feedback */}
            {selfie && (
              <div className="space-y-2">
                {selfie.validation.errors.map((error, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 text-sm text-danger"
                  >
                    <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
                    <span>{error}</span>
                  </div>
                ))}
                {selfie.validation.warnings.map((warning, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 text-sm text-warning"
                  >
                    <Info className="h-4 w-4 shrink-0 mt-0.5" />
                    <span>{warning}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Upload progress */}
            {isUploading && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Uploading...</span>
                  <span>{uploadProgress.selfie}%</span>
                </div>
                <Progress value={uploadProgress.selfie} className="h-2" />
              </div>
            )}
          </CardContent>
        </Card>

        {/* ID Document Upload */}
        <Card className="glass">
          <CardHeader>
            <CardTitle className="text-base">ID Document</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div
              {...idDropzone.getRootProps()}
              className={cn(
                "relative border-2 border-dashed rounded-lg p-6 cursor-pointer transition-colors",
                idDropzone.isDragActive
                  ? "border-primary bg-primary/5"
                  : idDoc && !idDoc.validation.isValid
                    ? "border-danger bg-danger/5"
                    : idDoc && idDoc.validation.warnings.length > 0
                      ? "border-warning bg-warning/5"
                      : idDoc
                        ? "border-success bg-success/5"
                        : "border-border hover:border-primary/50"
              )}
            >
              <input {...idDropzone.getInputProps()} />
              {isValidating.idDoc ? (
                <div className="text-center py-8">
                  <Loader2 className="h-12 w-12 mx-auto text-primary animate-spin mb-4" />
                  <p className="text-sm text-muted-foreground">
                    Validating image...
                  </p>
                </div>
              ) : idDoc ? (
                <div className="relative">
                  <img
                    src={idDoc.preview}
                    alt="ID preview"
                    className="w-full aspect-[3/4] object-cover rounded"
                  />
                  <Button
                    variant="destructive"
                    size="icon"
                    className="absolute top-2 right-2"
                    onClick={(e) => {
                      e.stopPropagation();
                      setIdDoc(null);
                    }}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                  <div
                    className={cn(
                      "absolute bottom-2 left-2 flex items-center gap-1 text-xs px-2 py-1 rounded",
                      idDoc.validation.isValid
                        ? "bg-success/90 text-success-foreground"
                        : "bg-danger/90 text-danger-foreground"
                    )}
                  >
                    {idDoc.validation.isValid ? (
                      <CheckCircle2 className="h-3 w-3" />
                    ) : (
                      <AlertCircle className="h-3 w-3" />
                    )}
                    {idDoc.validation.isValid ? "Ready" : "Has errors"}
                  </div>
                  {idDoc.validation.dimensions && (
                    <div className="absolute bottom-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                      {idDoc.validation.dimensions.width}×
                      {idDoc.validation.dimensions.height}
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8">
                  <FileImage className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <p className="text-sm text-muted-foreground">
                    Drag & drop or click to upload
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    JPG, PNG or WebP • Max 10MB
                  </p>
                </div>
              )}
            </div>

            {/* Validation feedback */}
            {idDoc && (
              <div className="space-y-2">
                {idDoc.validation.errors.map((error, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 text-sm text-danger"
                  >
                    <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
                    <span>{error}</span>
                  </div>
                ))}
                {idDoc.validation.warnings.map((warning, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 text-sm text-warning"
                  >
                    <Info className="h-4 w-4 shrink-0 mt-0.5" />
                    <span>{warning}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Upload progress */}
            {isUploading && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Uploading...</span>
                  <span>{uploadProgress.idDoc}%</span>
                </div>
                <Progress value={uploadProgress.idDoc} className="h-2" />
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Device Info (Optional) */}
      <Card className="glass">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            Device Metadata
            <span className="text-xs font-normal text-muted-foreground">
              (Auto-detected)
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="device_os">Device OS</Label>
              <Input
                id="device_os"
                placeholder="e.g., iOS 17.2"
                value={deviceInfo.device_os}
                onChange={(e) =>
                  setDeviceInfo({ ...deviceInfo, device_os: e.target.value })
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="device_model">Device Model</Label>
              <Input
                id="device_model"
                placeholder="e.g., iPhone 15 Pro"
                value={deviceInfo.device_model}
                onChange={(e) =>
                  setDeviceInfo({ ...deviceInfo, device_model: e.target.value })
                }
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="ip_country">Country Code</Label>
              <Input
                id="ip_country"
                placeholder="e.g., US"
                maxLength={2}
                value={deviceInfo.ip_country}
                onChange={(e) =>
                  setDeviceInfo({
                    ...deviceInfo,
                    ip_country: e.target.value.toUpperCase(),
                  })
                }
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Overall progress when uploading */}
      <AnimatePresence>
        {isUploading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card className="glass border-primary/50">
              <CardContent className="py-6">
                <div className="flex items-center gap-6">
                  <ProgressRing
                    progress={overallProgress}
                    size={64}
                    strokeWidth={5}
                    showPercentage={true}
                  />
                  <div className="flex-1 space-y-2">
                    <div className="font-medium">Uploading files...</div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Selfie</span>
                        <span className="font-mono">{uploadProgress.selfie}%</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">ID Document</span>
                        <span className="font-mono">{uploadProgress.idDoc}%</span>
                      </div>
                    </div>
                    <Progress value={overallProgress} className="h-2" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Submit */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-muted-foreground">
          {hasValidationErrors && (
            <span className="text-danger flex items-center gap-1">
              <AlertCircle className="h-4 w-4" />
              Fix validation errors to continue
            </span>
          )}
          {!hasValidationErrors && hasWarnings && (
            <span className="text-warning flex items-center gap-1">
              <Info className="h-4 w-4" />
              Review warnings above (you can still upload)
            </span>
          )}
        </div>
        <Button
          size="lg"
          disabled={!canSubmit}
          onClick={() => uploadMutation.mutate()}
        >
          {uploadMutation.isPending ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Uploading...
            </>
          ) : (
            <>
              <Upload className="mr-2 h-4 w-4" />
              Upload & Analyze
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
