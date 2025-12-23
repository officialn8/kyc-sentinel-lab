"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import {
  CheckCircle2,
  FileImage,
  Loader2,
  Upload,
  X,
} from "lucide-react";
import { api, uploadToPresignedUrl } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "@/components/ui/use-toast";

interface UploadFile {
  file: File;
  preview: string;
}

export default function UploadPage() {
  const router = useRouter();
  const [selfie, setSelfie] = useState<UploadFile | null>(null);
  const [idDoc, setIdDoc] = useState<UploadFile | null>(null);
  const [deviceInfo, setDeviceInfo] = useState({
    device_os: "",
    device_model: "",
    ip_country: "",
  });

  const uploadMutation = useMutation({
    mutationFn: async () => {
      if (!selfie || !idDoc) {
        throw new Error("Please upload both selfie and ID document");
      }

      // Create session
      const { session, upload_urls } = await api.createSession({
        source: "upload",
        ...deviceInfo,
      });

      // Upload files
      await Promise.all([
        uploadToPresignedUrl(upload_urls.selfie_upload_url, selfie.file),
        uploadToPresignedUrl(upload_urls.id_upload_url, idDoc.file),
      ]);

      // Finalize session
      await api.finalizeSession(session.id);

      return session;
    },
    onSuccess: (session) => {
      toast({
        title: "Session created",
        description: "Processing started. You will be redirected shortly.",
      });
      router.push(`/sessions/${session.id}`);
    },
    onError: (error) => {
      toast({
        title: "Upload failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const selfieDropzone = useDropzone({
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    maxFiles: 1,
    onDrop: (files) => {
      if (files[0]) {
        setSelfie({
          file: files[0],
          preview: URL.createObjectURL(files[0]),
        });
      }
    },
  });

  const idDropzone = useDropzone({
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    maxFiles: 1,
    onDrop: (files) => {
      if (files[0]) {
        setIdDoc({
          file: files[0],
          preview: URL.createObjectURL(files[0]),
        });
      }
    },
  });

  const canSubmit = selfie && idDoc && !uploadMutation.isPending;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
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
          <CardContent>
            <div
              {...selfieDropzone.getRootProps()}
              className={cn(
                "relative border-2 border-dashed rounded-lg p-6 cursor-pointer transition-colors",
                selfieDropzone.isDragActive
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50"
              )}
            >
              <input {...selfieDropzone.getInputProps()} />
              {selfie ? (
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
                  <div className="absolute bottom-2 left-2 flex items-center gap-1 bg-success/90 text-success-foreground text-xs px-2 py-1 rounded">
                    <CheckCircle2 className="h-3 w-3" />
                    Ready
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <FileImage className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <p className="text-sm text-muted-foreground">
                    Drag & drop or click to upload
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    JPG, PNG or WebP
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* ID Document Upload */}
        <Card className="glass">
          <CardHeader>
            <CardTitle className="text-base">ID Document</CardTitle>
          </CardHeader>
          <CardContent>
            <div
              {...idDropzone.getRootProps()}
              className={cn(
                "relative border-2 border-dashed rounded-lg p-6 cursor-pointer transition-colors",
                idDropzone.isDragActive
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50"
              )}
            >
              <input {...idDropzone.getInputProps()} />
              {idDoc ? (
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
                  <div className="absolute bottom-2 left-2 flex items-center gap-1 bg-success/90 text-success-foreground text-xs px-2 py-1 rounded">
                    <CheckCircle2 className="h-3 w-3" />
                    Ready
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <FileImage className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <p className="text-sm text-muted-foreground">
                    Drag & drop or click to upload
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    JPG, PNG or WebP
                  </p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Device Info (Optional) */}
      <Card className="glass">
        <CardHeader>
          <CardTitle className="text-base">Device Metadata (Optional)</CardTitle>
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

      {/* Submit */}
      <div className="flex justify-end">
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




