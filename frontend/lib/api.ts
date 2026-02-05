/**
 * API client for KYC Sentinel backend
 */

// Always call same-origin `/api/*`.
// In production, `frontend/app/api/[...path]/route.ts` proxies to the backend and
// can inject server-only secrets (e.g., BACKEND_API_KEY) safely.
const API_BASE = "";

export interface Session {
  id: string;
  created_at: string;
  updated_at: string;
  status: "pending" | "processing" | "completed" | "failed";
  source: "upload" | "synthetic";
  attack_family?: string;
  attack_severity?: string;
  selfie_asset_key?: string;
  id_asset_key?: string;
}

export interface SessionDetail extends Session {
  device_os?: string;
  device_model?: string;
  ip_country?: string;
  capture_fps?: number;
  resolution?: string;
  selfie_url?: string;
  id_url?: string;
  selfie_crop_url?: string;
  id_crop_url?: string;
  result?: Result;
  reasons: Reason[];
}

export interface Result {
  id: string;
  session_id: string;
  created_at: string;
  risk_score: number;
  decision: "pass" | "review" | "fail";
  face_similarity?: number;
  pad_score?: number;
  doc_score?: number;
  model_version: string;
  rules_version: string;
}

export interface Reason {
  id: string;
  session_id: string;
  code: string;
  severity: "info" | "warn" | "high";
  message: string;
  evidence: Record<string, unknown>;
}

export interface SimilarFaceMatch {
  session_id: string;
  created_at: string;
  status: "pending" | "processing" | "completed" | "failed";
  source: "upload" | "synthetic";
  attack_family?: string;
  similarity_score: number;
  distance: number;
  decision?: "pass" | "review" | "fail";
  risk_score?: number;
}

export interface SimilarFacesListResponse {
  source_session_id: string;
  threshold: number;
  matches: SimilarFaceMatch[];
  total_matches: number;
}

export interface PresignedUpload {
  method?: "POST" | "PUT";
  url: string;
  fields: Record<string, string>;
  headers?: Record<string, string>;
  asset_key: string;
  expires_in: number;
}

export interface SessionCreateResponse {
  session: Session;
  selfie_upload: PresignedUpload;
  id_upload: PresignedUpload;
}

export interface SessionListResponse {
  items: Session[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
}

export interface MetricsSummary {
  total_sessions: number;
  completed_sessions: number;
  pass_count: number;
  review_count: number;
  fail_count: number;
  avg_risk_score: number;
  detection_rate: number;
}

export interface AttackFamilyMetrics {
  family: string;
  total: number;
  detected: number;
  missed: number;
  detection_rate: number;
  avg_risk_score: number;
}

export interface AttackFamilyBreakdown {
  families: AttackFamilyMetrics[];
}

export interface ConfusionCell {
  actual: string;
  predicted: string;
  count: number;
}

export interface ConfusionMatrixData {
  cells: ConfusionCell[];
  total: number;
}

export interface AttackFamily {
  id: string;
  name: string;
  description: string;
  severities: string[];
}

async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

export const api = {
  // Sessions
  createSession: (data: {
    source?: string;
    attack_family?: string;
    attack_severity?: string;
    device_os?: string;
    device_model?: string;
    ip_country?: string;
    selfie_filename?: string;
    selfie_content_type?: string;
    id_filename?: string;
    id_content_type?: string;
  }) =>
    fetchApi<SessionCreateResponse>("/api/sessions", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  finalizeSession: (id: string) =>
    fetchApi<{ status: string; message: string }>(
      `/api/sessions/${id}/finalize`,
      {
        method: "POST",
      }
    ),

  listSessions: (params?: {
    page?: number;
    page_size?: number;
    status?: string;
    source?: string;
    attack_family?: string;
    decision?: string;
    search?: string;
  }) => {
    const searchParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null && value !== "") {
          searchParams.set(key, String(value));
        }
      });
    }
    const query = searchParams.toString();
    return fetchApi<SessionListResponse>(
      `/api/sessions${query ? `?${query}` : ""}`
    );
  },

  getSession: (id: string) => fetchApi<SessionDetail>(`/api/sessions/${id}`),

  deleteSession: (id: string) =>
    fetchApi<{ status: string; id: string }>(`/api/sessions/${id}`, {
      method: "DELETE",
    }),

  findSimilarSessions: (id: string, limit = 10) =>
    fetchApi<SimilarFacesListResponse>(
      `/api/sessions/${id}/similar?limit=${limit}`
    ),

  // Simulation
  listAttackFamilies: () =>
    fetchApi<AttackFamily[]>("/api/simulate/families"),

  generateSyntheticSessions: (data: {
    attack_family: string;
    attack_severity: string;
    count: number;
  }) =>
    fetchApi<Session[]>("/api/simulate", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  // Metrics
  getMetricsSummary: () => fetchApi<MetricsSummary>("/api/metrics/summary"),

  getAttackFamilyBreakdown: () =>
    fetchApi<AttackFamilyBreakdown>("/api/metrics/by-attack-family"),

  getConfusionMatrix: () =>
    fetchApi<ConfusionMatrixData>("/api/metrics/confusion-matrix"),
};

// Upload helpers

// New POST-based upload with progress tracking
export async function uploadToPresignedPost(
  upload: PresignedUpload,
  file: File,
  onProgress?: (percent: number) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    const formData = new FormData();

    // Fields MUST come before file (S3 requirement)
    Object.entries(upload.fields).forEach(([key, value]) => {
      formData.append(key, value);
    });
    formData.append('file', file);

    const xhr = new XMLHttpRequest();

    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
      } else {
        reject(new Error(`Upload failed: ${xhr.status}`));
      }
    };

    xhr.onerror = () => reject(new Error('Network error during upload'));
    xhr.open('POST', upload.url);
    xhr.send(formData);
  });
}

// PUT-based upload with progress tracking (R2-compatible)
export async function uploadToPresignedPut(
  upload: PresignedUpload,
  file: File,
  onProgress?: (percent: number) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
      } else {
        reject(new Error(`Upload failed: ${xhr.status}`));
      }
    };

    xhr.onerror = () => reject(new Error('Network error during upload'));
    xhr.open('PUT', upload.url);

    const headers = upload.headers || {};
    Object.entries(headers).forEach(([key, value]) => {
      xhr.setRequestHeader(key, value);
    });

    xhr.send(file);
  });
}

// Legacy PUT-based upload (deprecated)
export async function uploadToPresignedUrl(
  url: string,
  file: File
): Promise<void> {
  console.warn('uploadToPresignedUrl is deprecated. Use uploadToPresignedPost instead.');
  const res = await fetch(url, {
    method: "PUT",
    body: file,
    headers: {
      "Content-Type": file.type,
    },
  });

  if (!res.ok) {
    throw new Error(`Upload failed: ${res.status}`);
  }
}

