/**
 * API client for KYC Sentinel backend
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

export interface PresignedUrlResponse {
  selfie_upload_url: string;
  selfie_asset_key: string;
  id_upload_url: string;
  id_asset_key: string;
  expires_in: number;
}

export interface SessionCreateResponse {
  session: Session;
  upload_urls: PresignedUrlResponse;
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
    fetchApi<Session[]>(`/api/sessions/${id}/similar?limit=${limit}`),

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

// Upload helper
export async function uploadToPresignedUrl(
  url: string,
  file: File
): Promise<void> {
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



