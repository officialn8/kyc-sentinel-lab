/**
 * Shared TypeScript types
 */

export type SessionStatus = "pending" | "processing" | "completed" | "failed";
export type SessionSource = "upload" | "synthetic";
export type Decision = "pass" | "review" | "fail";
export type Severity = "info" | "warn" | "high";
export type AttackFamilyType =
  | "replay"
  | "injection"
  | "face_swap"
  | "doc_tamper"
  | "benign";
export type AttackSeverityType = "low" | "medium" | "high";

export interface ReasonCode {
  code: string;
  category: "face" | "pad" | "doc" | "meta";
  defaultSeverity: Severity;
  description: string;
}

export const REASON_CODES: Record<string, ReasonCode> = {
  FACE_MISMATCH: {
    code: "FACE_MISMATCH",
    category: "face",
    defaultSeverity: "high",
    description: "Selfie face does not match ID photo",
  },
  FACE_NOT_DETECTED_SELFIE: {
    code: "FACE_NOT_DETECTED_SELFIE",
    category: "face",
    defaultSeverity: "high",
    description: "No face detected in selfie image",
  },
  FACE_NOT_DETECTED_ID: {
    code: "FACE_NOT_DETECTED_ID",
    category: "face",
    defaultSeverity: "high",
    description: "No face detected in ID document",
  },
  MULTIPLE_FACES_SELFIE: {
    code: "MULTIPLE_FACES_SELFIE",
    category: "face",
    defaultSeverity: "warn",
    description: "Multiple faces detected in selfie",
  },
  PAD_SUSPECT_REPLAY: {
    code: "PAD_SUSPECT_REPLAY",
    category: "pad",
    defaultSeverity: "high",
    description: "Screen capture artifacts suggesting replay attack",
  },
  PAD_SCREEN_ARTIFACTS: {
    code: "PAD_SCREEN_ARTIFACTS",
    category: "pad",
    defaultSeverity: "warn",
    description: "Moiré patterns or screen glare detected",
  },
  PAD_LOW_MOTION_ENTROPY: {
    code: "PAD_LOW_MOTION_ENTROPY",
    category: "pad",
    defaultSeverity: "warn",
    description: "Insufficient natural movement in video",
  },
  PAD_FRAME_STUTTER: {
    code: "PAD_FRAME_STUTTER",
    category: "pad",
    defaultSeverity: "warn",
    description: "Inconsistent frame timing detected",
  },
  PAD_INJECTION_ARTIFACTS: {
    code: "PAD_INJECTION_ARTIFACTS",
    category: "pad",
    defaultSeverity: "high",
    description: "Virtual camera injection artifacts",
  },
  PAD_FACE_BOUNDARY_MISMATCH: {
    code: "PAD_FACE_BOUNDARY_MISMATCH",
    category: "pad",
    defaultSeverity: "high",
    description: "Face boundary inconsistent with background",
  },
  DOC_TEMPLATE_MISMATCH: {
    code: "DOC_TEMPLATE_MISMATCH",
    category: "doc",
    defaultSeverity: "warn",
    description: "Document layout doesn't match expected template",
  },
  DOC_OCR_LOW_CONFIDENCE: {
    code: "DOC_OCR_LOW_CONFIDENCE",
    category: "doc",
    defaultSeverity: "warn",
    description: "Low confidence in document text extraction",
  },
  DOC_FONT_INCONSISTENT: {
    code: "DOC_FONT_INCONSISTENT",
    category: "doc",
    defaultSeverity: "warn",
    description: "Inconsistent fonts in document text",
  },
  DOC_TEXT_MISALIGNED: {
    code: "DOC_TEXT_MISALIGNED",
    category: "doc",
    defaultSeverity: "warn",
    description: "Text alignment issues in document",
  },
  DOC_EDGE_ARTIFACTS: {
    code: "DOC_EDGE_ARTIFACTS",
    category: "doc",
    defaultSeverity: "warn",
    description: "Editing artifacts around document edges",
  },
  DOC_METADATA_SUSPICIOUS: {
    code: "DOC_METADATA_SUSPICIOUS",
    category: "doc",
    defaultSeverity: "info",
    description: "Suspicious image metadata",
  },
  META_HIGH_RISK_DEVICE: {
    code: "META_HIGH_RISK_DEVICE",
    category: "meta",
    defaultSeverity: "warn",
    description: "Session from high-risk device profile",
  },
  META_SUSPICIOUS_TIMING: {
    code: "META_SUSPICIOUS_TIMING",
    category: "meta",
    defaultSeverity: "info",
    description: "Timing patterns suggest automation",
  },
  META_MULTIPLE_RETRIES: {
    code: "META_MULTIPLE_RETRIES",
    category: "meta",
    defaultSeverity: "info",
    description: "Multiple retry attempts detected",
  },
};

export const ATTACK_FAMILIES: Record<
  AttackFamilyType,
  { name: string; color: string }
> = {
  replay: { name: "Replay Attack", color: "bg-orange-500" },
  injection: { name: "Injection Attack", color: "bg-purple-500" },
  face_swap: { name: "Face Swap", color: "bg-red-500" },
  doc_tamper: { name: "Document Tampering", color: "bg-yellow-500" },
  benign: { name: "Benign", color: "bg-green-500" },
};

export const DECISION_CONFIG: Record<
  Decision,
  { label: string; color: string; bgColor: string }
> = {
  pass: { label: "Pass", color: "text-success", bgColor: "bg-success" },
  review: { label: "Review", color: "text-warning", bgColor: "bg-warning" },
  fail: { label: "Fail", color: "text-danger", bgColor: "bg-danger" },
};

export const SEVERITY_CONFIG: Record<
  Severity,
  { label: string; color: string; bgColor: string }
> = {
  info: { label: "Info", color: "text-blue-500", bgColor: "bg-blue-500" },
  warn: { label: "Warning", color: "text-warning", bgColor: "bg-warning" },
  high: { label: "High", color: "text-danger", bgColor: "bg-danger" },
};



