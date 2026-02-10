import { auth } from "@clerk/nextjs/server";
import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const METRICS_ENDPOINTS = new Set([
  "summary",
  "by-attack-family",
  "confusion-matrix",
]);

function isAllowedProxyRoute(method: string, pathParts: string[]): boolean {
  const [root, second, third] = pathParts;

  if (root === "sessions") {
    if (pathParts.length === 1) {
      return method === "GET" || method === "POST";
    }

    if (!second || !UUID_RE.test(second)) {
      return false;
    }

    if (pathParts.length === 2) {
      return method === "GET" || method === "DELETE";
    }

    if (pathParts.length === 3 && third === "finalize") {
      return method === "POST";
    }

    if (pathParts.length === 3 && third === "similar") {
      return method === "GET";
    }

    return false;
  }

  if (root === "simulate") {
    if (pathParts.length === 1) {
      return method === "POST";
    }
    return pathParts.length === 2 && second === "families" && method === "GET";
  }

  if (root === "metrics") {
    return pathParts.length === 2 && !!second && METRICS_ENDPOINTS.has(second) && method === "GET";
  }

  return false;
}

function getBackendBaseUrl() {
  return (
    process.env.BACKEND_API_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://localhost:8000"
  );
}

async function proxy(req: NextRequest, pathParts: string[]) {
  const { userId, orgId } = await auth();
  if (!userId) {
    return NextResponse.json({ detail: "Authentication required" }, { status: 401 });
  }
  if (!orgId) {
    return NextResponse.json({ detail: "Organization context required" }, { status: 403 });
  }

  const method = req.method.toUpperCase();
  if (!isAllowedProxyRoute(method, pathParts)) {
    return NextResponse.json({ detail: "Forbidden proxy route" }, { status: 403 });
  }

  const backendBase = getBackendBaseUrl().replace(/\/+$/, "");

  const incomingUrl = new URL(req.url);
  const targetUrl = `${backendBase}/api/${pathParts.join("/")}${incomingUrl.search}`;

  const headers = new Headers(req.headers);
  headers.delete("host");
  headers.delete("authorization");
  headers.delete("x-api-key");
  headers.delete("cookie");
  headers.delete("x-forwarded-for");
  headers.delete("x-real-ip");
  headers.delete("x-authenticated-org-id");
  headers.delete("x-tenant-id");

  const apiKey = process.env.BACKEND_API_KEY;
  if (apiKey) {
    headers.set("X-API-Key", apiKey);
  }
  headers.set("X-Authenticated-User-Id", userId);
  headers.set("X-Authenticated-Org-Id", orgId);

  // Forward body for non-GET/HEAD requests
  const body =
    method === "GET" || method === "HEAD" ? undefined : await req.arrayBuffer();

  const res = await fetch(targetUrl, {
    method,
    headers,
    body,
    // Avoid caching API responses at the edge by default
    cache: "no-store",
  });

  // Pass through most headers (content-type, etc.). Avoid hop-by-hop headers.
  const outHeaders = new Headers(res.headers);
  outHeaders.delete("connection");
  outHeaders.delete("keep-alive");
  outHeaders.delete("proxy-authenticate");
  outHeaders.delete("proxy-authorization");
  outHeaders.delete("te");
  outHeaders.delete("trailers");
  outHeaders.delete("transfer-encoding");
  outHeaders.delete("upgrade");

  return new Response(res.body, {
    status: res.status,
    headers: outHeaders,
  });
}

// Next.js 16+: params is now a Promise that must be awaited
type RouteContext = { params: Promise<{ path: string[] }> };

export async function GET(req: NextRequest, { params }: RouteContext) {
  const { path } = await params;
  return proxy(req, path);
}

export async function POST(req: NextRequest, { params }: RouteContext) {
  const { path } = await params;
  return proxy(req, path);
}

export async function PUT(req: NextRequest, { params }: RouteContext) {
  const { path } = await params;
  return proxy(req, path);
}

export async function PATCH(req: NextRequest, { params }: RouteContext) {
  const { path } = await params;
  return proxy(req, path);
}

export async function DELETE(req: NextRequest, { params }: RouteContext) {
  const { path } = await params;
  return proxy(req, path);
}
