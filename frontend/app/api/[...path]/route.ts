import { NextRequest } from "next/server";

export const runtime = "nodejs";

function getBackendBaseUrl() {
  return (
    process.env.BACKEND_API_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://localhost:8000"
  );
}

async function proxy(req: NextRequest, pathParts: string[]) {
  const backendBase = getBackendBaseUrl().replace(/\/+$/, "");

  const incomingUrl = new URL(req.url);
  const targetUrl = `${backendBase}/api/${pathParts.join("/")}${incomingUrl.search}`;

  const headers = new Headers(req.headers);
  headers.delete("host");

  const apiKey = process.env.BACKEND_API_KEY;
  if (apiKey) {
    headers.set("X-API-Key", apiKey);
  }

  const method = req.method.toUpperCase();

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

export async function GET(
  req: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxy(req, params.path);
}

export async function POST(
  req: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxy(req, params.path);
}

export async function PUT(
  req: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxy(req, params.path);
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxy(req, params.path);
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: { path: string[] } }
) {
  return proxy(req, params.path);
}


