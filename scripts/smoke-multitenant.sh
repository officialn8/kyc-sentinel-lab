#!/bin/bash
# Multi-tenant smoke test for backend deployment validation.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
ORG_ALPHA="${ORG_ALPHA:-org_smokealpha}"
ORG_BETA="${ORG_BETA:-org_smokebeta}"
API_KEY="${API_KEY:-}"
BASIC_AUTH="${BASIC_AUTH:-}" # format: user:password

PASS_COUNT=0
FAIL_COUNT=0
HTTP_STATUS=""
HTTP_BODY=""

print_usage() {
  cat <<'EOF'
Usage:
  BASE_URL=http://localhost:8000 ./scripts/smoke-multitenant.sh

Optional env vars:
  ORG_ALPHA   Tenant header value for tenant A (default: org_smokealpha)
  ORG_BETA    Tenant header value for tenant B (default: org_smokebeta)
  API_KEY     Backend API key, if auth is enabled
  BASIC_AUTH  Basic auth creds in user:password format, if auth is enabled
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

validate_org_id() {
  local value="$1"
  local label="$2"
  if [[ ! "$value" =~ ^org_[A-Za-z0-9]+$ ]]; then
    echo "[error] ${label} must match ^org_[A-Za-z0-9]+$: got '${value}'"
    exit 2
  fi
}

validate_org_id "$ORG_ALPHA" "ORG_ALPHA"
validate_org_id "$ORG_BETA" "ORG_BETA"

info() {
  echo "[info] $*"
}

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  echo "[pass] $*"
}

fail() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  echo "[fail] $*"
  echo "  status=${HTTP_STATUS}"
  if [[ -n "${HTTP_BODY}" ]]; then
    echo "  body=${HTTP_BODY}"
  fi
}

http() {
  local method="$1"
  local path="$2"
  local data="${3:-}"
  shift 3
  local -a headers
  headers=("$@")

  local tmp_body
  tmp_body="$(mktemp)"
  local curl_args=(
    -sS
    -o "$tmp_body"
    -w "%{http_code}"
    -X "$method"
    "${BASE_URL}${path}"
    -H "Accept: application/json"
  )

  if [[ -n "$API_KEY" ]]; then
    curl_args+=(-H "X-API-Key: $API_KEY")
  fi
  if [[ -n "$BASIC_AUTH" ]]; then
    curl_args+=(-u "$BASIC_AUTH")
  fi

  if [[ ${#headers[@]} -gt 0 ]]; then
    local header
    for header in "${headers[@]}"; do
      curl_args+=(-H "$header")
    done
  fi

  if [[ -n "$data" ]]; then
    curl_args+=(-H "Content-Type: application/json" --data "$data")
  fi

  if ! HTTP_STATUS="$(curl "${curl_args[@]}")"; then
    HTTP_STATUS="000"
    HTTP_BODY="curl request failed"
    rm -f "$tmp_body"
    return 0
  fi

  HTTP_BODY="$(cat "$tmp_body")"
  rm -f "$tmp_body"
}

expect_status() {
  local expected="$1"
  local label="$2"
  if [[ "$HTTP_STATUS" == "$expected" ]]; then
    pass "$label"
  else
    fail "$label (expected ${expected})"
  fi
}

json_get() {
  local path="$1"
  local json_input="$2"
  JSON_PATH="$path" python3 -c '
import json
import os
import sys

try:
    path = os.environ["JSON_PATH"].split(".")
    raw = sys.stdin.read()
    obj = json.loads(raw)
    for part in path:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = obj[part]
except Exception:
    sys.exit(1)

if obj is None:
    print("")
elif isinstance(obj, (dict, list)):
    print(json.dumps(obj))
else:
    print(obj)
' <<<"$json_input"
}

assert_nonempty() {
  local value="$1"
  local label="$2"
  if [[ -n "$value" ]]; then
    pass "$label"
  else
    fail "$label (empty value)"
  fi
}

info "Starting multi-tenant smoke test against ${BASE_URL}"

http GET "/health" ""
expect_status "200" "health endpoint responds"

http GET "/api/metrics/summary" ""
expect_status "403" "metrics blocked without tenant header"

http POST "/api/sessions" '{"source":"upload"}' "X-Authenticated-Org-Id: ${ORG_ALPHA}"
expect_status "200" "create session for tenant alpha"
ALPHA_SESSION_ID="$(json_get "session.id" "$HTTP_BODY" || true)"
assert_nonempty "$ALPHA_SESSION_ID" "alpha session id extracted"

http POST "/api/sessions" '{"source":"upload"}' "X-Authenticated-Org-Id: ${ORG_BETA}"
expect_status "200" "create session for tenant beta"
BETA_SESSION_ID="$(json_get "session.id" "$HTTP_BODY" || true)"
assert_nonempty "$BETA_SESSION_ID" "beta session id extracted"

if [[ -z "$ALPHA_SESSION_ID" || -z "$BETA_SESSION_ID" ]]; then
  info "Session creation failed; skipping dependent tenant-isolation checks."
  echo
  echo "Smoke test summary: ${PASS_COUNT} passed, ${FAIL_COUNT} failed"
  exit 1
fi

http GET "/api/sessions/${ALPHA_SESSION_ID}" "" "X-Authenticated-Org-Id: ${ORG_ALPHA}"
expect_status "200" "alpha can read own session"

http GET "/api/sessions/${ALPHA_SESSION_ID}" "" "X-Authenticated-Org-Id: ${ORG_BETA}"
expect_status "404" "beta cannot read alpha session"

http GET "/api/sessions/${ALPHA_SESSION_ID}/similar" "" "X-Authenticated-Org-Id: ${ORG_BETA}"
expect_status "404" "beta cannot query alpha similar endpoint"

http POST "/api/upload/presigned?key=sessions/${ALPHA_SESSION_ID}/selfie.jpg" ""
expect_status "403" "upload presign blocked without tenant header"

http POST "/api/upload/presigned?key=sessions/${ALPHA_SESSION_ID}/selfie.jpg" "" "X-Authenticated-Org-Id: ${ORG_ALPHA}"
expect_status "200" "alpha can request own upload presign"

http POST "/api/upload/presigned?key=sessions/${ALPHA_SESSION_ID}/selfie.jpg" "" "X-Authenticated-Org-Id: ${ORG_BETA}"
expect_status "404" "beta cannot request alpha upload presign"

http GET "/api/metrics/summary" "" "X-Authenticated-Org-Id: ${ORG_ALPHA}"
expect_status "200" "alpha metrics endpoint works"

http GET "/api/metrics/summary" "" "X-Authenticated-Org-Id: ${ORG_BETA}"
expect_status "200" "beta metrics endpoint works"

echo
echo "Smoke test summary: ${PASS_COUNT} passed, ${FAIL_COUNT} failed"

if [[ "$FAIL_COUNT" -gt 0 ]]; then
  exit 1
fi
