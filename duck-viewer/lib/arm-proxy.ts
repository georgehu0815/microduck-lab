import {
  hasOnlyFiniteJson,
  isSafeArtifactPath,
  parseArmRuns,
  parseArmState,
  validateResetBody,
  validateStepBody,
  validateTeacherBody,
} from "@/lib/arm-contract";

const ARM_BACKEND = "http://127.0.0.1:8812";
const JSON_LIMIT = 2 * 1024 * 1024;
const REQUEST_LIMIT = 32 * 1024;
const TIMEOUT_MS = 4_000;

export const ARM_GET_PATHS = new Set(["health", "state", "runs", "artifact"]);
export const ARM_POST_PATHS = new Set(["reset", "step", "teacher"]);

function jsonError(
  error: string,
  status: number,
  available = status < 500
) {
  return Response.json(
    { available, error },
    { status, headers: { "Cache-Control": "no-store" } }
  );
}

async function readJsonBody(request: Request): Promise<unknown> {
  const declared = Number(request.headers.get("content-length") ?? "0");
  if (Number.isFinite(declared) && declared > REQUEST_LIMIT) {
    throw new Error("Request body is too large.");
  }
  const text = await request.text();
  if (Buffer.byteLength(text) > REQUEST_LIMIT) {
    throw new Error("Request body is too large.");
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new Error("Request body must be valid JSON.");
  }
}

function validateMutation(path: string, value: unknown) {
  if (path === "reset") return validateResetBody(value);
  if (path === "step") return validateStepBody(value);
  if (path === "teacher") return validateTeacherBody(value);
  return null;
}

async function backendFetch(path: string, init?: RequestInit) {
  return fetch(`${ARM_BACKEND}/${path}`, {
    ...init,
    cache: "no-store",
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
}

async function proxyJson(path: string, init?: RequestInit) {
  const response = await backendFetch(path, init);
  if (!response.ok) {
    return jsonError(
      `Arm backend rejected /${path} (${response.status}).`,
      502,
      false
    );
  }
  const declared = Number(response.headers.get("content-length") ?? "0");
  if (Number.isFinite(declared) && declared > JSON_LIMIT) {
    return jsonError("Arm backend response is too large.", 502);
  }
  const text = await response.text();
  if (Buffer.byteLength(text) > JSON_LIMIT) {
    return jsonError("Arm backend response is too large.", 502);
  }
  let payload: unknown;
  try {
    payload = JSON.parse(text);
  } catch {
    return jsonError("Arm backend returned invalid JSON.", 502);
  }
  if (!hasOnlyFiniteJson(payload)) {
    return jsonError("Arm backend returned non-finite JSON.", 502);
  }
  if (response.ok && path === "state" && !parseArmState(payload)) {
    return jsonError("Arm backend returned an invalid state contract.", 502);
  }
  if (response.ok && path === "runs" && !parseArmRuns(payload)) {
    return jsonError("Arm backend returned an invalid runs contract.", 502);
  }
  if (
    response.ok &&
    (path === "reset" || path === "step" || path === "teacher") &&
    !parseArmState(payload)
  ) {
    return jsonError("Arm backend returned an invalid state contract.", 502);
  }
  return new Response(JSON.stringify(payload), {
    status: response.status,
    headers: {
      "Cache-Control": "no-store",
      "Content-Type": "application/json; charset=utf-8",
    },
  });
}

export async function handleArmGet(request: Request, path: string) {
  if (!ARM_GET_PATHS.has(path)) {
    return jsonError("Unknown Arm API path.", 404);
  }
  try {
    if (path !== "artifact") return await proxyJson(path);
    const artifactPath = new URL(request.url).searchParams.get("path");
    if (!isSafeArtifactPath(artifactPath)) {
      return jsonError("A safe relative artifact path is required.", 400);
    }
    const headers = new Headers();
    const range = request.headers.get("range");
    if (range) headers.set("Range", range);
    const response = await backendFetch(
      `artifact?path=${encodeURIComponent(artifactPath)}`,
      { headers }
    );
    if (!response.ok) {
      return jsonError(
        response.status === 404 ? "Artifact not found." : "Artifact unavailable.",
        response.status === 404 ? 404 : 502
      );
    }
    const outgoing = new Headers({ "Cache-Control": "no-store" });
    for (const name of [
      "accept-ranges",
      "content-length",
      "content-range",
      "content-type",
    ]) {
      const value = response.headers.get(name);
      if (value) outgoing.set(name, value);
    }
    return new Response(response.body, {
      status: response.status,
      headers: outgoing,
    });
  } catch {
    return jsonError("Arm simulation backend unavailable on localhost:8812.", 503);
  }
}

export async function handleArmPost(request: Request, path: string) {
  if (!ARM_POST_PATHS.has(path)) {
    return jsonError("Unknown Arm API path.", 404);
  }
  try {
    const body = validateMutation(path, await readJsonBody(request));
    if (!body) return jsonError(`Invalid ${path} request.`, 400);
    return await proxyJson(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (error) {
    if (
      error instanceof Error &&
      (error.message.includes("Request body") || error.message.includes("JSON"))
    ) {
      return jsonError(error.message, 400);
    }
    return jsonError("Arm simulation backend unavailable on localhost:8812.", 503);
  }
}
