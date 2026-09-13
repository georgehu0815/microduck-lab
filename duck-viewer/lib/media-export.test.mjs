import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

const video = Buffer.from("test video bytes");
const videoHash = createHash("sha256").update(video).digest("hex");

async function fixture(context, mode) {
  const directory = await mkdtemp(path.join(tmpdir(), "microduck-export-test-"));
  const root = path.join(directory, "viewer");
  const destination = path.join(root, "public/static");
  await mkdir(path.join(root, "scripts"), { recursive: true });
  await mkdir(destination, { recursive: true });
  await writeFile(path.join(destination, "previous.txt"), "keep previous snapshot");
  const script = path.join(root, "scripts/export-static-media.mjs");
  await copyFile(new URL("../scripts/export-static-media.mjs", import.meta.url), script);
  const snapshot = { renderVerified: true, renderEvidenceId: videoHash, evaluation: { source_sha256: "policy-hash" } };
  const server = createServer((request, response) => {
    const url = new URL(request.url, "http://localhost");
    if (url.pathname === "/api/arm/videos") {
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify({ hardwareEnabled: false, videos: [{ videoUrl: "/arm.mp4", videoHash: mode === "bad-hash" ? "0".repeat(64) : videoHash }] }));
    } else if (url.pathname === "/api/rlx/runs") {
      response.end(JSON.stringify({ runs: [{ experimentId: "dance", runName: "latest", video: true, renderVerified: true, renderEvidenceId: videoHash }] }));
    } else if (url.pathname === "/api/rlx") {
      response.end(JSON.stringify(snapshot));
    } else if (url.pathname === "/api/rlx/artifact" && url.searchParams.get("kind") === "sheet") {
      response.writeHead(mode === "missing-sheet" ? 500 : 200).end("test sheet bytes");
    } else if (url.pathname === "/arm.mp4" || url.pathname === "/api/rlx/artifact") {
      response.end(video);
    } else response.writeHead(404).end();
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  context.after(async () => {
    await new Promise((resolve) => server.close(resolve));
    await rm(directory, { recursive: true, force: true });
  });
  const result = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [script], { env: { ...process.env, DUCK_VIEWER_CATALOG_URL: `http://127.0.0.1:${server.address().port}` } });
    let errors = "";
    child.stdout.resume();
    child.stderr.on("data", (chunk) => { errors += chunk; });
    child.on("error", reject);
    child.on("close", (code) => resolve({ code, errors }));
  });
  return { destination, result };
}

test("media export uses content hashes and installs a complete verified snapshot", async (context) => {
  const { destination, result } = await fixture(context, "success");
  assert.equal(result.code, 0, result.errors);
  const catalog = JSON.parse(await readFile(path.join(destination, "arm-videos/catalog.json")));
  assert.equal(catalog.videos[0].videoUrl, `/static/arm-videos/media/${videoHash}.mp4`);
  const manifest = JSON.parse(await readFile(path.join(destination, "manifest.json")));
  assert.equal(manifest.files.length, 4);
  assert.equal(manifest.files.find((file) => file.path.endsWith("video.mp4")).sha256, videoHash);
  await assert.rejects(readFile(path.join(destination, "previous.txt")), { code: "ENOENT" });
});

for (const mode of ["bad-hash", "missing-sheet"]) {
  test(`media export preserves previous snapshot when ${mode}`, async (context) => {
    const { destination, result } = await fixture(context, mode);
    assert.notEqual(result.code, 0);
    assert.equal(await readFile(path.join(destination, "previous.txt"), "utf8"), "keep previous snapshot");
    await assert.rejects(readFile(path.join(destination, "manifest.json")), { code: "ENOENT" });
  });
}
