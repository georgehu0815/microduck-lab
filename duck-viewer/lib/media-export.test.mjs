import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { copyFile, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

const video = Buffer.from("test video bytes");
const videoHash = createHash("sha256").update(video).digest("hex");

async function fixture(context, mode, setup) {
  const directory = await mkdtemp(path.join(tmpdir(), "microduck-export-test-"));
  const root = path.join(directory, "viewer");
  const destination = path.join(root, "public/static");
  await mkdir(path.join(root, "scripts"), { recursive: true });
  await mkdir(destination, { recursive: true });
  await writeFile(path.join(destination, "previous.txt"), "keep previous snapshot");
  const wingpod = setup ? await setup(destination) : undefined;
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
  return { destination, result, wingpod };
}

async function addWingPod(destination) {
  const wingpodRoot = path.join(destination, "wingpod");
  const boundFiles = new Map([
    ["hero.png", Buffer.from("WingPod hero")],
    ["cases/nominal-0.mp4", Buffer.from("WingPod nominal case video")],
    ["cases/nominal-0.json", Buffer.from('{"taskSuccess":true}')],
  ]);
  const manifest = { files: {}, all_cases_included: false };
  for (const [filename, bytes] of boundFiles) {
    await mkdir(path.dirname(path.join(wingpodRoot, filename)), { recursive: true });
    await writeFile(path.join(wingpodRoot, filename), bytes);
    manifest.files[filename] = {
      sha256: createHash("sha256").update(bytes).digest("hex"),
      bytes: bytes.length,
      source: `artifacts/${filename}`,
    };
  }
  const manifestBytes = Buffer.from(JSON.stringify(manifest, null, 2) + "\n");
  await writeFile(path.join(wingpodRoot, "manifest.json"), manifestBytes);
  await writeFile(path.join(wingpodRoot, "unbound.txt"), "must not be published");
  return { boundFiles, manifest, manifestBytes, wingpodRoot };
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

test("media export preserves only hash-bound WingPod files and its exact manifest", async (context) => {
  const { destination, result, wingpod } = await fixture(context, "success", addWingPod);
  assert.equal(result.code, 0, result.errors);
  const manifest = JSON.parse(await readFile(path.join(destination, "manifest.json")));
  const expectedFiles = new Map([...wingpod.boundFiles, ["manifest.json", wingpod.manifestBytes]]);
  assert.equal(manifest.files.length, 4 + expectedFiles.size);
  for (const [filename, bytes] of expectedFiles) {
    assert.deepEqual(await readFile(path.join(destination, "wingpod", filename)), bytes);
    const record = manifest.files.find((file) => file.path === `/static/wingpod/${filename}`);
    assert.equal(record.bytes, bytes.length);
    assert.equal(record.sha256, createHash("sha256").update(bytes).digest("hex"));
  }
  await assert.rejects(readFile(path.join(destination, "wingpod/unbound.txt")), { code: "ENOENT" });
});

for (const failure of ["hash", "bytes", "malformed", "traversal", "absolute", "backslash", "encoded-traversal", "file-symlink", "directory-symlink", "manifest-symlink", "root-symlink", "missing-file", "not-regular"]) {
  test(`media export retains the previous snapshot for WingPod ${failure}`, async (context) => {
    let previousManifest;
    const { destination, result } = await fixture(context, "success", async (destination) => {
      const wingpod = await addWingPod(destination);
      const manifestPath = path.join(wingpod.wingpodRoot, "manifest.json");
      const heroPath = path.join(wingpod.wingpodRoot, "hero.png");
      if (failure === "hash") {
        await writeFile(heroPath, Buffer.from("WingPod fake"));
      } else if (failure === "bytes") {
        wingpod.manifest.files["hero.png"].bytes += 1;
      } else if (["traversal", "absolute", "backslash", "encoded-traversal"].includes(failure)) {
        const filenames = {
          traversal: "../previous.txt",
          absolute: "/previous.txt",
          backslash: "..\\previous.txt",
          "encoded-traversal": "%2e%2e/previous.txt",
        };
        wingpod.manifest.files = { [filenames[failure]]: wingpod.manifest.files["hero.png"] };
      } else if (failure === "file-symlink") {
        await copyFile(heroPath, path.join(destination, "linked-hero.png"));
        await rm(heroPath);
        await symlink("../linked-hero.png", heroPath);
      } else if (failure === "directory-symlink") {
        const external = path.join(destination, "linked-cases");
        await mkdir(external);
        for (const [filename, bytes] of wingpod.boundFiles) {
          if (filename.startsWith("cases/")) await writeFile(path.join(external, path.basename(filename)), bytes);
        }
        await rm(path.join(wingpod.wingpodRoot, "cases"), { recursive: true });
        await symlink("../linked-cases", path.join(wingpod.wingpodRoot, "cases"));
      } else if (failure === "missing-file" || failure === "not-regular") {
        await rm(heroPath);
        if (failure === "not-regular") await mkdir(heroPath);
      }
      previousManifest = failure === "malformed" ? Buffer.from("{invalid json") : Buffer.from(JSON.stringify(wingpod.manifest));
      await writeFile(manifestPath, previousManifest);
      if (failure === "manifest-symlink") {
        await writeFile(path.join(destination, "linked-manifest.json"), previousManifest);
        await rm(manifestPath);
        await symlink("../linked-manifest.json", manifestPath);
      } else if (failure === "root-symlink") {
        const linked = path.join(destination, "linked-wingpod");
        await mkdir(linked);
        await writeFile(path.join(linked, "manifest.json"), previousManifest);
        await rm(wingpod.wingpodRoot, { recursive: true });
        await symlink("linked-wingpod", wingpod.wingpodRoot);
      }
      return wingpod;
    });
    assert.notEqual(result.code, 0, result.errors);
    if (failure === "hash") assert.match(result.errors, /Changed WingPod hash/);
    assert.equal(await readFile(path.join(destination, "previous.txt"), "utf8"), "keep previous snapshot");
    assert.deepEqual(await readFile(path.join(destination, "wingpod/manifest.json")), previousManifest);
    await assert.rejects(readFile(path.join(destination, "manifest.json")), { code: "ENOENT" });
  });
}
