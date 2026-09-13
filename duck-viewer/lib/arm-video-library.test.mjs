import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import { test } from "node:test";
import ts from "typescript";

const nativeRequire = createRequire(import.meta.url);
const moduleCache = new Map();

function load(filename) {
  const filePath = path.resolve(path.dirname(new URL(import.meta.url).pathname), filename);
  if (moduleCache.has(filePath)) return moduleCache.get(filePath);
  const source = fs.readFileSync(filePath, "utf8");
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      esModuleInterop: true,
    },
  });
  const exports = {};
  moduleCache.set(filePath, exports);
  const localRequire = (specifier) => {
    if (specifier.startsWith(".")) {
      const resolved = path.resolve(path.dirname(filePath), specifier);
      return load(path.extname(resolved) ? resolved : `${resolved}.ts`);
    }
    return nativeRequire(specifier);
  };
  vm.runInNewContext(
    outputText,
    {
      Buffer,
      Headers,
      Map,
      Object,
      Request,
      Response,
      Set,
      URL,
      exports,
      process,
      require: localRequire,
    },
    { filename: filePath }
  );
  return exports;
}

const { listArmVideoLibrary, serveArmMedia } = load("arm-video-library.ts");
const plain = (value) => JSON.parse(JSON.stringify(value));
const CURRENT = {
  environment: "a".repeat(64),
  pipeline: "b".repeat(64),
};
const HISTORICAL = {
  environment: "c".repeat(64),
  pipeline: "d".repeat(64),
};

function hash(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

function fixture(t) {
  const workspaceRoot = fs.mkdtempSync(path.join(os.tmpdir(), "arm-videos-"));
  const artifactRoot = path.join(workspaceRoot, "rlx", "runs", "arm");
  fs.mkdirSync(artifactRoot, { recursive: true });
  t.after(() => fs.rmSync(workspaceRoot, { recursive: true, force: true }));

  function write(relativePath, value) {
    const filePath = path.join(artifactRoot, relativePath);
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(
      filePath,
      Buffer.isBuffer(value)
        ? value
        : typeof value === "string"
          ? value
          : JSON.stringify(value, null, 2)
    );
    return filePath;
  }

  function addEpisode({
    batch,
    id,
    caseId = "arm-carry-v1",
    trainingSeed = 101,
    evaluationSeed = 80000,
    bytes = Buffer.from(`video:${id}`),
    passed = true,
    failedGates = [],
    sourceHashes = CURRENT,
    manifest = true,
    receiptOverrides = {},
  }) {
    const videoRelative = `${batch}/videos/${id}/rollout.mp4`;
    const videoPath = write(videoRelative, bytes);
    const videoHash = hash(bytes);
    const receiptRelative = `${batch}/videos/${id}/render-receipt.json`;
    const gates = {
      no_drop: !failedGates.includes("no_drop"),
      stable: !failedGates.includes("stable"),
    };
    const receipt = {
      case_id: caseId,
      checkpoint: path.join(
        workspaceRoot,
        "rlx",
        "runs",
        "arm",
        "training",
        caseId,
        `seed-${trainingSeed}`,
        "ppo-residual.zip"
      ),
      episode: {
        seed: evaluationSeed,
        elapsed_seconds: 4,
        video_timing: { encoded_duration_seconds: 4.25 },
        metrics: { passed, gates },
      },
      source_hashes: sourceHashes,
      video_sha256: videoHash,
      media_hashes: { "rollout.mp4": videoHash },
      ...receiptOverrides,
    };
    const receiptPath = write(receiptRelative, receipt);
    if (manifest) {
      const manifestPath = path.join(artifactRoot, batch, "video-evidence.json");
      let evidence = { videos: [] };
      if (fs.existsSync(manifestPath)) {
        evidence = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
      }
      evidence.videos.push({
        id,
        case_id: caseId,
        training_seed: trainingSeed,
        expected_episode: { seed: evaluationSeed },
        selection_reasons: passed ? ["first-new-seed"] : ["historical-failure"],
        receipt: receiptPath,
        receipt_sha256: hash(fs.readFileSync(receiptPath)),
        video_sha256: videoHash,
        duration_seconds: 4.25,
        passed,
        failed_gates: failedGates,
      });
      write(`${batch}/video-evidence.json`, evidence);
    }
    return { bytes, receiptPath, receiptRelative, videoHash, videoPath, videoRelative };
  }

  function addCompilation({
    batch,
    name,
    chapters,
    sourceHashes = CURRENT,
    bytes = Buffer.from(`compilation:${batch}:${name}`),
  }) {
    const videoRelative = `${batch}/${name}.mp4`;
    write(videoRelative, bytes);
    write(`${batch}/${name}.chapters.json`, {
      path: `${name}.mp4`,
      sha256: hash(bytes),
      duration_s: chapters.length * 4.25,
      full_decode_passed: true,
      chapters: chapters.map((chapter, index) => ({
        start_s: index * 4.25,
        end_s: (index + 1) * 4.25,
        id: chapter.id,
        passed: chapter.passed,
        source_sha256: chapter.videoHash,
      })),
    });
    write(`${batch}/protocol.json`, { source_hashes: sourceHashes });
    return videoRelative;
  }

  return {
    addCompilation,
    addEpisode,
    artifactRoot,
    workspaceRoot,
    write,
  };
}

test("discovers, verifies, and deduplicates episode videos with aliases", async (t) => {
  const f = fixture(t);
  const episode = f.addEpisode({
    batch: "current-batch",
    id: "arm-carry-v1-train101-eval80000",
  });
  const aliasRelative = "training/arm-carry-v1/seed-101/rollout.mp4";
  f.write(aliasRelative, episode.bytes);
  const aliasReceipt = JSON.parse(fs.readFileSync(episode.receiptPath, "utf8"));
  f.write("training/arm-carry-v1/seed-101/render-receipt.json", aliasReceipt);

  const library = await listArmVideoLibrary({
    workspaceRoot: f.workspaceRoot,
    currentHashes: CURRENT,
  });

  assert.equal(library.totalFiles, 2);
  assert.equal(library.uniqueVideos, 1);
  assert.equal(library.hardwareEnabled, false);
  assert.deepEqual(plain(library.videos[0]), {
    id: "arm-carry-v1-train101-eval80000",
    title: "定点搬运 · train 101 / eval 80000",
    caseIds: ["arm-carry-v1"],
    batch: "current-batch",
    path: episode.videoRelative,
    aliases: [aliasRelative],
    kind: "episode",
    provenance: "current",
    outcome: "passed",
    trainingSeed: 101,
    evaluationSeed: 80000,
    durationSeconds: 4.25,
    failedGates: [],
    selectionReasons: ["first-new-seed"],
    videoUrl: `/api/arm/media?path=${encodeURIComponent(episode.videoRelative)}`,
    evidenceUrl: `/api/arm/media?path=${encodeURIComponent(episode.receiptRelative)}`,
  });
});

test("keeps historical failures separate from provenance and builds conservative compilations", async (t) => {
  const f = fixture(t);
  const passed = f.addEpisode({
    batch: "historical-batch",
    id: "arm-reach-v1-train101-eval80000",
    caseId: "arm-reach-v1",
    sourceHashes: HISTORICAL,
  });
  const failed = f.addEpisode({
    batch: "historical-batch",
    id: "arms-co-carry-v1-train202-eval80307",
    caseId: "arms-co-carry-v1",
    trainingSeed: 202,
    evaluationSeed: 80307,
    passed: false,
    failedGates: ["no_drop"],
    sourceHashes: HISTORICAL,
  });
  f.addCompilation({
    batch: "historical-batch",
    name: "mixed-evidence",
    chapters: [
      { id: "pass", passed: true, videoHash: passed.videoHash },
      { id: "fail", passed: false, videoHash: failed.videoHash },
    ],
    sourceHashes: HISTORICAL,
  });
  f.addCompilation({
    batch: "historical-batch",
    name: "bad-compilation",
    chapters: [{ id: "wrong", passed: true, videoHash: failed.videoHash }],
    sourceHashes: HISTORICAL,
  });

  const library = await listArmVideoLibrary({
    workspaceRoot: f.workspaceRoot,
    currentHashes: CURRENT,
  });
  const failedVideo = library.videos.find((video) =>
    video.id.includes("arms-co-carry")
  );
  const mixed = library.videos.find((video) => video.path.endsWith("mixed-evidence.mp4"));
  const bad = library.videos.find((video) => video.path.endsWith("bad-compilation.mp4"));

  assert.equal(failedVideo.provenance, "historical");
  assert.equal(failedVideo.outcome, "failed");
  assert.deepEqual(plain(failedVideo.failedGates), ["no_drop"]);
  assert.equal(mixed.provenance, "historical");
  assert.equal(mixed.outcome, "mixed");
  assert.deepEqual(
    plain(mixed.caseIds.sort()),
    ["arm-reach-v1", "arms-co-carry-v1"]
  );
  assert.equal(bad.provenance, "unverified");
  assert.equal(bad.outcome, "unverified");
  assert.ok(
    library.warnings.some((warning) =>
      warning.includes("Unverified compilation constituents")
    )
  );
});

test("fails closed for bad hashes and malformed metadata without dropping MP4 files", async (t) => {
  const f = fixture(t);
  const badHash = f.addEpisode({
    batch: "bad-hash",
    id: "arm-carry-v1-train101-eval90000",
    receiptOverrides: { video_sha256: "f".repeat(64) },
  });
  f.write("malformed/videos/arm-reach-v1-train101-eval7/rollout.mp4", "raw-video");
  f.write(
    "malformed/videos/arm-reach-v1-train101-eval7/render-receipt.json",
    "{"
  );
  f.write("malformed/video-evidence.json", "{");

  const library = await listArmVideoLibrary({
    workspaceRoot: f.workspaceRoot,
    currentHashes: CURRENT,
  });
  const invalid = library.videos.find((video) => video.path === badHash.videoRelative);
  const malformed = library.videos.find((video) => video.path.startsWith("malformed/"));

  assert.equal(library.totalFiles, 2);
  assert.equal(invalid.provenance, "unverified");
  assert.equal(invalid.outcome, "unverified");
  assert.equal(invalid.evidenceUrl, null);
  assert.equal(malformed.provenance, "unverified");
  assert.deepEqual(plain(malformed.caseIds), []);
  assert.equal(malformed.trainingSeed, null);
  assert.equal(malformed.evaluationSeed, null);
  assert.ok(library.warnings.some((warning) => warning.includes("Invalid render receipt")));
  assert.ok(library.warnings.some((warning) => warning.includes("Malformed video evidence")));
});

test("returns an empty fail-closed library when roots or current hashes are unavailable", async (t) => {
  const f = fixture(t);
  fs.rmSync(f.artifactRoot, { recursive: true, force: true });
  const missing = await listArmVideoLibrary({
    workspaceRoot: f.workspaceRoot,
    currentHashes: CURRENT,
  });
  assert.deepEqual(plain(missing.videos), []);
  assert.equal(missing.totalFiles, 0);
  assert.ok(missing.warnings.includes("Arm artifact root is missing."));

  fs.mkdirSync(f.artifactRoot, { recursive: true });
  f.addEpisode({
    batch: "no-current-hashes",
    id: "arm-carry-v1-train101-eval1",
  });
  const unavailable = await listArmVideoLibrary({
    workspaceRoot: f.workspaceRoot,
    currentHashes: null,
  });
  assert.equal(unavailable.videos[0].provenance, "unverified");
  assert.equal(unavailable.videos[0].outcome, "unverified");
});

test("serves full, ranged, suffix, and HEAD MP4 responses", async (t) => {
  const f = fixture(t);
  const episode = f.addEpisode({
    batch: "media",
    id: "arm-carry-v1-train101-eval2",
    bytes: Buffer.from("0123456789"),
  });
  const url = `http://local/api/arm/media?path=${encodeURIComponent(episode.videoRelative)}`;
  const options = { workspaceRoot: f.workspaceRoot, currentHashes: CURRENT };

  const full = await serveArmMedia(new Request(url), options);
  assert.equal(full.status, 200);
  assert.equal(full.headers.get("content-length"), "10");
  assert.equal(Buffer.from(await full.arrayBuffer()).toString(), "0123456789");

  const range = await serveArmMedia(
    new Request(url, { headers: { Range: "bytes=2-5" } }),
    options
  );
  assert.equal(range.status, 206);
  assert.equal(range.headers.get("content-range"), "bytes 2-5/10");
  assert.equal(Buffer.from(await range.arrayBuffer()).toString(), "2345");

  const suffix = await serveArmMedia(
    new Request(url, { headers: { Range: "bytes=-3" } }),
    options
  );
  assert.equal(suffix.status, 206);
  assert.equal(Buffer.from(await suffix.arrayBuffer()).toString(), "789");

  const invalid = await serveArmMedia(
    new Request(url, { headers: { Range: "bytes=20-30" } }),
    options
  );
  assert.equal(invalid.status, 416);
  assert.equal(invalid.headers.get("content-range"), "bytes */10");

  const head = await serveArmMedia(new Request(url, { method: "HEAD" }), options);
  assert.equal(head.status, 200);
  assert.equal(head.headers.get("content-length"), "10");
  assert.equal((await head.arrayBuffer()).byteLength, 0);
});

test("serves only catalog-authorized JSON and denies traversal and symlinks", async (t) => {
  const f = fixture(t);
  const episode = f.addEpisode({
    batch: "secure",
    id: "arm-carry-v1-train101-eval3",
  });
  f.write("private.json", { secret: true });
  const outside = path.join(os.tmpdir(), `arm-video-outside-${path.basename(f.workspaceRoot)}.mp4`);
  fs.writeFileSync(outside, "outside");
  t.after(() => fs.rmSync(outside, { force: true }));
  fs.symlinkSync(outside, path.join(f.artifactRoot, "linked.mp4"));
  const options = { workspaceRoot: f.workspaceRoot, currentHashes: CURRENT };

  const receipt = await serveArmMedia(
    new Request(
      `http://local/api/arm/media?path=${encodeURIComponent(episode.receiptRelative)}`
    ),
    options
  );
  assert.equal(receipt.status, 200);
  assert.equal(receipt.headers.get("content-type"), "application/json");

  for (const relativePath of [
    "private.json",
    "../outside.mp4",
    "%2e%2e/outside.mp4",
    "linked.mp4",
  ]) {
    const response = await serveArmMedia(
      new Request(
        `http://local/api/arm/media?path=${encodeURIComponent(relativePath)}`
      ),
      options
    );
    assert.equal(response.status, 404, relativePath);
  }
});

test("real repository inventory includes every regular MP4 and expected evidence classes", async () => {
  const workspaceRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../..");
  const artifactRoot = path.join(workspaceRoot, "rlx", "runs", "arm");
  const physical = [];
  function walk(directory) {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const candidate = path.join(directory, entry.name);
      if (entry.isDirectory()) walk(candidate);
      else if (entry.isFile() && entry.name.endsWith(".mp4")) physical.push(candidate);
    }
  }
  walk(artifactRoot);
  const uniqueHashes = new Set(
    physical.map((filePath) => hash(fs.readFileSync(filePath)))
  );

  const library = await listArmVideoLibrary({ workspaceRoot });

  assert.equal(library.totalFiles, physical.length);
  assert.equal(library.uniqueVideos, uniqueHashes.size);
  assert.equal(new Set(library.videos.map((video) => video.id)).size, library.videos.length);
  assert.ok(
    library.videos.some(
      (video) =>
        video.path ===
          "strict-heldout-20260912-v3/six-cases-overview.mp4" &&
        video.provenance === "current" &&
        video.outcome === "passed"
    )
  );
  assert.ok(
    library.videos.some(
      (video) =>
        video.path === "recheck-20260912-v2/all-failures.mp4" &&
        video.provenance === "historical" &&
        video.outcome === "failed"
    )
  );
  assert.ok(
    library.videos.some(
      (video) =>
        video.aliases.length > 0 &&
        video.path.includes("strict-heldout-20260912-v3/videos/")
    )
  );
});
