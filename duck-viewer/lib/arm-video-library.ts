import { createHash } from "node:crypto";
import {
  lstat,
  open,
  readFile,
  readdir,
  realpath,
  stat,
} from "node:fs/promises";
import path from "node:path";

import {
  ARM_CASES,
  isArmCaseId,
  type ArmCaseId,
} from "./arm-contract";
import type { ArmVideo, ArmVideoLibrary } from "./arm-videos";

const ENVIRONMENT_PATH = path.join("rlx", "rlx", "environments", "arm.py");
const PIPELINE_PATH = path.join("rlx", "examples", "ppo_microduck_arm.py");
const SHA256 = /^[a-f0-9]{64}$/;
const CASE_TITLES = new Map(ARM_CASES.map((item) => [item.id, item.title]));

export interface ArmVideoLibraryOptions {
  workspaceRoot?: string;
  artifactRoot?: string;
  currentHashes?: {
    environment: string;
    pipeline: string;
  } | null;
}

interface JsonRecord {
  [key: string]: unknown;
}

interface FileRecord {
  absolutePath: string;
  relativePath: string;
  hash: string;
  size: number;
}

interface Candidate {
  file: FileRecord;
  id: string;
  title: string;
  caseIds: ArmCaseId[];
  batch: string;
  kind: "episode" | "compilation";
  provenance: ArmVideo["provenance"];
  outcome: ArmVideo["outcome"];
  trainingSeed: number | null;
  evaluationSeed: number | null;
  durationSeconds: number | null;
  failedGates: string[];
  selectionReasons: string[];
  evidenceRecordedAt: string | null;
  receiptRecordedAt: string | null;
  evidencePath: string | null;
  verified: boolean;
  source: "manifest" | "receipt" | "chapters" | "none";
}

interface EvidenceEntry {
  id: string;
  caseId: ArmCaseId;
  trainingSeed: number;
  evaluationSeed: number;
  durationSeconds: number;
  passed: boolean;
  failedGates: string[];
  selectionReasons: string[];
  videoHash: string;
  receiptHash: string;
  receiptPath: string;
  evidenceRecordedAt: string | null;
}

interface ReceiptEvidence {
  caseId: ArmCaseId;
  trainingSeed: number | null;
  evaluationSeed: number;
  durationSeconds: number | null;
  passed: boolean;
  failedGates: string[];
  provenance: ArmVideo["provenance"];
  receiptRecordedAt: string | null;
}

interface ChapterEvidence {
  id: string;
  passed: boolean;
  sourceHash: string;
}

interface CompilationEvidence {
  durationSeconds: number;
  chapters: ChapterEvidence[];
}

interface MediaFile {
  absolutePath: string;
  relativePath: string;
  size: number;
  contentType: "application/json" | "video/mp4";
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function safeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function stringArray(value: unknown): string[] | null {
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
    return null;
  }
  return [...new Set(value)];
}

function recordedAt(value: unknown): string | null {
  if (
    typeof value !== "string" ||
    !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(value) ||
    !Number.isFinite(Date.parse(value))
  ) {
    return null;
  }
  return value.endsWith("+00:00") ? `${value.slice(0, -6)}Z` : value;
}

function slashPath(value: string): string {
  return value.split(path.sep).join("/");
}

function isWithin(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative === "" ||
    (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function workspaceRootFrom(cwd: string): string {
  return path.basename(cwd) === "duck-viewer" ? path.dirname(cwd) : cwd;
}

function roots(options: ArmVideoLibraryOptions) {
  const workspaceRoot = path.resolve(
    options.workspaceRoot ?? workspaceRootFrom(process.cwd())
  );
  const artifactRoot = path.resolve(
    /* turbopackIgnore: true */
    options.artifactRoot ?? path.join(workspaceRoot, "rlx", "runs", "arm")
  );
  return { workspaceRoot, artifactRoot };
}

function isSafeRelativePath(value: unknown): value is string {
  if (
    typeof value !== "string" ||
    value.length < 1 ||
    value.length > 1024 ||
    value.includes("\0") ||
    value.includes("\\") ||
    value.startsWith("/") ||
    /^[a-z][a-z0-9+.-]*:/i.test(value)
  ) {
    return false;
  }
  return value.split("/").every((part) =>
    part.length > 0 && part !== "." && part !== ".." && !part.startsWith("~")
  );
}

async function sha256(filePath: string): Promise<string> {
  return createHash("sha256").update(await readFile(filePath)).digest("hex");
}

async function readJson(filePath: string): Promise<unknown> {
  return JSON.parse(await readFile(filePath, "utf8"));
}

async function currentSourceHashes(
  workspaceRoot: string,
  supplied: ArmVideoLibraryOptions["currentHashes"]
): Promise<ArmVideoLibraryOptions["currentHashes"]> {
  if (supplied !== undefined) return supplied;
  try {
    const [environment, pipeline] = await Promise.all([
      sha256(path.join(workspaceRoot, ENVIRONMENT_PATH)),
      sha256(path.join(workspaceRoot, PIPELINE_PATH)),
    ]);
    return { environment, pipeline };
  } catch {
    return null;
  }
}

async function scanFiles(
  artifactRoot: string,
  warnings: string[]
): Promise<{ videos: FileRecord[]; jsonFiles: string[]; resolvedRoot: string }> {
  const videos: FileRecord[] = [];
  const jsonFiles: string[] = [];
  let resolvedRoot: string;
  try {
    resolvedRoot = await realpath(artifactRoot);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      warnings.push("Arm artifact root is missing.");
      return { videos, jsonFiles, resolvedRoot: artifactRoot };
    }
    throw error;
  }

  async function visit(directory: string): Promise<void> {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries) {
      const candidate = path.join(directory, entry.name);
      if (entry.isSymbolicLink()) {
        if (entry.name.endsWith(".mp4") || entry.name.endsWith(".json")) {
          warnings.push(`Skipped symlink artifact: ${slashPath(path.relative(artifactRoot, candidate))}`);
        }
        continue;
      }
      if (entry.isDirectory()) {
        await visit(candidate);
        continue;
      }
      if (!entry.isFile()) continue;
      if (entry.name.endsWith(".json")) {
        const resolved = await realpath(candidate);
        if (isWithin(resolvedRoot, resolved)) jsonFiles.push(resolved);
      }
      if (!entry.name.toLowerCase().endsWith(".mp4")) continue;
      const resolved = await realpath(candidate);
      if (!isWithin(resolvedRoot, resolved)) {
        warnings.push(`Skipped artifact outside root: ${entry.name}`);
        continue;
      }
      const info = await stat(resolved);
      videos.push({
        absolutePath: resolved,
        relativePath: slashPath(path.relative(artifactRoot, candidate)),
        hash: await sha256(resolved),
        size: info.size,
      });
    }
  }

  await visit(artifactRoot);
  videos.sort((left, right) => left.relativePath.localeCompare(right.relativePath));
  jsonFiles.sort();
  return { videos, jsonFiles, resolvedRoot };
}

function sourceHashes(value: unknown): { environment: string; pipeline: string } | null {
  if (
    !isRecord(value) ||
    typeof value.environment !== "string" ||
    typeof value.pipeline !== "string" ||
    !SHA256.test(value.environment) ||
    !SHA256.test(value.pipeline)
  ) {
    return null;
  }
  return { environment: value.environment, pipeline: value.pipeline };
}

function provenanceFor(
  hashes: { environment: string; pipeline: string } | null,
  current: ArmVideoLibraryOptions["currentHashes"]
): ArmVideo["provenance"] {
  if (!hashes || !current) return "unverified";
  if (
    current &&
    hashes.environment === current.environment &&
    hashes.pipeline === current.pipeline
  ) {
    return "current";
  }
  return "historical";
}

function failedGatesFrom(value: unknown): string[] | null {
  if (!isRecord(value)) return null;
  return Object.entries(value)
    .filter(([, passed]) => passed === false)
    .map(([gate]) => gate)
    .sort();
}

function parseEvidenceEntry(
  value: unknown,
  batchDir: string,
  evidenceRecordedAt: string | null
): EvidenceEntry | null {
  if (!isRecord(value)) return null;
  const failedGates = stringArray(value.failed_gates);
  const selectionReasons = stringArray(value.selection_reasons);
  const expected = isRecord(value.expected_episode) ? value.expected_episode : null;
  if (
    typeof value.id !== "string" ||
    !isArmCaseId(value.case_id) ||
    !safeInteger(value.training_seed) ||
    !expected ||
    !safeInteger(expected.seed) ||
    !finiteNumber(value.duration_seconds) ||
    value.duration_seconds < 0 ||
    typeof value.passed !== "boolean" ||
    !failedGates ||
    !selectionReasons ||
    typeof value.video_sha256 !== "string" ||
    !SHA256.test(value.video_sha256) ||
    typeof value.receipt_sha256 !== "string" ||
    !SHA256.test(value.receipt_sha256) ||
    typeof value.receipt !== "string"
  ) {
    return null;
  }
  const receiptPath = path.isAbsolute(value.receipt)
    ? path.resolve(value.receipt)
    : path.resolve(batchDir, value.receipt);
  return {
    id: value.id,
    caseId: value.case_id,
    trainingSeed: value.training_seed,
    evaluationSeed: expected.seed,
    durationSeconds: value.duration_seconds,
    passed: value.passed,
    failedGates,
    selectionReasons,
    videoHash: value.video_sha256,
    receiptHash: value.receipt_sha256,
    receiptPath,
    evidenceRecordedAt,
  };
}

async function evidenceEntries(
  jsonFiles: string[],
  artifactRoot: string,
  warnings: string[]
): Promise<{
  entries: Map<string, EvidenceEntry>;
  batchRecordedAt: Map<string, string>;
}> {
  const entries = new Map<string, EvidenceEntry>();
  const batchRecordedAt = new Map<string, string>();
  for (const filePath of jsonFiles.filter((item) => path.basename(item) === "video-evidence.json")) {
    const relative = slashPath(path.relative(artifactRoot, filePath));
    try {
      const raw = await readJson(filePath);
      if (
        !isRecord(raw) ||
        raw.evidence_verification_passed !== true ||
        !Array.isArray(raw.videos)
      ) {
        warnings.push(`Malformed video evidence: ${relative}`);
        continue;
      }
      const manifestRecordedAt = recordedAt(raw.created_at);
      if (manifestRecordedAt) {
        batchRecordedAt.set(path.dirname(filePath), manifestRecordedAt);
      }
      for (const item of raw.videos) {
        const parsed = parseEvidenceEntry(
          item,
          path.dirname(filePath),
          manifestRecordedAt
        );
        if (!parsed) {
          warnings.push(`Malformed video evidence entry: ${relative}`);
          continue;
        }
        const expectedPath = slashPath(
          path.relative(
            artifactRoot,
            path.join(path.dirname(filePath), "videos", parsed.id, "rollout.mp4")
          )
        );
        entries.set(expectedPath, parsed);
      }
    } catch {
      warnings.push(`Malformed video evidence: ${relative}`);
    }
  }
  return { entries, batchRecordedAt };
}

function parseReceipt(
  raw: unknown,
  actualHash: string,
  currentHashes: ArmVideoLibraryOptions["currentHashes"]
): ReceiptEvidence | null {
  if (!isRecord(raw) || !isArmCaseId(raw.case_id)) return null;
  const episode = isRecord(raw.episode) ? raw.episode : null;
  const metrics = episode && isRecord(episode.metrics) ? episode.metrics : null;
  const receiptVideoHash =
    typeof raw.video_sha256 === "string"
      ? raw.video_sha256
      : isRecord(raw.media_hashes) && typeof raw.media_hashes["rollout.mp4"] === "string"
        ? raw.media_hashes["rollout.mp4"]
        : null;
  const alternateVideoHash =
    isRecord(raw.media_hashes) && typeof raw.media_hashes["rollout.mp4"] === "string"
      ? raw.media_hashes["rollout.mp4"]
      : receiptVideoHash;
  if (
    !receiptVideoHash ||
    !SHA256.test(receiptVideoHash) ||
    receiptVideoHash !== actualHash ||
    alternateVideoHash !== actualHash ||
    !episode ||
    !safeInteger(episode.seed) ||
    !metrics ||
    typeof metrics.passed !== "boolean"
  ) {
    return null;
  }
  const gates = failedGatesFrom(metrics.gates);
  if (!gates) return null;
  const declaredHashes =
    sourceHashes(raw.source_hashes) ??
    (isRecord(raw.checkpoint_metadata)
      ? sourceHashes(raw.checkpoint_metadata.source_hashes)
      : null);
  const duration =
    isRecord(episode.video_timing) &&
    finiteNumber(episode.video_timing.encoded_duration_seconds)
      ? episode.video_timing.encoded_duration_seconds
      : finiteNumber(episode.elapsed_seconds)
        ? episode.elapsed_seconds
        : null;
  return {
    caseId: raw.case_id,
    trainingSeed: seedFromCheckpoint(raw.checkpoint),
    evaluationSeed: episode.seed,
    durationSeconds: duration,
    passed: metrics.passed,
    failedGates: gates,
    provenance: provenanceFor(declaredHashes, currentHashes),
    receiptRecordedAt: recordedAt(raw.created_at),
  };
}

function seedFromCheckpoint(value: unknown): number | null {
  if (typeof value !== "string") return null;
  const match = /(?:^|[/\\])seed-(\d+)(?:[/\\]|$)/.exec(value);
  return match ? Number(match[1]) : null;
}

function receiptPathFor(videoPath: string): string {
  return path.join(path.dirname(videoPath), "render-receipt.json");
}

async function verifiedReceipt(
  file: FileRecord,
  currentHashes: ArmVideoLibraryOptions["currentHashes"],
  warnings: string[]
): Promise<{ evidence: ReceiptEvidence; path: string; hash: string } | null> {
  const receiptPath = receiptPathFor(file.absolutePath);
  try {
    const [raw, hash] = await Promise.all([readJson(receiptPath), sha256(receiptPath)]);
    const evidence = parseReceipt(raw, file.hash, currentHashes);
    if (!evidence) {
      warnings.push(`Invalid render receipt for ${file.relativePath}`);
      return null;
    }
    return { evidence, path: receiptPath, hash };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
      warnings.push(`Malformed render receipt for ${file.relativePath}`);
    }
    return null;
  }
}

function displayTitle(
  caseIds: ArmCaseId[],
  trainingSeed: number | null,
  evaluationSeed: number | null,
  fallback: string
): string {
  if (caseIds.length === 1) {
    const title = CASE_TITLES.get(caseIds[0]) ?? caseIds[0];
    const seeds = [
      trainingSeed === null ? null : `train ${trainingSeed}`,
      evaluationSeed === null ? null : `eval ${evaluationSeed}`,
    ].filter(Boolean);
    return seeds.length ? `${title} · ${seeds.join(" / ")}` : title;
  }
  return fallback
    .replace(/\.mp4$/i, "")
    .split(/[-_]+/)
    .filter(Boolean)
    .map((part) => part[0].toUpperCase() + part.slice(1))
    .join(" ");
}

function baseCandidate(file: FileRecord): Candidate {
  const parts = file.relativePath.split("/");
  const fallback = path.basename(file.relativePath);
  return {
    file,
    id: `video-${file.hash.slice(0, 16)}`,
    title: displayTitle([], null, null, fallback),
    caseIds: [],
    batch: parts[0] ?? "",
    kind: "episode",
    provenance: "unverified",
    outcome: "unverified",
    trainingSeed: null,
    evaluationSeed: null,
    durationSeconds: null,
    failedGates: [],
    selectionReasons: [],
    evidenceRecordedAt: null,
    receiptRecordedAt: null,
    evidencePath: null,
    verified: false,
    source: "none",
  };
}

async function episodeCandidate(
  file: FileRecord,
  artifactRoot: string,
  entry: EvidenceEntry | undefined,
  currentHashes: ArmVideoLibraryOptions["currentHashes"],
  warnings: string[]
): Promise<Candidate> {
  const candidate = baseCandidate(file);
  const receipt = await verifiedReceipt(file, currentHashes, warnings);
  if (entry && receipt) {
    const receiptRelative = slashPath(path.relative(artifactRoot, receipt.path));
    let entryReceiptPath: string | null = null;
    try {
      entryReceiptPath = await realpath(entry.receiptPath);
    } catch {
      // The manifest cannot verify a missing or inaccessible receipt.
    }
    const entryValid =
      entry.videoHash === file.hash &&
      entry.receiptHash === receipt.hash &&
      entryReceiptPath === receipt.path &&
      entry.caseId === receipt.evidence.caseId &&
      entry.evaluationSeed === receipt.evidence.evaluationSeed &&
      entry.passed === receipt.evidence.passed;
    if (entryValid) {
      return {
        ...candidate,
        id: entry.id,
        title: displayTitle(
          [entry.caseId],
          entry.trainingSeed,
          entry.evaluationSeed,
          path.basename(file.relativePath)
        ),
        caseIds: [entry.caseId],
        provenance: receipt.evidence.provenance,
        outcome:
          receipt.evidence.provenance === "unverified"
            ? "unverified"
            : entry.passed
              ? "passed"
              : "failed",
        trainingSeed: entry.trainingSeed,
        evaluationSeed: entry.evaluationSeed,
        durationSeconds: entry.durationSeconds,
        failedGates: entry.failedGates,
        selectionReasons: entry.selectionReasons,
        evidenceRecordedAt: entry.evidenceRecordedAt,
        receiptRecordedAt: receipt.evidence.receiptRecordedAt,
        evidencePath: receiptRelative,
        verified: receipt.evidence.provenance !== "unverified",
        source: "manifest",
      };
    }
    warnings.push(`Evidence hash or receipt mismatch for ${file.relativePath}`);
  }
  if (receipt) {
    const evidence = receipt.evidence;
    return {
      ...candidate,
      id:
        /(?:^|\/)([^/]+-train\d+-eval\d+)(?:\/|$)/.exec(file.relativePath)?.[1] ??
        candidate.id,
      title: displayTitle(
        [evidence.caseId],
        evidence.trainingSeed,
        evidence.evaluationSeed,
        path.basename(file.relativePath)
      ),
      caseIds: [evidence.caseId],
      provenance: evidence.provenance,
      outcome:
        evidence.provenance === "unverified"
          ? "unverified"
          : evidence.passed
            ? "passed"
            : "failed",
      trainingSeed: evidence.trainingSeed,
      evaluationSeed: evidence.evaluationSeed,
      durationSeconds: evidence.durationSeconds,
      failedGates: evidence.failedGates,
      receiptRecordedAt: evidence.receiptRecordedAt,
      evidencePath: slashPath(path.relative(artifactRoot, receipt.path)),
      verified: evidence.provenance !== "unverified",
      source: "receipt",
    };
  }
  return candidate;
}

function parseCompilation(raw: unknown, file: FileRecord): CompilationEvidence | null {
  if (
    !isRecord(raw) ||
    typeof raw.path !== "string" ||
    path.basename(raw.path) !== path.basename(file.relativePath) ||
    typeof raw.sha256 !== "string" ||
    raw.sha256 !== file.hash ||
    !finiteNumber(raw.duration_s) ||
    raw.duration_s < 0 ||
    raw.full_decode_passed !== true ||
    !Array.isArray(raw.chapters) ||
    raw.chapters.length === 0
  ) {
    return null;
  }
  const chapters: ChapterEvidence[] = [];
  for (const value of raw.chapters) {
    if (
      !isRecord(value) ||
      typeof value.id !== "string" ||
      typeof value.passed !== "boolean" ||
      typeof value.source_sha256 !== "string" ||
      !SHA256.test(value.source_sha256)
    ) {
      return null;
    }
    chapters.push({
      id: value.id,
      passed: value.passed,
      sourceHash: value.source_sha256,
    });
  }
  return { durationSeconds: raw.duration_s, chapters };
}

async function compilationCandidate(
  file: FileRecord,
  artifactRoot: string,
  candidatesByHash: Map<string, Candidate[]>,
  evidenceRecordedAt: string | null,
  currentHashes: ArmVideoLibraryOptions["currentHashes"],
  warnings: string[]
): Promise<Candidate> {
  const candidate = { ...baseCandidate(file), kind: "compilation" as const };
  const chaptersPath = path.join(
    path.dirname(file.absolutePath),
    `${path.basename(file.absolutePath, ".mp4")}.chapters.json`
  );
  try {
    const parsed = parseCompilation(await readJson(chaptersPath), file);
    if (!parsed) {
      warnings.push(`Invalid compilation receipt for ${file.relativePath}`);
      return candidate;
    }
    const constituents = parsed.chapters.map((chapter) => {
      const matches = candidatesByHash.get(chapter.sourceHash) ?? [];
      return matches.find((item) => item.verified && item.outcome !== "unverified");
    });
    const allVerified =
      constituents.every(Boolean) &&
      constituents.every((item, index) =>
        item?.outcome === (parsed.chapters[index].passed ? "passed" : "failed")
      );
    if (!allVerified) {
      warnings.push(`Unverified compilation constituents for ${file.relativePath}`);
      return {
        ...candidate,
        durationSeconds: parsed.durationSeconds,
        evidencePath: slashPath(path.relative(artifactRoot, chaptersPath)),
      };
    }
    const verified = constituents as Candidate[];
    const caseIds = [...new Set(verified.flatMap((item) => item.caseIds))];
    const outcomes = new Set(verified.map((item) => item.outcome));
    const provenances = new Set(verified.map((item) => item.provenance));
    const hashes = await batchSourceHashes(path.dirname(file.absolutePath));
    const ownProvenance = provenanceFor(hashes, currentHashes);
    const provenance =
      ownProvenance === "current" && provenances.size === 1 && provenances.has("current")
        ? "current"
        : ownProvenance === "unverified" || provenances.has("unverified")
          ? "unverified"
          : "historical";
    const outcome =
      outcomes.size === 1
        ? [...outcomes][0] as "passed" | "failed"
        : "mixed";
    return {
      ...candidate,
      id: `${candidate.batch}-${path.basename(file.relativePath, ".mp4")}`,
      title: displayTitle(caseIds, null, null, path.basename(file.relativePath)),
      caseIds,
      provenance,
      outcome,
      durationSeconds: parsed.durationSeconds,
      failedGates: [...new Set(verified.flatMap((item) => item.failedGates))].sort(),
      selectionReasons: [],
      evidenceRecordedAt,
      receiptRecordedAt: null,
      evidencePath: slashPath(path.relative(artifactRoot, chaptersPath)),
      verified: provenance !== "unverified",
      source: "chapters",
    };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
      warnings.push(`Malformed compilation receipt for ${file.relativePath}`);
    }
    return candidate;
  }
}

async function batchSourceHashes(
  directory: string
): Promise<{ environment: string; pipeline: string } | null> {
  for (const name of ["protocol.json", "summary.json"]) {
    try {
      const raw = await readJson(
        path.join(/* turbopackIgnore: true */ directory, name)
      );
      if (isRecord(raw)) {
        const hashes = sourceHashes(raw.source_hashes);
        if (hashes) return hashes;
      }
    } catch {
      // Optional metadata is ignored unless it is needed to verify a claim.
    }
  }
  return null;
}

async function hasCompilationReceipt(file: FileRecord): Promise<boolean> {
  try {
    return (
      await stat(
        path.join(
          path.dirname(file.absolutePath),
          `${path.basename(file.absolutePath, ".mp4")}.chapters.json`
        )
      )
    ).isFile();
  } catch {
    return false;
  }
}

function candidateRank(candidate: Candidate): number {
  const provenance =
    candidate.provenance === "current"
      ? 300
      : candidate.provenance === "historical"
        ? 200
        : 0;
  const source =
    candidate.source === "manifest"
      ? 30
      : candidate.source === "chapters"
        ? 20
        : candidate.source === "receipt"
          ? 10
          : 0;
  return provenance + source;
}

function videoUrl(relativePath: string): string {
  return `/api/arm/media?path=${encodeURIComponent(relativePath)}`;
}

function mergeGroup(group: Candidate[]): ArmVideo {
  const sorted = [...group].sort((left, right) =>
    candidateRank(right) - candidateRank(left) ||
    left.file.relativePath.localeCompare(right.file.relativePath)
  );
  const primary = sorted[0];
  const aliases = sorted
    .slice(1)
    .map((item) => item.file.relativePath)
    .sort();
  return {
    id: primary.id,
    title: primary.title,
    caseIds: primary.caseIds,
    batch: primary.batch,
    path: primary.file.relativePath,
    aliases,
    kind: primary.kind,
    provenance: primary.provenance,
    outcome: primary.outcome,
    trainingSeed: primary.trainingSeed,
    evaluationSeed: primary.evaluationSeed,
    durationSeconds: primary.durationSeconds,
    failedGates: primary.failedGates,
    selectionReasons: primary.selectionReasons,
    videoHash: primary.file.hash,
    evidenceRecordedAt: primary.evidenceRecordedAt,
    receiptRecordedAt: primary.receiptRecordedAt,
    videoUrl: videoUrl(primary.file.relativePath),
    evidenceUrl: primary.evidencePath ? videoUrl(primary.evidencePath) : null,
  };
}

function ensureUniqueIds(videos: ArmVideo[]): ArmVideo[] {
  const counts = new Map<string, number>();
  for (const video of videos) {
    counts.set(video.id, (counts.get(video.id) ?? 0) + 1);
  }
  return videos.map((video) => {
    if ((counts.get(video.id) ?? 0) < 2) return video;
    const suffix = createHash("sha256").update(video.path).digest("hex").slice(0, 8);
    return { ...video, id: `${video.batch}:${video.id}:${suffix}` };
  });
}

export async function listArmVideoLibrary(
  options: ArmVideoLibraryOptions = {}
): Promise<ArmVideoLibrary> {
  const { workspaceRoot, artifactRoot } = roots(options);
  const warnings: string[] = [];
  const [files, hashes] = await Promise.all([
    scanFiles(artifactRoot, warnings),
    currentSourceHashes(workspaceRoot, options.currentHashes),
  ]);
  if (!hashes) warnings.push("Current Arm environment or pipeline hashes are unavailable.");

  const catalogRoot = files.resolvedRoot;
  const evidence = await evidenceEntries(files.jsonFiles, catalogRoot, warnings);
  const compilationFlags = await Promise.all(files.videos.map(hasCompilationReceipt));
  const episodeFiles = files.videos.filter((_, index) => !compilationFlags[index]);
  const episodeCandidates = await Promise.all(
    episodeFiles.map((file) =>
      episodeCandidate(
        file,
        catalogRoot,
        evidence.entries.get(file.relativePath),
        hashes,
        warnings
      )
    )
  );
  const candidatesByHash = Map.groupBy(episodeCandidates, (item) => item.file.hash);
  const compilationFiles = files.videos.filter((_, index) => compilationFlags[index]);
  const compilationCandidates = await Promise.all(
    compilationFiles.map((file) =>
      compilationCandidate(
        file,
        catalogRoot,
        candidatesByHash,
        evidence.batchRecordedAt.get(path.dirname(file.absolutePath)) ?? null,
        hashes,
        warnings
      )
    )
  );
  const allCandidates = [...episodeCandidates, ...compilationCandidates];
  const videos = ensureUniqueIds(
    [...Map.groupBy(allCandidates, (item) => item.file.hash).values()]
    .map(mergeGroup)
    .sort((left, right) =>
      left.batch.localeCompare(right.batch) ||
      left.kind.localeCompare(right.kind) ||
      left.title.localeCompare(right.title) ||
      left.path.localeCompare(right.path)
    )
  );

  return {
    videos,
    totalFiles: files.videos.length,
    uniqueVideos: videos.length,
    hardwareEnabled: false,
    warnings: [...new Set(warnings)].sort(),
  };
}

async function assertNoSymlink(root: string, relativePath: string): Promise<void> {
  let current = root;
  for (const part of relativePath.split("/")) {
    current = path.join(current, part);
    if ((await lstat(current)).isSymbolicLink()) {
      throw new Error("Symlink artifacts are not allowed.");
    }
  }
}

async function resolveMediaFile(
  relativePath: string,
  options: ArmVideoLibraryOptions
): Promise<MediaFile> {
  if (!isSafeRelativePath(relativePath)) throw new Error("Invalid artifact path.");
  const { artifactRoot } = roots(options);
  const resolvedRoot = await realpath(
    /* turbopackIgnore: true */ artifactRoot
  );
  await assertNoSymlink(resolvedRoot, relativePath);
  const candidate = path.join(resolvedRoot, ...relativePath.split("/"));
  const resolved = await realpath(candidate);
  if (!isWithin(resolvedRoot, resolved) || !(await stat(resolved)).isFile()) {
    throw new Error("Artifact not found.");
  }

  const extension = path.extname(relativePath).toLowerCase();
  if (extension === ".mp4") {
    return {
      absolutePath: resolved,
      relativePath,
      size: (await stat(resolved)).size,
      contentType: "video/mp4",
    };
  }
  if (extension !== ".json") throw new Error("Artifact type is not allowed.");

  const library = await listArmVideoLibrary(options);
  const evidencePaths = new Set(
    library.videos
      .map((video) => video.evidenceUrl)
      .filter((value): value is string => value !== null)
      .map((url) => new URL(url, "http://local").searchParams.get("path"))
      .filter((value): value is string => value !== null)
  );
  if (!evidencePaths.has(relativePath)) {
    throw new Error("JSON artifact is not an authorized receipt.");
  }
  return {
    absolutePath: resolved,
    relativePath,
    size: (await stat(resolved)).size,
    contentType: "application/json",
  };
}

function rangeFor(value: string, size: number): { start: number; end: number } | null {
  const match = /^bytes=(\d*)-(\d*)$/.exec(value);
  if (!match || (!match[1] && !match[2]) || size < 1) return null;
  if (!match[1]) {
    const suffix = Number(match[2]);
    if (!Number.isSafeInteger(suffix) || suffix <= 0) return null;
    return { start: Math.max(0, size - suffix), end: size - 1 };
  }
  const start = Number(match[1]);
  const end = match[2] ? Math.min(Number(match[2]), size - 1) : size - 1;
  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(end) ||
    start < 0 ||
    start >= size ||
    end < start
  ) {
    return null;
  }
  return { start, end };
}

export async function serveArmMedia(
  request: Request,
  options: ArmVideoLibraryOptions = {}
): Promise<Response> {
  const requestedPath = new URL(request.url).searchParams.get("path");
  if (!requestedPath) {
    return Response.json({ error: "An artifact path is required." }, { status: 400 });
  }
  try {
    const media = await resolveMediaFile(requestedPath, options);
    const headers = new Headers({
      "Accept-Ranges": media.contentType === "video/mp4" ? "bytes" : "none",
      "Cache-Control": "no-store",
      "Content-Disposition": `inline; filename="${path.basename(media.relativePath)}"`,
      "Content-Type": media.contentType,
    });
    const rangeHeader = request.headers.get("range");
    if (rangeHeader && media.contentType === "video/mp4") {
      const range = rangeFor(rangeHeader, media.size);
      if (!range) {
        headers.set("Content-Range", `bytes */${media.size}`);
        return new Response(null, { status: 416, headers });
      }
      const length = range.end - range.start + 1;
      headers.set("Content-Length", String(length));
      headers.set("Content-Range", `bytes ${range.start}-${range.end}/${media.size}`);
      if (request.method === "HEAD") {
        return new Response(null, { status: 206, headers });
      }
      const handle = await open(media.absolutePath, "r");
      try {
        const data = Buffer.alloc(length);
        await handle.read(data, 0, length, range.start);
        return new Response(new Uint8Array(data), { status: 206, headers });
      } finally {
        await handle.close();
      }
    }
    headers.set("Content-Length", String(media.size));
    if (request.method === "HEAD") return new Response(null, { headers });
    return new Response(new Uint8Array(await readFile(media.absolutePath)), { headers });
  } catch {
    return Response.json({ error: "Artifact not found." }, { status: 404 });
  }
}
