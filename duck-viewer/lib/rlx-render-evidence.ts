import { createHash } from "node:crypto";
import {
  mkdirSync,
  readFileSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";

export type RenderEvidencePreparation = {
  sourceFiles: string[];
  source: string;
  recipeKey: string;
  clipPath: string | null;
  externalFiles?: string[];
};

export type RenderRecipeOptions = {
  backflip_protocol_version?: string;
  stand_policy_sha256?: string;
};

export type RenderEvidenceContext = RenderEvidencePreparation & {
  sourceFilesSha256: Record<string, string>;
  sourceSha256: string;
  clipSha256: string | null;
  externalFiles: string[];
  externalFilesSha256: Record<string, string>;
};

export type RenderEvidenceMedia = {
  path: string;
  sha256: string;
  size: number;
};

export type RenderEvidenceReceipt = {
  version: 1;
  sourceFiles: string[];
  source: string;
  sourceFilesSha256: Record<string, string>;
  sourceSha256: string;
  recipeKey: string;
  clipPath: string | null;
  clipSha256: string | null;
  externalFiles?: string[];
  externalFilesSha256?: Record<string, string>;
  recipeOptions?: RenderRecipeOptions;
  video: RenderEvidenceMedia;
  sheet: RenderEvidenceMedia;
};

export type RenderEvidenceOutputs = {
  video: string;
  sheet: string;
  receipt: string;
};

export type RenderEvidenceValidation = RenderEvidencePreparation & {
  recipeOptions?: RenderRecipeOptions;
  video: string;
  sheet: string;
};

type HashCacheEntry = {
  identity: string;
  sha256: string;
};

const hashCache = new Map<string, HashCacheEntry>();

function fileIdentity(file: string): { identity: string; size: number } {
  const fileStat = statSync(file, { bigint: true });
  if (!fileStat.isFile()) throw new Error(`Expected a file: ${file}`);
  return {
    identity: [
      fileStat.dev,
      fileStat.ino,
      fileStat.size,
      fileStat.mtimeNs,
      fileStat.ctimeNs,
    ].join(":"),
    size: Number(fileStat.size),
  };
}

function hashFile(file: string): { sha256: string; size: number } {
  const before = fileIdentity(file);
  const cached = hashCache.get(file);
  if (cached?.identity === before.identity) {
    return { sha256: cached.sha256, size: before.size };
  }
  const sha256 = createHash("sha256").update(readFileSync(file)).digest("hex");
  const after = fileIdentity(file);
  if (before.identity !== after.identity) {
    throw new Error(`File changed while hashing: ${file}`);
  }
  hashCache.set(file, { identity: after.identity, sha256 });
  return { sha256, size: after.size };
}

function uniqueFiles(files: string[]): string[] {
  return [...new Set(files)].sort();
}

function captureInputs(input: RenderEvidencePreparation): RenderEvidenceContext {
  const sourceFiles = uniqueFiles(input.sourceFiles);
  const externalFiles = uniqueFiles(input.externalFiles ?? []);
  if (!sourceFiles.includes(input.source)) {
    throw new Error("Render source must be included in sourceFiles.");
  }
  const sourceFilesSha256 = Object.fromEntries(
    sourceFiles.map((file) => [file, hashFile(file).sha256])
  );
  return {
    ...input,
    sourceFiles,
    sourceFilesSha256,
    sourceSha256: sourceFilesSha256[input.source],
    clipSha256: input.clipPath ? hashFile(input.clipPath).sha256 : null,
    externalFiles,
    externalFilesSha256: Object.fromEntries(
      externalFiles.map((file) => [file, hashFile(file).sha256])
    ),
  };
}

function inputsMatch(context: RenderEvidenceContext | RenderEvidenceReceipt): boolean {
  try {
    const externalFiles = uniqueFiles(context.externalFiles ?? []);
    const externalFilesSha256 = context.externalFilesSha256 ?? {};
    if (
      context.sourceFiles.length !==
        Object.keys(context.sourceFilesSha256).length ||
      context.sourceFilesSha256[context.source] !== context.sourceSha256 ||
      externalFiles.length !== Object.keys(externalFilesSha256).length
    ) {
      return false;
    }
    for (const file of context.sourceFiles) {
      if (hashFile(file).sha256 !== context.sourceFilesSha256[file]) {
        return false;
      }
    }
    for (const file of externalFiles) {
      if (hashFile(file).sha256 !== externalFilesSha256[file]) {
        return false;
      }
    }
    return context.clipPath
      ? hashFile(context.clipPath).sha256 === context.clipSha256
      : context.clipSha256 === null;
  } catch {
    return false;
  }
}

function atomicWriteReceipt(
  receiptPath: string,
  receipt: RenderEvidenceReceipt
): void {
  mkdirSync(path.dirname(receiptPath), { recursive: true });
  const temporary = `${receiptPath}.${process.pid}.${Math.random()
    .toString(16)
    .slice(2)}.tmp`;
  try {
    writeFileSync(temporary, `${JSON.stringify(receipt, null, 2)}\n`, {
      flag: "wx",
    });
    renameSync(temporary, receiptPath);
  } catch (error) {
    try {
      unlinkSync(temporary);
    } catch {}
    throw error;
  }
}

export function prepareRenderEvidence(
  input: RenderEvidencePreparation
): RenderEvidenceContext {
  return captureInputs(input);
}

export function finalizeRenderEvidence(
  context: RenderEvidenceContext,
  output: RenderEvidenceOutputs,
  recipeOptions?: RenderRecipeOptions
): RenderEvidenceReceipt | null {
  try {
    if (!inputsMatch(context)) return null;
    const videoHash = hashFile(output.video);
    const sheetHash = hashFile(output.sheet);
    if (!inputsMatch(context)) return null;
    const receipt: RenderEvidenceReceipt = {
      version: 1,
      sourceFiles: [...context.sourceFiles],
      source: context.source,
      sourceFilesSha256: { ...context.sourceFilesSha256 },
      sourceSha256: context.sourceSha256,
      recipeKey: context.recipeKey,
      clipPath: context.clipPath,
      clipSha256: context.clipSha256,
      externalFiles: [...context.externalFiles],
      externalFilesSha256: { ...context.externalFilesSha256 },
      ...(recipeOptions ? { recipeOptions: { ...recipeOptions } } : {}),
      video: { path: output.video, ...videoHash },
      sheet: { path: output.sheet, ...sheetHash },
    };
    atomicWriteReceipt(output.receipt, receipt);
    return receipt;
  } catch {
    return null;
  }
}

function isReceipt(value: unknown): value is RenderEvidenceReceipt {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const receipt = value as Partial<RenderEvidenceReceipt>;
  return (
    receipt.version === 1 &&
    Array.isArray(receipt.sourceFiles) &&
    receipt.sourceFiles.every((file) => typeof file === "string") &&
    typeof receipt.source === "string" &&
    Boolean(receipt.sourceFilesSha256) &&
    typeof receipt.sourceFilesSha256 === "object" &&
    !Array.isArray(receipt.sourceFilesSha256) &&
    typeof receipt.sourceSha256 === "string" &&
    typeof receipt.recipeKey === "string" &&
    (receipt.clipPath === null || typeof receipt.clipPath === "string") &&
    (receipt.clipSha256 === null || typeof receipt.clipSha256 === "string") &&
    (receipt.externalFiles === undefined ||
      (Array.isArray(receipt.externalFiles) &&
        receipt.externalFiles.every((file) => typeof file === "string"))) &&
    (receipt.externalFilesSha256 === undefined ||
      (Boolean(receipt.externalFilesSha256) &&
        typeof receipt.externalFilesSha256 === "object" &&
        !Array.isArray(receipt.externalFilesSha256))) &&
    (receipt.recipeOptions === undefined ||
      (Boolean(receipt.recipeOptions) &&
        typeof receipt.recipeOptions === "object" &&
        !Array.isArray(receipt.recipeOptions) &&
        (receipt.recipeOptions.backflip_protocol_version === undefined ||
          typeof receipt.recipeOptions.backflip_protocol_version === "string") &&
        (receipt.recipeOptions.stand_policy_sha256 === undefined ||
          typeof receipt.recipeOptions.stand_policy_sha256 === "string"))) &&
    isMedia(receipt.video) &&
    isMedia(receipt.sheet)
  );
}

function isMedia(value: unknown): value is RenderEvidenceMedia {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const media = value as Partial<RenderEvidenceMedia>;
  return (
    typeof media.path === "string" &&
    typeof media.sha256 === "string" &&
    typeof media.size === "number" &&
    Number.isSafeInteger(media.size) &&
    media.size >= 0
  );
}

function sameFiles(left: string[], right: string[]): boolean {
  return (
    left.length === right.length &&
    left.every((file, index) => file === right[index])
  );
}

export function readRenderEvidence(
  receiptPath: string,
  expected: RenderEvidenceValidation
): RenderEvidenceReceipt | null {
  try {
    const parsed: unknown = JSON.parse(readFileSync(receiptPath, "utf8"));
    if (!isReceipt(parsed)) return null;
    const sourceFiles = uniqueFiles(expected.sourceFiles);
    const externalFiles = uniqueFiles(expected.externalFiles ?? []);
    if (
      !sourceFiles.includes(expected.source) ||
      !sameFiles(parsed.sourceFiles, sourceFiles) ||
      !sameFiles(uniqueFiles(parsed.externalFiles ?? []), externalFiles) ||
      parsed.source !== expected.source ||
      parsed.recipeKey !== expected.recipeKey ||
      parsed.clipPath !== expected.clipPath ||
      JSON.stringify(parsed.recipeOptions ?? {}) !==
        JSON.stringify(expected.recipeOptions ?? {}) ||
      parsed.video.path !== expected.video ||
      parsed.sheet.path !== expected.sheet
    ) {
      return null;
    }
    if (!inputsMatch(parsed)) return null;
    const video = hashFile(expected.video);
    const sheet = hashFile(expected.sheet);
    if (
      video.sha256 !== parsed.video.sha256 ||
      video.size !== parsed.video.size ||
      sheet.sha256 !== parsed.sheet.sha256 ||
      sheet.size !== parsed.sheet.size
    ) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}
