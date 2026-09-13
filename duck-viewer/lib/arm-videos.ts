import type { ArmCaseId } from "./arm-contract";

export interface ArmVideo {
  id: string;
  title: string;
  caseIds: ArmCaseId[];
  batch: string;
  path: string;
  aliases: string[];
  kind: "episode" | "compilation";
  provenance: "current" | "historical" | "unverified";
  outcome: "passed" | "failed" | "mixed" | "unverified";
  trainingSeed: number | null;
  evaluationSeed: number | null;
  durationSeconds: number | null;
  failedGates: string[];
  selectionReasons: string[];
  videoHash: string;
  evidenceRecordedAt: string | null;
  receiptRecordedAt: string | null;
  videoUrl: string;
  evidenceUrl: string | null;
}

export interface ArmVideoLibrary {
  videos: ArmVideo[];
  totalFiles: number;
  uniqueVideos: number;
  hardwareEnabled: false;
  warnings: string[];
}
