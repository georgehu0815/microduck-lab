import { ARM_CASE_IDS, type ArmCaseId } from "./arm-contract";
import type { ArmVideo } from "./arm-videos";

const CASE_ORDER = new Map(ARM_CASE_IDS.map((caseId, index) => [caseId, index]));

export function armVideoRecordedAt(video: ArmVideo): string | null {
  for (const value of [video.receiptRecordedAt, video.evidenceRecordedAt]) {
    if (value && Number.isFinite(Date.parse(value))) return value;
  }
  return null;
}

export function compareArmVideosNewestFirst(
  left: ArmVideo,
  right: ArmVideo
): number {
  const leftTime = armVideoRecordedAt(left);
  const rightTime = armVideoRecordedAt(right);
  if (leftTime && rightTime) {
    const timestampOrder = Date.parse(rightTime) - Date.parse(leftTime);
    if (timestampOrder !== 0) return timestampOrder;
  } else if (leftTime || rightTime) {
    return leftTime ? -1 : 1;
  }

  return (
    (CASE_ORDER.get(left.caseIds[0]) ?? ARM_CASE_IDS.length) -
      (CASE_ORDER.get(right.caseIds[0]) ?? ARM_CASE_IDS.length) ||
    left.batch.localeCompare(right.batch) ||
    left.id.localeCompare(right.id) ||
    left.path.localeCompare(right.path) ||
    left.videoHash.localeCompare(right.videoHash)
  );
}

export function selectLatestCurrentSuccesses(videos: ArmVideo[]): ArmVideo[] {
  const latestByCase = new Map<ArmCaseId, ArmVideo>();
  for (const video of [...videos].sort(compareArmVideosNewestFirst)) {
    if (
      video.kind !== "episode" ||
      video.provenance !== "current" ||
      video.outcome !== "passed" ||
      !armVideoRecordedAt(video)
    ) {
      continue;
    }
    for (const caseId of video.caseIds) {
      if (!latestByCase.has(caseId)) latestByCase.set(caseId, video);
    }
  }
  return [...latestByCase.values()].sort(compareArmVideosNewestFirst);
}

export function missingCurrentSuccessCaseIds(
  videos: ArmVideo[]
): ArmCaseId[] {
  const selectedCases = new Set(
    selectLatestCurrentSuccesses(videos).flatMap((video) => video.caseIds)
  );
  return ARM_CASE_IDS.filter((caseId) => !selectedCases.has(caseId));
}
