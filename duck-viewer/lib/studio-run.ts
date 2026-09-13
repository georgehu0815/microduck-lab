import type { DrawingTool, ExperimentId } from "./experiments";
import { IS_STATIC_EXPORT, publicAssetUrl } from "./static-assets";

export interface SavedRun {
  experimentId: ExperimentId;
  drawingTool?: DrawingTool;
  runName: string;
  taskPassed: boolean;
  skillAssessed: boolean;
  video: boolean;
  checkpoint?: boolean;
  onnx?: boolean;
  policyModifiedAt?: string | null;
  modifiedAt?: string;
  trainedAt?: string | null;
  renderVerified?: boolean;
  renderEvidenceId?: string | null;
  balanceOnly?: boolean;
}

export async function fetchSavedRuns(signal?: AbortSignal): Promise<SavedRun[]> {
  const catalogUrl = IS_STATIC_EXPORT
    ? publicAssetUrl("/static/studio-runs/catalog.json")
    : "/api/rlx/runs";
  const response = await fetch(catalogUrl, { cache: "no-store", signal });
  if (!response.ok) throw new Error("Saved run catalog unavailable.");
  const data = await response.json() as { runs: SavedRun[] };
  const artifactTime = (run: SavedRun) => Math.max(
    Date.parse(run.trainedAt ?? "") || 0,
    Date.parse(run.policyModifiedAt ?? "") || 0,
  ) || Date.parse(run.modifiedAt ?? "") || 0;
  return data.runs.sort((left, right) => artifactTime(right) - artifactTime(left)
    || left.runName.localeCompare(right.runName));
}

export function latestVerifiedRun(runs: readonly SavedRun[], experimentId: ExperimentId): SavedRun | undefined {
  const trainedTime = (run: SavedRun) => Date.parse(
    run.trainedAt ?? run.policyModifiedAt ?? run.modifiedAt ?? ""
  );
  return runs.filter((run) => {
    const previewAccepted = run.taskPassed ||
      (experimentId === "basketball" && run.balanceOnly === true);
    const sourceAvailable = run.checkpoint ||
      (experimentId === "basketball" && run.onnx);
    return run.experimentId === experimentId && previewAccepted &&
      sourceAvailable && run.video && run.renderVerified &&
      run.renderEvidenceId && Number.isFinite(trainedTime(run));
  }).sort((left, right) => trainedTime(right) - trainedTime(left) ||
    left.runName.localeCompare(right.runName))[0];
}

export function latestDiagnosticRun(runs: readonly SavedRun[], experimentId: ExperimentId): SavedRun | undefined {
  if (experimentId !== "bridge" && experimentId !== "drawing") return undefined;
  const trainedTime = (run: SavedRun) => Date.parse(
    run.trainedAt ?? run.policyModifiedAt ?? run.modifiedAt ?? ""
  );
  return runs.filter((run) =>
    run.experimentId === experimentId &&
    !run.taskPassed &&
    run.skillAssessed &&
    Boolean(run.checkpoint || run.onnx) &&
    run.video &&
    run.renderVerified &&
    Boolean(run.renderEvidenceId) &&
    Number.isFinite(trainedTime(run))
  ).sort((left, right) => trainedTime(right) - trainedTime(left) ||
    left.runName.localeCompare(right.runName))[0];
}

export function latestReviewRun(runs: readonly SavedRun[], experimentId: ExperimentId): SavedRun | undefined {
  const verified = latestVerifiedRun(runs, experimentId);
  if (verified) return verified;
  const diagnostic = latestDiagnosticRun(runs, experimentId);
  if (diagnostic) return diagnostic;
  const matching = runs.filter((run) => run.experimentId === experimentId);
  return matching.find((run) => run.taskPassed)
    ?? matching.find((run) => experimentId === "basketball" && run.balanceOnly === true)
    ?? matching[0];
}

export function availableProfileRunName(
  runName: string,
  profile: "smoke" | "full",
  reservedNames: readonly string[],
): string {
  const canonical = (name: string) => name.trim().toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48);
  const reserved = new Set(reservedNames.map(canonical));
  const current = canonical(runName);
  if (!reserved.has(current)) return runName;
  const base = current.replace(/-(smoke|full)(-\d+)?$/, "") || "run";
  for (let sequence = 1; ; sequence += 1) {
    const suffix = `-${profile}${sequence === 1 ? "" : `-${sequence}`}`;
    const candidate = `${base.slice(0, 48 - suffix.length)}${suffix}`;
    if (!reserved.has(candidate)) return candidate;
  }
}
