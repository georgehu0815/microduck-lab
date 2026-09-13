import { readdir, realpath, stat } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";
import { EXPERIMENTS } from "@/lib/experiments";
import { evaluationVerdict } from "@/lib/evaluation";
import { artifactPath, sanitizeRunName, snapshot } from "@/lib/rlx-job";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const groups = await Promise.all(EXPERIMENTS.map(async (experiment) => {
      const root = path.dirname(path.dirname(artifactPath(experiment.id, "catalog", "checkpoint")));
      const entries = await readdir(root, { withFileTypes: true }).catch(() => []);
      const runs = await Promise.all(entries.filter((entry) => entry.isDirectory()).map(async (entry) => {
        try {
          if (sanitizeRunName(entry.name) !== entry.name) return null;
          const directory = path.join(root, entry.name);
          if (await realpath(directory) !== directory) return null;
          const modified = await stat(artifactPath(experiment.id, entry.name, "evaluation"))
            .catch(() => stat(artifactPath(experiment.id, entry.name, "checkpoint")))
            .catch(() => stat(artifactPath(experiment.id, entry.name, "onnx")));
          const trained = await stat(artifactPath(experiment.id, entry.name, "checkpoint")).catch(() => null);
          const policy = await stat(artifactPath(experiment.id, entry.name, "onnx")).catch(() => null);
          const state = await snapshot(experiment.id, entry.name);
          return {
            experimentId: experiment.id,
            drawingTool: state.drawingTool,
            runName: entry.name,
            modifiedAt: modified.mtime.toISOString(),
            trainedAt: trained?.mtime.toISOString() ?? null,
            policyModifiedAt: policy?.mtime.toISOString() ?? null,
            onnx: state.artifacts.onnx,
            ...evaluationVerdict(state.evaluation, experiment.id),
            video: state.artifacts.renderVideo,
            checkpoint: state.artifacts.checkpoint,
            renderVerified: state.renderVerified,
            renderEvidenceId: state.renderEvidenceId,
          };
        } catch {
          return null;
        }
      }));
      return runs.filter((run) => run !== null);
    }));
    return NextResponse.json({ runs: groups.flat().sort((left, right) => right.modifiedAt.localeCompare(left.modifiedAt)) });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Cannot list saved runs." }, { status: 500 });
  }
}
