export const ARM_CASE_IDS = [
  "arm-reach-v1",
  "arm-pick-place-v1",
  "arm-relocate-v1",
  "arm-carry-v1",
  "arms-handover-v1",
  "arms-co-carry-v1",
] as const;

export type ArmCaseId = (typeof ARM_CASE_IDS)[number];
export type ArmGeomType = "box" | "sphere" | "capsule" | "cylinder";

export interface ArmCaseDefinition {
  id: ArmCaseId;
  title: string;
  summary: string;
  observationDim: 66 | 116;
  actionDim: 6 | 12;
  group: "单臂" | "双臂";
}

export interface ArmGeom {
  name: string;
  type: ArmGeomType;
  size: [number, number, number];
  pos: [number, number, number];
  quat: [number, number, number, number];
  rgba: [number, number, number, number];
}

export interface ArmState {
  case_id: ArmCaseId;
  seed: number;
  step: number;
  time: number;
  contract: string;
  observation_dim: 66 | 116;
  action_dim: 6 | 12;
  controller: string;
  stage: string;
  terminated: boolean;
  truncated: boolean;
  hardware_enabled: false;
  joints: number[];
  geoms: ArmGeom[];
  metrics: Record<string, unknown>;
}

export interface ArmRun {
  case_id: ArmCaseId;
  run_id: string;
  path: string;
  controller: string;
  passed: boolean | null;
  metrics: Record<string, unknown>;
  video_path?: string;
}

export interface ArmRunsResponse {
  runs: ArmRun[];
  hardware_enabled: false;
}

export const ARM_CASES: ArmCaseDefinition[] = [
  {
    id: "arm-reach-v1",
    title: "末端触达",
    summary: "观察坐标、关节与目标误差，完成无载 reach。",
    observationDim: 66,
    actionDim: 6,
    group: "单臂",
  },
  {
    id: "arm-pick-place-v1",
    title: "抓取与放置",
    summary: "真实接触、抬升、移动、释放，不把推入目标算抓取。",
    observationDim: 66,
    actionDim: 6,
    group: "单臂",
  },
  {
    id: "arm-relocate-v1",
    title: "绕障移物",
    summary: "在固定障碍场景中抬高并重新放置物体。",
    observationDim: 66,
    actionDim: 6,
    group: "单臂",
  },
  {
    id: "arm-carry-v1",
    title: "定点搬运",
    summary: "固定底座机械臂依次经过航点；不是鸭子行走搬运。",
    observationDim: 66,
    actionDim: 6,
    group: "单臂",
  },
  {
    id: "arms-handover-v1",
    title: "双臂交接",
    summary: "接收臂建立抓握后，发送臂才释放。",
    observationDim: 116,
    actionDim: 12,
    group: "双臂",
  },
  {
    id: "arms-co-carry-v1",
    title: "协作搬托盘",
    summary: "两臂共同抓持、抬升、搬运与稳定释放。",
    observationDim: 116,
    actionDim: 12,
    group: "双臂",
  },
];

const CASE_SET = new Set<string>(ARM_CASE_IDS);
const GEOM_TYPES = new Set<string>(["box", "sphere", "capsule", "cylinder"]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isFiniteTuple(value: unknown, length: number): value is number[] {
  return Array.isArray(value) &&
    value.length === length &&
    value.every(isFiniteNumber);
}

export function isArmCaseId(value: unknown): value is ArmCaseId {
  return typeof value === "string" && CASE_SET.has(value);
}

export function hasOnlyFiniteJson(
  value: unknown,
  depth = 0,
  seen = new Set<object>()
): boolean {
  if (depth > 12) return false;
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "boolean"
  ) return true;
  if (typeof value === "number") return Number.isFinite(value);
  if (typeof value !== "object" || seen.has(value)) return false;
  seen.add(value);
  const values = Array.isArray(value) ? value : Object.values(value);
  if (values.length > 20_000) return false;
  return values.every((entry) => hasOnlyFiniteJson(entry, depth + 1, seen));
}

export function validateResetBody(value: unknown):
  | { case_id: ArmCaseId; seed: number }
  | null {
  if (!isRecord(value)) return null;
  const keys = Object.keys(value);
  if (
    keys.length !== 2 ||
    !keys.includes("case_id") ||
    !keys.includes("seed") ||
    !isArmCaseId(value.case_id) ||
    !Number.isSafeInteger(value.seed) ||
    (value.seed as number) < 0 ||
    (value.seed as number) > 2_147_483_647
  ) return null;
  return { case_id: value.case_id, seed: value.seed as number };
}

export function validateStepBody(value: unknown):
  | { action: number[] }
  | null {
  if (!isRecord(value) || Object.keys(value).length !== 1) return null;
  if (
    !Array.isArray(value.action) ||
    (value.action.length !== 6 && value.action.length !== 12) ||
    !value.action.every((entry) => isFiniteNumber(entry) && Math.abs(entry) <= 1)
  ) return null;
  return { action: [...value.action] };
}

export function validateTeacherBody(value: unknown):
  | { ticks: number }
  | null {
  if (!isRecord(value) || Object.keys(value).length !== 1) return null;
  if (
    !Number.isSafeInteger(value.ticks) ||
    (value.ticks as number) < 1 ||
    (value.ticks as number) > 50
  ) return null;
  return { ticks: value.ticks as number };
}

export function isSafeArtifactPath(value: unknown): value is string {
  if (
    typeof value !== "string" ||
    value.length < 1 ||
    value.length > 512 ||
    value.includes("\0") ||
    value.includes("\\") ||
    value.startsWith("/") ||
    /^[a-z][a-z0-9+.-]*:/i.test(value)
  ) return false;
  const parts = value.split("/");
  return parts.every((part) =>
    part.length > 0 &&
    part !== "." &&
    part !== ".." &&
    !part.startsWith("~")
  );
}

export function parseArmState(value: unknown): ArmState | null {
  if (!isRecord(value) || !hasOnlyFiniteJson(value)) return null;
  const actionDim = value.action_dim;
  const observationDim = value.observation_dim;
  if (
    !isArmCaseId(value.case_id) ||
    !Number.isSafeInteger(value.seed) ||
    !Number.isSafeInteger(value.step) ||
    !isFiniteNumber(value.time) ||
    typeof value.contract !== "string" ||
    (observationDim !== 66 && observationDim !== 116) ||
    (actionDim !== 6 && actionDim !== 12) ||
    typeof value.controller !== "string" ||
    typeof value.stage !== "string" ||
    typeof value.terminated !== "boolean" ||
    typeof value.truncated !== "boolean" ||
    value.hardware_enabled !== false ||
    !Array.isArray(value.joints) ||
    value.joints.length !== actionDim ||
    !value.joints.every(isFiniteNumber) ||
    !Array.isArray(value.geoms) ||
    !isRecord(value.metrics)
  ) return null;

  const definition = ARM_CASES.find((entry) => entry.id === value.case_id);
  if (
    !definition ||
    definition.actionDim !== actionDim ||
    definition.observationDim !== observationDim
  ) return null;

  const geoms: ArmGeom[] = [];
  for (const geom of value.geoms) {
    if (
      !isRecord(geom) ||
      typeof geom.name !== "string" ||
      typeof geom.type !== "string" ||
      !GEOM_TYPES.has(geom.type) ||
      !isFiniteTuple(geom.size, 3) ||
      geom.size.some((entry) => entry < 0) ||
      !isFiniteTuple(geom.pos, 3) ||
      !isFiniteTuple(geom.quat, 4) ||
      !isFiniteTuple(geom.rgba, 4)
    ) return null;
    geoms.push({
      name: geom.name,
      type: geom.type as ArmGeomType,
      size: geom.size as [number, number, number],
      pos: geom.pos as [number, number, number],
      quat: geom.quat as [number, number, number, number],
      rgba: geom.rgba as [number, number, number, number],
    });
  }

  return {
    case_id: value.case_id,
    seed: value.seed as number,
    step: value.step as number,
    time: value.time,
    contract: value.contract,
    observation_dim: observationDim,
    action_dim: actionDim,
    controller: value.controller,
    stage: value.stage,
    terminated: value.terminated,
    truncated: value.truncated,
    hardware_enabled: false,
    joints: [...value.joints],
    geoms,
    metrics: value.metrics,
  };
}

export function parseArmRuns(value: unknown): ArmRunsResponse | null {
  if (
    !isRecord(value) ||
    value.hardware_enabled !== false ||
    !Array.isArray(value.runs) ||
    value.runs.length > 2_000 ||
    !hasOnlyFiniteJson(value)
  ) return null;
  const runs: ArmRun[] = [];
  for (const run of value.runs) {
    if (
      !isRecord(run) ||
      !isArmCaseId(run.case_id) ||
      typeof run.run_id !== "string" ||
      typeof run.path !== "string" ||
      typeof run.controller !== "string" ||
      (run.passed !== true && run.passed !== false && run.passed !== null) ||
      !isRecord(run.metrics) ||
      (run.video_path !== undefined && !isSafeArtifactPath(run.video_path))
    ) return null;
    runs.push({
      case_id: run.case_id,
      run_id: run.run_id,
      path: run.path,
      controller: run.controller,
      passed: run.passed,
      metrics: run.metrics,
      ...(run.video_path ? { video_path: run.video_path } : {}),
    });
  }
  return { runs, hardware_enabled: false };
}
