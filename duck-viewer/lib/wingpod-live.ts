export type WingPodLiveGeomType = "box" | "sphere" | "ellipsoid" | "capsule" | "cylinder";

export interface WingPodLiveGeom {
  name: string;
  type: WingPodLiveGeomType;
  size: [number, number, number];
  pos: [number, number, number];
  quat: [number, number, number, number];
  rgba: [number, number, number, number];
}

export interface WingPodLiveState {
  case_id: "wingpod-tennis-v1";
  seed: number;
  step: number;
  time: number;
  contract: "wingpod-tennis-live-v1";
  observation_dim: number;
  action_dim: 15;
  controller: "authored_ik_fsm_teacher_not_ppo";
  stage: string;
  terminated: boolean;
  truncated: boolean;
  hardware_enabled: false;
  joints: number[];
  geoms: WingPodLiveGeom[];
  metrics: {
    simulation_only: true;
    trained_policy: false;
    vision_control: false;
    task_success: boolean;
    failure_reason: string | null;
    controller_stage: string;
    ball_position_m: number[];
    ball_inside_bin: boolean;
    tcp_ball_distance_m: number;
    carry_distance_m: number;
  };
}

const GEOM_TYPES = new Set<WingPodLiveGeomType>([
  "box",
  "sphere",
  "ellipsoid",
  "capsule",
  "cylinder",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isFiniteTuple(value: unknown, length: number): value is number[] {
  return Array.isArray(value) && value.length === length && value.every(isFiniteNumber);
}

export function validateWingPodMutation(
  path: string,
  value: unknown,
): { seed: number } | { ticks: number } | null {
  if (!isRecord(value) || Object.keys(value).length !== 1) return null;
  if (path === "wingpod-reset") {
    return Number.isSafeInteger(value.seed) && Number(value.seed) >= 0 && Number(value.seed) <= 2_147_483_647
      ? { seed: Number(value.seed) }
      : null;
  }
  if (path === "wingpod-teacher") {
    return Number.isSafeInteger(value.ticks) && Number(value.ticks) >= 1 && Number(value.ticks) <= 10
      ? { ticks: Number(value.ticks) }
      : null;
  }
  return null;
}

export function parseWingPodLiveState(value: unknown): WingPodLiveState | null {
  if (
    !isRecord(value)
    || value.case_id !== "wingpod-tennis-v1"
    || value.contract !== "wingpod-tennis-live-v1"
    || value.action_dim !== 15
    || value.controller !== "authored_ik_fsm_teacher_not_ppo"
    || value.hardware_enabled !== false
    || !Number.isSafeInteger(value.seed)
    || !Number.isSafeInteger(value.step)
    || !Number.isSafeInteger(value.observation_dim)
    || !isFiniteNumber(value.time)
    || typeof value.stage !== "string"
    || typeof value.terminated !== "boolean"
    || typeof value.truncated !== "boolean"
    || !Array.isArray(value.joints)
    || value.joints.length !== 15
    || !value.joints.every(isFiniteNumber)
    || !Array.isArray(value.geoms)
    || !isRecord(value.metrics)
  ) return null;

  const geoms: WingPodLiveGeom[] = [];
  for (const geom of value.geoms) {
    if (
      !isRecord(geom)
      || typeof geom.name !== "string"
      || !GEOM_TYPES.has(geom.type as WingPodLiveGeomType)
      || !isFiniteTuple(geom.size, 3)
      || geom.size.some((entry) => entry < 0)
      || !isFiniteTuple(geom.pos, 3)
      || !isFiniteTuple(geom.quat, 4)
      || !isFiniteTuple(geom.rgba, 4)
    ) return null;
    geoms.push({
      name: geom.name,
      type: geom.type as WingPodLiveGeomType,
      size: geom.size as [number, number, number],
      pos: geom.pos as [number, number, number],
      quat: geom.quat as [number, number, number, number],
      rgba: geom.rgba as [number, number, number, number],
    });
  }

  const metrics = value.metrics;
  if (
    metrics.simulation_only !== true
    || metrics.trained_policy !== false
    || metrics.vision_control !== false
    || typeof metrics.task_success !== "boolean"
    || !(metrics.failure_reason === null || typeof metrics.failure_reason === "string")
    || typeof metrics.controller_stage !== "string"
    || !isFiniteTuple(metrics.ball_position_m, 3)
    || typeof metrics.ball_inside_bin !== "boolean"
    || !isFiniteNumber(metrics.tcp_ball_distance_m)
    || !isFiniteNumber(metrics.carry_distance_m)
  ) return null;

  return value as unknown as WingPodLiveState;
}