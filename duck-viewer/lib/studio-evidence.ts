import type { ExperimentId } from "./experiments";

export function skillEvidence(report: Record<string, unknown> | null, scenario: ExperimentId) {
  const key = scenario === "dance"
    ? "dance_assessment"
    : scenario === "swing"
      ? "swing_assessment"
      : scenario === "backflip"
        ? "backflip_assessment"
        : scenario === "drawing"
          ? "drawing_assessment"
        : "locomotion_assessment";
  const assessment = report?.[key] as Record<string, unknown> | undefined;
  const episodes = Array.isArray(assessment?.episodes)
    ? assessment.episodes.filter((episode): episode is Record<string, unknown> => Boolean(episode) && typeof episode === "object")
    : [];
  const metrics = scenario === "dance"
    ? ["pose_rmse_rad", "dynamic_gain", "leg_dynamic_gain", "moving_leg_joint_count", "upright_fraction"]
    : scenario === "swing"
      ? ["bidirectional_span_deg", "both_strings_tensioned_fraction", "valid_geometry_fraction", "max_alignment", "max_abs_lateral_m", "min_spring_tension_n"]
      : scenario === "backflip"
        ? ["rotation_rad", "landing_policy_steps", "stand_hold_seconds", "min_stand_height_m", "stand_upright_fraction", "assist_after_release_steps", "assist_force_n", "assist_torque_nm"]
        : scenario === "drawing"
          ? ["coverage", "precision", "symmetric_chamfer_m", "recorded_length_m", "reference_length_m", "length_ratio", "ink_contact_fraction"]
        : ["mean_command_directed_speed_m_s", "command_directed_displacement_m", "alternating_support_switches", "left_air_fraction", "right_air_fraction", "aerial_fraction", "upright_fraction"];
  return {
    episodes,
    passed: episodes.filter((episode) => episode.passed === true).length,
    criteria: assessment?.criteria && typeof assessment.criteria === "object" ? assessment.criteria as Record<string, unknown> : {},
    ranges: metrics.flatMap((metric) => {
      const values = episodes.map((episode) => episode[metric]).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
      return values.length ? [{ metric, minimum: Math.min(...values), maximum: Math.max(...values), measured: values.length }] : [];
    }),
  };
}

export function rolloutMetricRanges(
  report: Record<string, unknown> | null,
  metrics: string[]
) {
  const reported = report?.recipe_metrics;
  if (!reported || typeof reported !== "object" || Array.isArray(reported)) return [];
  return metrics.flatMap((metric) => {
    const range = (reported as Record<string, unknown>)[metric];
    if (!range || typeof range !== "object" || Array.isArray(range)) return [];
    const minimum = (range as Record<string, unknown>).min;
    const maximum = (range as Record<string, unknown>).max;
    return typeof minimum === "number" &&
      Number.isFinite(minimum) &&
      typeof maximum === "number" &&
      Number.isFinite(maximum)
      ? [{ metric, minimum, maximum }]
      : [];
  });
}

export function evidenceLabel(key: string) {
  return key.replaceAll("_", " ");
}
