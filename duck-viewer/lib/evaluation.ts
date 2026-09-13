import type { ExperimentId } from "./experiments";

/** Project evaluator verdicts without deriving skill acceptance from motion metrics. */
export function evaluationVerdict(
  report: Record<string, unknown> | null,
  experimentId: ExperimentId
) {
  const scope = report?.evaluation_mode;
  const explicitSkillStatus =
    report?.skill_status === "passed" || report?.skill_status === "failed";
  const supportsSkillAssessment =
    experimentId === "dance" || experimentId === "swing" ||
    experimentId === "running" || experimentId === "stilts" ||
    experimentId === "backflip" || experimentId === "basketball" ||
    experimentId === "bridge" || experimentId === "drawing";
  const matchingScenario = report?.recipe === experimentId;
  const basketball = report?.basketball_assessment as
    | Record<string, unknown>
    | undefined;
  const bridge = report?.bridge_assessment as
    | Record<string, unknown>
    | undefined;
  const drawing = report?.drawing_assessment as
    | Record<string, unknown>
    | undefined;
  const basketballCases = Array.isArray(basketball?.cases)
    ? basketball.cases.filter(
        (value): value is Record<string, unknown> =>
          typeof value === "object" && value !== null && !Array.isArray(value)
      )
    : [];
  const balanceCase = basketballCases.find(
    (value) => value.classification === "zero_command_balance"
  );
  const steeringCases = basketballCases.filter(
    (value) => value.classification === "commanded_rolling"
  );
  const evaluatorReport = report?.report as Record<string, unknown> | undefined;
  const protocol = evaluatorReport?.protocol as Record<string, unknown> | undefined;
  const balanceSeconds = protocol?.seconds;
  const basketballBalancePassed =
    experimentId === "basketball" &&
    balanceCase?.passed === true &&
    typeof balanceCase.trial_count === "number" &&
    balanceCase.trial_count > 0 &&
    balanceCase.sustained_balance_count === balanceCase.trial_count &&
    typeof balanceSeconds === "number" &&
    Number.isFinite(balanceSeconds) &&
    balanceSeconds >= 60;
  const basketballSteeringPassed =
    basketballBalancePassed &&
    steeringCases.length >= 3 &&
    steeringCases.every((value) => value.passed === true);
  const bridgeCriteria = bridge?.criteria as Record<string, unknown> | undefined;
  const bridgeEpisodes = Array.isArray(bridge?.episodes)
    ? bridge.episodes.filter(
        (value): value is Record<string, unknown> =>
          typeof value === "object" && value !== null && !Array.isArray(value)
      )
    : [];
  const bridgeUnassisted =
    bridgeCriteria?.requires_unassisted === true &&
    bridgeEpisodes.length > 0 &&
    bridgeEpisodes.every((episode) => episode.assisted_steps === 0);
  const drawingUnassisted = drawing?.unassisted === true;
  const scenarioGatePassed =
    experimentId === "basketball"
      ? false
      : experimentId === "bridge"
        ? bridge?.passed === true && bridgeUnassisted
        : experimentId === "drawing"
          ? drawing?.passed === true && drawingUnassisted
        : true;
  const skillAssessed = scope === "skill" && matchingScenario &&
    report?.evaluation_settings_match !== false && supportsSkillAssessment && explicitSkillStatus;
  const settings = report?.evaluation as Record<string, unknown> | undefined;
  const criteria = settings?.swing_criteria as Record<string, unknown> | undefined;
  const target = criteria?.min_bidirectional_span_deg;
  return {
    swingMinSpanDeg: typeof target === "number" && Number.isFinite(target) ? target : null,
    scope: scope === "skill" || scope === "pipeline" ? scope : null,
    pipelinePassed: report?.pipeline_passed === true,
    skillAssessed,
    balanceOnly:
      skillAssessed &&
      report?.pipeline_passed === true &&
      basketballBalancePassed &&
      !basketballSteeringPassed,
    basketballBalancePassed,
    basketballSteeringPassed,
    bridgeUnassisted,
    drawingUnassisted,
    taskPassed: skillAssessed && report?.pipeline_passed === true && report?.passed === true &&
      report?.skill_status === "passed" && scenarioGatePassed,
  };
}
