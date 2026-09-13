"""Strict per-episode acceptance for an assisted backflip showcase."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from numbers import Integral, Real
from typing import Any


CONTROL_HZ = 50
REQUIRED_METRICS = (
    "rotation_rad",
    "spotter_active",
    "assist_torque_norm",
    "launch_complete",
    "landing_policy_steps",
    "handed_off",
    "upright",
    "height_m",
    "both_feet",
    "nonfoot_contact",
    "gyro_norm",
)
_BINARY_METRICS = (
    "spotter_active",
    "launch_complete",
    "handed_off",
    "both_feet",
    "nonfoot_contact",
)


@dataclass(frozen=True)
class BackflipEvaluationCriteria:
    """Fixed physical evidence required for the assisted showcase."""

    version: int = 2
    required_steps: int = 300
    control_hz: int = CONTROL_HZ
    min_rotation_rad: float = 5.8
    max_rotation_rad: float = 2.0 * math.pi
    min_landing_policy_steps: int = 10
    min_pre_handoff_stable_steps: int = 10
    min_stand_hold_steps: int = 100
    min_stand_upright: float = 0.9
    min_stand_height_m: float = 0.105
    max_stand_gyro_norm: float = math.sqrt(2.0)
    requires_launch_assistance: bool = True
    credits_unassisted_skill: bool = False
    all_episodes_required: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.required_steps, bool)
            or not isinstance(self.required_steps, Integral)
            or self.required_steps < 6 * CONTROL_HZ
        ):
            raise ValueError(
                "backflip evaluation requires at least 300 steps (6 seconds)"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numeric_metrics(metrics: Any) -> dict[str, float] | None:
    if not isinstance(metrics, dict):
        return None
    values: dict[str, float] = {}
    for key in REQUIRED_METRICS:
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        converted = float(value)
        if not math.isfinite(converted):
            return None
        values[key] = converted
    return values


@dataclass
class _Episode:
    env_index: int
    index: int
    steps: int = 0
    measured_steps: int = 0
    rotation_rad: float = 0.0
    landing_policy_steps: int = 0
    landing_policy_steps_before_handoff: int = 0
    landing_stable_steps: int = 0
    pre_handoff_stable_steps: int = 0
    rotation_at_handoff: float | None = None
    assistance_observed: bool = False
    release_observed: bool = False
    handed_off_observed: bool = False
    previous_rotation_rad: float | None = None
    previous_landing_policy_steps: int | None = None
    previous_launch_complete: int | None = None
    previous_handed_off: int | None = None
    post_handoff_steps: int = 0
    post_handoff_upright_steps: int = 0
    post_handoff_min_height_m: float | None = None
    trailing_stand_steps: int = 0
    assist_after_release_steps: int = 0
    failures: set[str] = field(default_factory=set)

    def observe(
        self,
        metrics: dict[str, Any],
        criteria: BackflipEvaluationCriteria,
    ) -> None:
        self.steps += 1
        values = _numeric_metrics(metrics)
        if values is None:
            self.failures.add("missing, malformed, or non-finite backflip metrics")
            self.landing_stable_steps = 0
            self.previous_rotation_rad = None
            return
        if any(values[key] not in (0.0, 1.0) for key in _BINARY_METRICS):
            self.failures.add("invalid binary backflip metrics")
            self.landing_stable_steps = 0
            self.previous_rotation_rad = None
            return
        landing_steps_value = values["landing_policy_steps"]
        if (
            landing_steps_value < 0.0
            or not landing_steps_value.is_integer()
            or values["rotation_rad"] < 0.0
            or values["rotation_rad"] > criteria.max_rotation_rad
            or values["assist_torque_norm"] < 0.0
            or values["gyro_norm"] < 0.0
        ):
            self.failures.add("invalid cumulative or physical backflip metrics")
            self.landing_stable_steps = 0
            self.previous_rotation_rad = None
            return

        rotation = values["rotation_rad"]
        landing_steps = int(landing_steps_value)
        launch_complete = int(values["launch_complete"])
        handed_off = int(values["handed_off"])
        spotter_active = int(values["spotter_active"])
        assist_active = spotter_active == 1 or values["assist_torque_norm"] > 0.0

        if (
            self.previous_rotation_rad is not None
            and rotation < self.previous_rotation_rad
        ):
            self.failures.add("rotation_rad must be cumulative within an episode")
        if (
            self.previous_landing_policy_steps is not None
            and landing_steps < self.previous_landing_policy_steps
        ):
            self.failures.add(
                "landing_policy_steps must be cumulative within an episode"
            )
        if self.previous_launch_complete == 1 and launch_complete == 0:
            self.failures.add("launch_complete regressed after release")
        if self.previous_handed_off == 1 and handed_off == 0:
            self.failures.add("handed_off regressed after handoff")

        self.measured_steps += 1
        self.rotation_rad = max(self.rotation_rad, rotation)
        self.landing_policy_steps = max(self.landing_policy_steps, landing_steps)

        if spotter_active == 1 and values["assist_torque_norm"] > 0.0:
            self.assistance_observed = True
        release_now = (
            launch_complete == 1
            and self.previous_launch_complete == 0
            and self.assistance_observed
        )
        self.release_observed = self.release_observed or release_now
        if release_now and landing_steps > 1:
            self.failures.add("landing_policy_steps did not reset at release")
        if self.release_observed and assist_active:
            self.assist_after_release_steps += 1

        stable = (
            values["upright"] >= criteria.min_stand_upright
            and values["height_m"] >= criteria.min_stand_height_m
            and values["both_feet"] == 1.0
            and values["nonfoot_contact"] == 0.0
            and values["gyro_norm"] < criteria.max_stand_gyro_norm
            and not assist_active
        )
        before_handoff = handed_off == 0 and not self.handed_off_observed
        if self.release_observed and before_handoff and not assist_active:
            self.landing_policy_steps_before_handoff = max(
                self.landing_policy_steps_before_handoff,
                landing_steps,
            )
        if before_handoff:
            landing_sample = (
                self.release_observed
                and launch_complete == 1
                and self.previous_landing_policy_steps is not None
                and landing_steps == self.previous_landing_policy_steps + 1
            )
            self.landing_stable_steps = (
                self.landing_stable_steps + 1 if landing_sample and stable else 0
            )

        handoff_now = handed_off == 1 and not self.handed_off_observed
        if handoff_now:
            self.pre_handoff_stable_steps = self.landing_stable_steps
            self.rotation_at_handoff = self.previous_rotation_rad
            if (
                self.rotation_at_handoff is None
                or self.rotation_at_handoff < criteria.min_rotation_rad
            ):
                self.failures.add("backward rotation below target before handoff")
        self.handed_off_observed = self.handed_off_observed or handoff_now
        if self.handed_off_observed and handed_off == 1:
            self.post_handoff_steps += 1
            self.post_handoff_upright_steps += int(
                values["upright"] >= criteria.min_stand_upright
            )
            self.post_handoff_min_height_m = (
                values["height_m"]
                if self.post_handoff_min_height_m is None
                else min(self.post_handoff_min_height_m, values["height_m"])
            )
            self.trailing_stand_steps = self.trailing_stand_steps + 1 if stable else 0

        self.previous_rotation_rad = rotation
        self.previous_landing_policy_steps = landing_steps
        self.previous_launch_complete = launch_complete
        self.previous_handed_off = handed_off

    def finish(
        self,
        criteria: BackflipEvaluationCriteria,
        *,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        failures = set(self.failures)
        complete = (
            truncated
            and not terminated
            and self.steps >= criteria.required_steps
        )
        if terminated:
            failures.add("episode terminated before showcase completion")
        if not complete:
            failures.add(
                f"episode did not complete the required "
                f"{criteria.required_steps / criteria.control_hz:g}-second horizon"
            )
        if self.steps == 0 or self.measured_steps != self.steps:
            failures.add("incomplete backflip metric coverage")
        if self.rotation_rad < criteria.min_rotation_rad:
            failures.add("backward rotation below target")
        if not self.assistance_observed:
            failures.add("launch assistance was not observed")
        if not self.release_observed:
            failures.add("assisted launch release was not recorded")
        if (
            self.landing_policy_steps_before_handoff
            < criteria.min_landing_policy_steps
        ):
            failures.add("insufficient unassisted landing-policy steps before handoff")
        if not self.handed_off_observed:
            failures.add("landing-to-stand handoff was not recorded")
        if self.pre_handoff_stable_steps < criteria.min_pre_handoff_stable_steps:
            failures.add(
                "insufficient consecutive stable landing-policy steps before handoff"
            )
        if self.trailing_stand_steps < criteria.min_stand_hold_steps:
            failures.add("final stable stand hold below target")
        if self.assist_after_release_steps:
            failures.add("spotter or assist torque remained active after release")

        stand_upright_fraction = (
            self.post_handoff_upright_steps / self.post_handoff_steps
            if self.post_handoff_steps
            else 0.0
        )
        return {
            "env_index": self.env_index,
            "episode_index": self.index,
            "steps": self.steps,
            "measured_steps": self.measured_steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "rotation_rad": self.rotation_rad,
            "landing_policy_steps": self.landing_policy_steps,
            "pre_handoff_stable_steps": self.pre_handoff_stable_steps,
            "rotation_at_handoff": self.rotation_at_handoff,
            "stand_hold_seconds": self.trailing_stand_steps / criteria.control_hz,
            "min_stand_height_m": self.post_handoff_min_height_m,
            "stand_upright_fraction": stand_upright_fraction,
            "assist_after_release_steps": self.assist_after_release_steps,
            "failures": sorted(failures),
            "passed": not failures,
            "complete": complete,
        }


class BackflipEvaluation:
    """Accumulate assisted-showcase evidence independently per vector lane."""

    def __init__(self, num_envs: int, required_steps: int) -> None:
        if (
            isinstance(num_envs, bool)
            or not isinstance(num_envs, Integral)
            or num_envs < 1
        ):
            raise ValueError("num_envs must be positive")
        if (
            isinstance(required_steps, bool)
            or not isinstance(required_steps, Integral)
        ):
            raise ValueError("required_steps must be an integer")
        self.criteria = BackflipEvaluationCriteria(required_steps=int(required_steps))
        self._current = [_Episode(index, 0) for index in range(int(num_envs))]
        self._episodes: list[dict[str, Any]] = []

    def observe(
        self,
        env_index: int,
        metrics: dict[str, Any],
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        if not 0 <= env_index < len(self._current):
            raise IndexError(f"env_index out of range: {env_index}")
        episode = self._current[env_index]
        episode.observe(metrics, self.criteria)
        if terminated or truncated:
            self._episodes.append(
                episode.finish(
                    self.criteria,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )
            self._current[env_index] = _Episode(env_index, episode.index + 1)

    def report(self) -> dict[str, Any]:
        episodes = [
            *self._episodes,
            *(
                episode.finish(
                    self.criteria,
                    terminated=False,
                    truncated=False,
                )
                for episode in self._current
                if episode.steps
            ),
        ]
        covered_lanes = {episode["env_index"] for episode in episodes}
        missing_lanes = sorted(set(range(len(self._current))) - covered_lanes)
        completed = sum(bool(episode["complete"]) for episode in episodes)
        failures = {
            failure
            for episode in episodes
            for failure in episode["failures"]
        }
        if missing_lanes or not episodes:
            failures.add("missing evaluation lane coverage")
        return {
            "criteria": self.criteria.to_dict(),
            "episodes": episodes,
            "completed_episodes": completed,
            "incomplete_episodes": len(episodes) - completed,
            "passed_episodes": sum(bool(episode["passed"]) for episode in episodes),
            "passed": bool(episodes) and not failures,
            "failures": sorted(failures),
            "measurement_scope": (
                "50 Hz per-episode samples for a clearly assisted launch, "
                "unassisted learned-policy landing window, explicit handoff, "
                "and final stable stand"
            ),
            "credit_scope": (
                "simulator assisted showcase acceptance only; passing does not "
                "establish PPO improvement, an unassisted learned backflip, "
                "or distinguish a null control"
            ),
            "hardware_safe": False,
        }


__all__ = [
    "BackflipEvaluation",
    "BackflipEvaluationCriteria",
    "REQUIRED_METRICS",
]
