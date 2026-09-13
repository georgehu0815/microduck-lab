"""Strict per-episode acceptance for an unassisted suspended-bridge crossing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from numbers import Integral, Real
from typing import Any


CONTROL_HZ = 50
REQUIRED_METRICS = (
    "bridge_crossed",
    "bridge_progress_m",
    "bridge_assistance",
    "robot_floor_contact",
    "nonfoot_support",
    "feet_support",
    "bridge_contact_observed",
)
_BINARY_METRICS = (
    "bridge_crossed",
    "robot_floor_contact",
    "nonfoot_support",
    "bridge_contact_observed",
)


@dataclass(frozen=True)
class BridgeEvaluationCriteria:
    version: int = 1
    required_steps: int = 1000
    control_hz: int = CONTROL_HZ
    min_progress_m: float = 1.60
    required_end_feet: int = 2
    requires_unassisted: bool = True
    requires_no_falls: bool = True
    all_episodes_required: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.required_steps, bool)
            or not isinstance(self.required_steps, Integral)
            or self.required_steps < 1
        ):
            raise ValueError("bridge required_steps must be a positive integer")

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
    max_progress_m: float = 0.0
    crossed: bool = False
    crossing_step: int | None = None
    start_support_observed: bool = False
    end_feet_at_crossing: int = 0
    assisted_steps: int = 0
    floor_contact_steps: int = 0
    nonfoot_support_steps: int = 0
    previous_crossed: float | None = None
    bridge_contact_observed: bool = False
    failures: set[str] = field(default_factory=set)

    def observe(
        self,
        metrics: dict[str, Any],
        criteria: BridgeEvaluationCriteria,
    ) -> None:
        self.steps += 1
        values = _numeric_metrics(metrics)
        if values is None:
            self.failures.add("missing, malformed, or non-finite bridge metrics")
            return
        if any(values[key] not in (0.0, 1.0) for key in _BINARY_METRICS):
            self.failures.add("invalid binary bridge metrics")
            return
        if (
            values["bridge_progress_m"] < -0.5
            or values["bridge_assistance"] < 0
            or values["bridge_assistance"] > 1
            or values["feet_support"] not in (0.0, 1.0, 2.0)
        ):
            self.failures.add("invalid physical bridge metrics")
            return
        if self.previous_crossed == 1.0 and values["bridge_crossed"] == 0.0:
            self.failures.add("bridge_crossed regressed within an episode")

        self.measured_steps += 1
        self.max_progress_m = max(
            self.max_progress_m,
            values["bridge_progress_m"],
        )
        self.start_support_observed = (
            self.start_support_observed
            or (
                values["bridge_progress_m"] <= 0.10
                and values["feet_support"] >= 1.0
            )
        )
        self.assisted_steps += int(values["bridge_assistance"] > 0.0)
        self.floor_contact_steps += int(values["robot_floor_contact"] == 1.0)
        self.nonfoot_support_steps += int(values["nonfoot_support"] == 1.0)
        self.bridge_contact_observed = (
            self.bridge_contact_observed
            or values["bridge_contact_observed"] == 1.0
        )

        crossing_now = values["bridge_crossed"] == 1.0 and not self.crossed
        if crossing_now:
            self.crossed = True
            self.crossing_step = self.steps
            self.end_feet_at_crossing = int(values["feet_support"])
            if self.end_feet_at_crossing != criteria.required_end_feet:
                self.failures.add("crossing lacked two-foot landing support")
            if not self.bridge_contact_observed:
                self.failures.add(
                    "crossing lacked suspended-plank foot-contact history"
                )
        self.previous_crossed = values["bridge_crossed"]

    def finish(
        self,
        criteria: BridgeEvaluationCriteria,
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
            failures.add("episode terminated or fell before evaluation completion")
        if not complete:
            failures.add(
                f"episode did not complete the required "
                f"{criteria.required_steps / criteria.control_hz:g}-second horizon"
            )
        if self.steps == 0 or self.measured_steps != self.steps:
            failures.add("incomplete bridge metric coverage")
        if not self.start_support_observed:
            failures.add("launch-platform foot support was not observed")
        if not self.crossed:
            failures.add("contact-validated end-to-end crossing was not observed")
        if not self.bridge_contact_observed:
            failures.add("suspended-plank foot contact was not observed")
        if self.max_progress_m < criteria.min_progress_m:
            failures.add("bridge progress remained below end-to-end target")
        if self.assisted_steps:
            failures.add("bridge assistance was active during evaluation")
        if self.floor_contact_steps:
            failures.add("robot contacted the floor")
        if self.nonfoot_support_steps:
            failures.add("non-foot robot support contact occurred")

        return {
            "env_index": self.env_index,
            "episode_index": self.index,
            "steps": self.steps,
            "measured_steps": self.measured_steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "complete": complete,
            "max_progress_m": self.max_progress_m,
            "crossed": self.crossed,
            "crossing_step": self.crossing_step,
            "start_support_observed": self.start_support_observed,
            "end_feet_at_crossing": self.end_feet_at_crossing,
            "bridge_contact_observed": self.bridge_contact_observed,
            "assisted_steps": self.assisted_steps,
            "floor_contact_steps": self.floor_contact_steps,
            "nonfoot_support_steps": self.nonfoot_support_steps,
            "passed": not failures,
            "failures": sorted(failures),
        }


class BridgeEvaluation:
    """Accumulate strict unassisted crossing evidence per vector lane."""

    criteria: BridgeEvaluationCriteria
    _current: list[_Episode]

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
            or required_steps < 1
        ):
            raise ValueError("required_steps must be a positive integer")
        self.criteria = BridgeEvaluationCriteria(required_steps=int(required_steps))
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
        failures = {
            failure
            for episode in episodes
            for failure in episode["failures"]
        }
        if set(range(len(self._current))) - covered_lanes or not episodes:
            failures.add("missing evaluation lane coverage")
        return {
            "passed": bool(episodes) and not failures,
            "failures": sorted(failures),
            "episodes": episodes,
            "criteria": self.criteria.to_dict(),
            "completed_episodes": sum(
                bool(episode["complete"]) for episode in episodes
            ),
            "passed_episodes": sum(
                bool(episode["passed"]) for episode in episodes
            ),
            "measurement_scope": (
                "50 Hz per-episode numeric recipe metrics; crossing credit "
                "requires full progress and contact-validated two-foot landing"
            ),
            "hardware_safe": False,
        }


__all__ = [
    "BridgeEvaluation",
    "BridgeEvaluationCriteria",
    "REQUIRED_METRICS",
]
