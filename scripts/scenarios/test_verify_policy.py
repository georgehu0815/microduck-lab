from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from verify_policy import VerificationError, validate_metadata, validate_telemetry


def sidecar(**metadata):
    return {
        "observation_dim": 61,
        "action_dim": 14,
        "metadata": {"recipe": "dance", "steps": 1200, **metadata},
    }


def telemetry(final_steps=1024, optimizer_steps=4, loss=0.25):
    return [
        {
            "phase": "collection",
            "steps": final_steps,
            "seconds": 1.0,
            "env_steps": final_steps,
            "mean_reward": 0.5,
        },
        {
            "phase": "update",
            "steps": final_steps,
            "seconds": 0.5,
            "env_steps": final_steps,
            "optimizer_steps": optimizer_steps,
            "mean_loss": loss,
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "entropy": 0.3,
            "approximate_kl": 0.01,
            "clip_fraction": 0.02,
            "explained_variance": 0.4,
        },
    ]


class MetadataTests(unittest.TestCase):
    def test_accepts_contract_and_resumed_total_steps(self):
        result = validate_metadata(sidecar(), "dance", 1000)
        self.assertEqual(result["metadata_steps"], 1200)

    def test_rejects_wrong_dimensions_recipe_and_steps(self):
        cases = [
            ({**sidecar(), "observation_dim": 60}, "dance", 1000),
            ({**sidecar(), "action_dim": 13}, "dance", 1000),
            (sidecar(), "swing", 1000),
            (sidecar(steps=999), "dance", 1000),
        ]
        for value, scenario, steps in cases:
            with self.subTest(value=value, scenario=scenario):
                with self.assertRaises(VerificationError):
                    validate_metadata(value, scenario, steps)


class TelemetryTests(unittest.TestCase):
    def test_accepts_finite_ppo_updates_and_local_transitions(self):
        result = validate_telemetry(telemetry(), 1000)
        self.assertEqual(result["optimizer_steps"], 4)
        self.assertEqual(result["final_local_env_steps"], 1024)
        self.assertEqual(result["collected_transitions"], 1024)

    def test_rejects_non_finite_ppo_telemetry(self):
        with self.assertRaisesRegex(VerificationError, "finite"):
            validate_telemetry(telemetry(loss=math.inf), 1000)

    def test_rejects_update_without_optimizer_work(self):
        with self.assertRaisesRegex(VerificationError, "greater than zero"):
            validate_telemetry(telemetry(optimizer_steps=0), 1000)

    def test_rejects_short_local_telemetry_despite_resumed_metadata(self):
        validate_metadata(sidecar(steps=5000), "dance", 1000)
        with self.assertRaisesRegex(VerificationError, "below required"):
            validate_telemetry(telemetry(final_steps=200), 1000)

    def test_rejects_missing_required_ppo_field(self):
        events = telemetry()
        del events[-1]["approximate_kl"]
        with self.assertRaisesRegex(VerificationError, "approximate_kl"):
            validate_telemetry(events, 1000)


if __name__ == "__main__":
    unittest.main()
