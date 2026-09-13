from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import build_arm_video_evidence as evidence


class EpisodeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "seed": 80000, "steps": 2, "passed": True,
            "gates": {"no_drop": True}, "options": {"position_noise": 0.001},
            "return": 1.5,
        }
        self.expected.update(dict.fromkeys((
            "drop_count", "invalid_contacts", "arm_collisions", "self_collisions",
            "waypoints_reached", "position_error_m", "tip_error_m", "peak_lift_m",
            "peak_contact_force_n", "peak_internal_force_n", "payload_slip_m", "transport_tilt_deg",
        ), 0))
        self.receipt = {"episode": {
            "seed": 80000, "steps": 2, "return": 1.5,
            "reset_options": {"position_noise": 0.001, "mass_scale": 1.0, "friction_scale": 1.0},
            "metrics": {key: deepcopy(value) for key, value in self.expected.items() if key not in {"seed", "steps", "options", "return"}},
        }}

    def test_equivalent_default_options_are_accepted(self):
        evidence.assert_episode_matches(self.expected, self.receipt)

    def test_rejects_changed_verdict(self):
        self.receipt["episode"]["metrics"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "verdict"):
            evidence.assert_episode_matches(self.expected, self.receipt)

    def test_rejects_changed_gates(self):
        self.receipt["episode"]["metrics"]["gates"]["no_drop"] = False
        with self.assertRaisesRegex(ValueError, "gates"):
            evidence.assert_episode_matches(self.expected, self.receipt)

    def test_rejects_changed_perturbations(self):
        self.receipt["episode"]["reset_options"]["mass_scale"] = 0.9
        with self.assertRaisesRegex(ValueError, "perturbations"):
            evidence.assert_episode_matches(self.expected, self.receipt)

    def test_rejects_changed_reward(self):
        self.receipt["episode"]["return"] = 2
        with self.assertRaisesRegex(ValueError, "return"):
            evidence.assert_episode_matches(self.expected, self.receipt)

    def test_rejects_changed_force(self):
        self.receipt["episode"]["metrics"]["peak_contact_force_n"] = 100
        with self.assertRaisesRegex(ValueError, "peak_contact_force_n"):
            evidence.assert_episode_matches(self.expected, self.receipt)


class SelectionTests(unittest.TestCase):
    def test_first_episode_is_not_replaced_with_a_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            first = {"passed": False, "seed": 80000, "options": {}}
            other_failure = {"passed": False, "seed": 80002, "options": {}}
            historical = {"passed": False, "seed": 50003, "options": {}}
            path.write_text(json.dumps({"results": [first, {"passed": True, "seed": 80001, "options": {}}, other_failure]}))
            summary = {"jobs": [{
                "case_id": "arm-reach-v1", "training_seed": 101,
                "checkpoint": "/checkpoint.zip", "evaluation": str(path),
                "evaluation_sha256": evidence.digest(path),
                "historical_failure_replays": [{"replay": historical}],
            }]}
            selected = evidence.select_episodes(summary)
            self.assertEqual([row["expected_episode"]["seed"] for row in selected], [80000, 50003, 80002])
            self.assertEqual(selected[0]["selection_reasons"], ["first-new-seed", "new-failure"])

    def test_changed_evaluation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            path.write_text('{"results": []}')
            with self.assertRaisesRegex(ValueError, "Evaluation changed"):
                evidence.select_episodes({"jobs": [{"evaluation": str(path), "evaluation_sha256": "wrong"}]})


if __name__ == "__main__":
    unittest.main()
