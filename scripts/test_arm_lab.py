from __future__ import annotations

import hashlib
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import quote

from scripts import arm_lab


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QuietHandler(arm_lab.Handler):
    def log_message(self, format, *args):
        return None


class ArmLabStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lab = arm_lab.ArmLab()

    def tearDown(self) -> None:
        self.lab.environment.close()

    def test_reset_returns_real_simulation_state_and_contract(self) -> None:
        state = self.lab.command(
            "reset",
            {"case_id": "arms-handover-v1", "seed": 9127},
        )

        self.assertEqual(state["case_id"], "arms-handover-v1")
        self.assertEqual(state["seed"], 9127)
        self.assertEqual(state["contract"], "md-dualarm-table-v1")
        self.assertEqual(state["observation_dim"], 116)
        self.assertEqual(state["action_dim"], 12)
        self.assertEqual(len(state["joints"]), 12)
        self.assertGreater(len(state["geoms"]), 0)
        self.assertEqual(state["step"], 0)
        self.assertEqual(state["controller"], "manual")
        self.assertFalse(state["hardware_enabled"])
        self.assertTrue(state["metrics"]["simulation_only"])
        self.assertFalse(state["metrics"]["physical_hardware_verified"])

    def test_step_mutates_real_environment_state_with_manual_simulation_label(self) -> None:
        before = self.lab.state()
        after = self.lab.command("step", {"action": [0.25, 0, 0, 0, 0, 0]})

        self.assertEqual(after["step"], before["step"] + 1)
        self.assertAlmostEqual(after["time"], 0.02)
        self.assertEqual(after["controller"], "manual_simulation")
        self.assertNotEqual(after["joints"], before["joints"])
        self.assertFalse(after["hardware_enabled"])
        json.dumps(after, allow_nan=False)

    def test_teacher_advances_real_state_and_records_authored_control(self) -> None:
        state = self.lab.command("teacher", {"ticks": 3})

        self.assertEqual(state["step"], 3)
        self.assertEqual(
            state["controller"],
            "authored_ik_fsm_teacher_not_ppo",
        )
        self.assertEqual(state["metrics"]["authored_action_calls"], 3)
        self.assertEqual(state["metrics"]["physics_assistance_count"], 0)
        json.dumps(state, allow_nan=False)
        self.assertFalse(state["hardware_enabled"])

    def test_command_payload_schema_and_numeric_limits(self) -> None:
        invalid = (
            ("reset", []),
            ("reset", {"case_id": arm_lab.CASES[0], "seed": True}),
            ("reset", {"case_id": arm_lab.CASES[0], "seed": -1}),
            ("reset", {"case_id": "not-a-case", "seed": 1}),
            ("reset", {"case_id": arm_lab.CASES[0], "seed": 1, "extra": 1}),
            ("step", {}),
            ("step", {"action": [0] * 5}),
            ("step", {"action": [0, 0, 0, 0, 0, 1.01]}),
            ("step", {"action": [0, 0, 0, 0, 0, float("nan")]}),
            ("step", {"action": [False, 0, 0, 0, 0, 0]}),
            ("step", {"action": [0] * 6, "extra": 1}),
            ("teacher", {"ticks": 0}),
            ("teacher", {"ticks": 51}),
            ("teacher", {"ticks": True}),
            ("teacher", {"ticks": 1, "extra": 1}),
            ("hardware", {}),
        )
        for operation, payload in invalid:
            with self.subTest(operation=operation, payload=payload):
                with self.assertRaises((ValueError, RuntimeError)):
                    self.lab.command(operation, payload)


class WingPodLabStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lab = arm_lab.WingPodLab()

    def tearDown(self) -> None:
        self.lab.environment.close()

    def test_live_state_contains_finalized_wingpod_scene(self) -> None:
        state = self.lab.state()

        self.assertEqual(state["case_id"], "wingpod-tennis-v1")
        self.assertEqual(state["contract"], "wingpod-tennis-live-v1")
        self.assertEqual(state["action_dim"], 15)
        self.assertGreater(len(state["geoms"]), 100)
        self.assertIn("ellipsoid", {geom["type"] for geom in state["geoms"]})
        self.assertFalse(state["metrics"]["trained_policy"])
        self.assertFalse(state["metrics"]["vision_control"])
        json.dumps(state, allow_nan=False)

    def test_live_teacher_steps_real_mujoco_state(self) -> None:
        before = self.lab.state()
        after = self.lab.command("wingpod-teacher", {"ticks": 2})

        self.assertEqual(after["step"], before["step"] + 2)
        self.assertGreater(after["time"], before["time"])
        self.assertNotEqual(after["joints"], before["joints"])

    def test_live_reset_and_command_bounds(self) -> None:
        state = self.lab.command("wingpod-reset", {"seed": 7})
        self.assertEqual(state["seed"], 7)
        self.assertEqual(state["step"], 0)
        for operation, payload in (
            ("wingpod-reset", {"seed": True}),
            ("wingpod-teacher", {"ticks": 0}),
            ("wingpod-teacher", {"ticks": 11}),
            ("wingpod-teacher", {"ticks": 1, "extra": 1}),
        ):
            with self.subTest(operation=operation, payload=payload):
                with self.assertRaises(ValueError):
                    self.lab.command(operation, payload)


class HttpArmLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.runs = Path(cls.temp.name) / "runs"
        cls.runs.mkdir()
        cls.lab = arm_lab.ArmLab()
        QuietHandler.lab = cls.lab
        cls.runs_patch = patch.object(arm_lab, "RUNS", cls.runs)
        cls.runs_patch.start()
        cls.server = arm_lab.ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.host, cls.port = cls.server.server_address

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.runs_patch.stop()
        cls.lab.environment.close()
        cls.temp.cleanup()

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def post_json(self, path: str, payload) -> tuple[int, dict, bytes]:
        body = json.dumps(payload, allow_nan=False).encode()
        return self.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )

    def test_http_is_explicitly_simulation_only_and_not_the_unix_coordinator(self) -> None:
        status, _, body = self.request("GET", "/health")
        health = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(health["service"], "microduck-arm-simulation")
        self.assertFalse(health["hardware_enabled"])

        status, _, body = self.post_json("/manipulation.acquire", {})
        refusal = json.loads(body)
        self.assertEqual(status, 400)
        self.assertFalse(refusal["hardware_enabled"])
        self.assertIn("hardware and training are not exposed", refusal["error"])

    def test_main_defaults_to_dedicated_loopback_port_8812(self) -> None:
        class FakeEnvironment:
            def close(self) -> None:
                return None

        class FakeLab:
            environment = FakeEnvironment()

        class FakeServer:
            address = None

            def __init__(self, address, handler) -> None:
                self.__class__.address = address
                self.handler = handler

            def serve_forever(self) -> None:
                return None

            def server_close(self) -> None:
                return None

        with (
            patch.object(sys, "argv", ["arm_lab.py"]),
            patch.object(arm_lab, "ArmLab", return_value=FakeLab()),
            patch.object(arm_lab, "ThreadingHTTPServer", FakeServer),
            patch("builtins.print"),
        ):
            arm_lab.main()

        self.assertEqual(FakeServer.address, ("127.0.0.1", 8812))

    def test_http_payload_content_type_size_json_and_schema_limits(self) -> None:
        status, _, _ = self.request(
            "POST",
            "/reset",
            body=b"{}",
            headers={"Content-Type": "text/plain", "Content-Length": "2"},
        )
        self.assertEqual(status, 415)

        status, _, body = self.request(
            "POST",
            "/reset",
            body=b"x" * 16385,
            headers={"Content-Type": "application/json", "Content-Length": "16385"},
        )
        self.assertEqual(status, 400)
        self.assertIn("16384", json.loads(body)["error"])

        for raw in (b"{", b"[]", b'{"seed":NaN}'):
            with self.subTest(raw=raw):
                status, _, body = self.request(
                    "POST",
                    "/reset",
                    body=raw,
                    headers={
                        "Content-Type": "application/json",
                        "Content-Length": str(len(raw)),
                    },
                )
                self.assertEqual(status, 400)
                self.assertFalse(json.loads(body)["hardware_enabled"])

        status, _, body = self.post_json(
            "/step",
            {"action": [0, 0, 0, 0, 0, 2]},
        )
        self.assertEqual(status, 400)
        self.assertIn("normalized", json.loads(body)["error"])

    def test_artifact_endpoint_rejects_traversal_symlinks_and_unlisted_extensions(self) -> None:
        allowed = self.runs / "run" / "metrics.json"
        allowed.parent.mkdir()
        allowed.write_text('{"ok":true}', encoding="utf-8")
        outside = Path(self.temp.name) / "secret.json"
        outside.write_text('{"secret":true}', encoding="utf-8")
        text = allowed.with_suffix(".txt")
        text.write_text("not an allowed artifact", encoding="utf-8")
        link = allowed.parent / "linked.json"
        try:
            link.symlink_to(outside)
        except OSError:
            link = None

        status, headers, body = self.request("GET", "/artifact?path=run/metrics.json")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"ok": True})
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")

        rejected = (
            "../secret.json",
            "/etc/passwd",
            "run/../run/../../secret.json",
            "run/metrics.txt",
        )
        if link is not None:
            rejected += ("run/linked.json",)
        for path in rejected:
            with self.subTest(path=path):
                status, _, _ = self.request(
                    "GET",
                    f"/artifact?path={quote(path, safe='')}",
                )
                self.assertEqual(status, 404)

    def test_artifact_ranges_are_bounded(self) -> None:
        artifact = self.runs / "clip.mp4"
        artifact.write_bytes(b"0123456789")

        status, headers, body = self.request(
            "GET",
            "/artifact?path=clip.mp4",
            headers={"Range": "bytes=2-5"},
        )
        self.assertEqual(status, 206)
        self.assertEqual(headers["Content-Range"], "bytes 2-5/10")
        self.assertEqual(body, b"2345")

        for byte_range in ("items=0-1", "bytes=20-", "bytes=8-2"):
            with self.subTest(byte_range=byte_range):
                status, _, _ = self.request(
                    "GET",
                    "/artifact?path=clip.mp4",
                    headers={"Range": byte_range},
                )
                self.assertEqual(status, 416)


class RunCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.runs = Path(self.temp.name)
        self.patch = patch.object(arm_lab, "RUNS", self.runs)
        self.patch.start()
        self.case_id = "arm-reach-v1"
        self.model_sha256 = arm_lab.model_hash(self.case_id)
        self.expected_sources = {
            "environment": arm_lab.file_hash(
                arm_lab.ROOT / "rlx/rlx/environments/arm.py"
            ),
            "pipeline": arm_lab.file_hash(
                arm_lab.ROOT / "rlx/examples/ppo_microduck_arm.py"
            ),
        }

    def tearDown(self) -> None:
        self.patch.stop()
        self.temp.cleanup()

    def write_report(
        self,
        *,
        controller: str = "teacher",
        source_hashes: dict | None = None,
        model_sha256: str | None = None,
        checkpoint_sha256: str | None = None,
        report_name: str | None = None,
    ) -> tuple[Path, dict]:
        directory = self.runs / self.case_id / "run-a"
        directory.mkdir(parents=True, exist_ok=True)
        report = {
            "case_id": self.case_id,
            "controller": controller,
            "passed": True,
            "source_hashes": self.expected_sources if source_hashes is None else source_hashes,
            "model_sha256": self.model_sha256 if model_sha256 is None else model_sha256,
            "simulation_only": True,
            "hardware_verified": False,
            "results": [],
        }
        if checkpoint_sha256 is not None:
            report["checkpoint_sha256"] = checkpoint_sha256
        default_name = {
            "ppo_residual": "evaluation.json",
            "bc": "bc-evaluation.json",
        }.get(controller, "teacher-evaluation.json")
        report_path = directory / (report_name or default_name)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return directory, report

    def write_training(
        self,
        directory: Path,
        checkpoint_sha256: str,
        **overrides,
    ) -> dict:
        metadata = {
            "case_id": self.case_id,
            "model_sha256": self.model_sha256,
            "source_hashes": self.expected_sources,
            "checkpoint_sha256": checkpoint_sha256,
        }
        metadata.update(overrides)
        (directory / "training.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        return metadata

    def write_receipt(
        self,
        directory: Path,
        report: dict,
        **overrides,
    ) -> dict:
        video = directory / "rollout.mp4"
        receipt = {
            "video_sha256": sha256(video),
            "source_hashes": self.expected_sources,
            "case_id": report["case_id"],
            "controller": report["controller"],
            "model_sha256": self.model_sha256,
            "checkpoint_sha256": (
                report.get("checkpoint_sha256")
                if report["controller"] == "ppo_residual"
                else None
            ),
        }
        receipt.update(overrides)
        (directory / "render-receipt.json").write_text(
            json.dumps(receipt),
            encoding="utf-8",
        )
        return receipt

    def test_report_source_mismatch_cannot_pass(self) -> None:
        self.write_report(source_hashes={"environment": "old", "pipeline": "old"})
        item = arm_lab.run_catalog()["runs"][0]
        self.assertFalse(item["passed"])
        self.assertFalse(item["metrics"]["sources_match"])

    def test_ppo_requires_checkpoint_and_complete_matching_training_metadata(self) -> None:
        directory, _ = self.write_report(
            controller="ppo_residual",
            checkpoint_sha256="missing",
        )
        item = arm_lab.run_catalog()["runs"][0]
        self.assertFalse(item["passed"])
        self.assertFalse(item["metrics"]["checkpoint_match"])

        checkpoint = directory / "ppo-residual.zip"
        checkpoint.write_bytes(b"checkpoint")
        checkpoint_sha256 = sha256(checkpoint)
        self.write_report(
            controller="ppo_residual",
            checkpoint_sha256=checkpoint_sha256,
        )
        item = arm_lab.run_catalog()["runs"][0]
        self.assertFalse(item["passed"])
        self.assertFalse(item["metrics"]["checkpoint_match"])

        valid_metadata = self.write_training(directory, checkpoint_sha256)
        for key in ("case_id", "model_sha256", "source_hashes", "checkpoint_sha256"):
            with self.subTest(training_field=key):
                invalid_metadata = dict(valid_metadata)
                invalid_metadata.pop(key)
                (directory / "training.json").write_text(
                    json.dumps(invalid_metadata),
                    encoding="utf-8",
                )
                item = arm_lab.run_catalog()["runs"][0]
                self.assertFalse(item["passed"])
                self.assertFalse(item["metrics"]["checkpoint_match"])

        self.write_training(directory, checkpoint_sha256)
        item = arm_lab.run_catalog()["runs"][0]
        self.assertTrue(item["passed"])
        self.assertTrue(item["metrics"]["checkpoint_match"])

    def test_video_requires_matching_video_and_source_receipt_hashes(self) -> None:
        directory, report = self.write_report()
        video = directory / "rollout.mp4"
        video.write_bytes(b"simulation video")

        self.write_receipt(directory, report, video_sha256="wrong")
        self.assertNotIn("video_path", arm_lab.run_catalog()["runs"][0])

        self.write_receipt(
            directory,
            report,
            source_hashes={"environment": "old", "pipeline": "old"},
        )
        self.assertNotIn("video_path", arm_lab.run_catalog()["runs"][0])

        self.write_receipt(directory, report)
        self.assertEqual(
            arm_lab.run_catalog()["runs"][0]["video_path"],
            "arm-reach-v1/run-a/rollout.mp4",
        )

    def test_report_model_hash_mismatch_fails_closed(self) -> None:
        self.write_report(model_sha256="wrong-model")
        item = arm_lab.run_catalog()["runs"][0]
        self.assertFalse(item["passed"])
        self.assertFalse(item["metrics"]["model_match"])

        report_path = self.runs / self.case_id / "run-a" / "teacher-evaluation.json"
        report = json.loads(report_path.read_text())
        report.pop("model_sha256")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        item = arm_lab.run_catalog()["runs"][0]
        self.assertFalse(item["passed"])
        self.assertFalse(item["metrics"]["model_match"])

    def test_video_receipt_binds_case_controller_model_and_checkpoint(self) -> None:
        directory, report = self.write_report()
        video = directory / "rollout.mp4"
        video.write_bytes(b"simulation video")

        invalid_values = {
            "case_id": "arms-co-carry-v1",
            "controller": "ppo_residual",
            "model_sha256": "wrong-model",
            "checkpoint_sha256": "wrong-checkpoint",
        }
        for key, value in invalid_values.items():
            with self.subTest(receipt_field=key):
                self.write_receipt(directory, report, **{key: value})
                item = arm_lab.run_catalog()["runs"][0]
                self.assertNotIn("video_path", item)

        self.write_receipt(directory, report)
        self.assertIn("video_path", arm_lab.run_catalog()["runs"][0])

    def test_catalog_rejects_controller_filename_mismatch(self) -> None:
        """Gap: a forged teacher controller can bypass PPO provenance by filename."""

        self.write_report(controller="teacher", report_name="evaluation.json")
        item = arm_lab.run_catalog()["runs"][0]
        self.assertFalse(item["passed"])

    def test_catalog_rejects_unprovenanced_bc_model(self) -> None:
        """Gap: a learned BC run passes without a model artifact or training receipt."""

        self.write_report(controller="bc")
        item = arm_lab.run_catalog()["runs"][0]
        self.assertFalse(item["passed"])

    def test_catalog_rejects_non_simulation_or_hardware_verified_claims(self) -> None:
        """Gap: report pass status ignores simulation_only and hardware_verified."""

        directory, report = self.write_report()
        report["simulation_only"] = False
        report["hardware_verified"] = True
        (directory / "teacher-evaluation.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        item = arm_lab.run_catalog()["runs"][0]
        self.assertFalse(item["passed"])

    def test_catalog_skips_non_object_report(self) -> None:
        """Gap: a JSON array report can crash /runs instead of being skipped."""

        directory = self.runs / self.case_id / "run-a"
        directory.mkdir(parents=True)
        (directory / "teacher-evaluation.json").write_text("[]", encoding="utf-8")
        self.assertEqual(arm_lab.run_catalog()["runs"], [])

    def test_catalog_skips_non_object_training_metadata(self) -> None:
        """Gap: a JSON array training record can crash /runs."""

        directory = self.runs / self.case_id / "run-a"
        directory.mkdir(parents=True)
        checkpoint = directory / "ppo-residual.zip"
        checkpoint.write_bytes(b"checkpoint")
        self.write_report(
            controller="ppo_residual",
            checkpoint_sha256=sha256(checkpoint),
        )
        (directory / "training.json").write_text("[]", encoding="utf-8")
        item = arm_lab.run_catalog()["runs"][0]
        self.assertFalse(item["passed"])

    def test_catalog_skips_non_object_render_receipt(self) -> None:
        """Gap: a JSON array render receipt can crash /runs."""

        directory, _ = self.write_report()
        (directory / "rollout.mp4").write_bytes(b"simulation video")
        (directory / "render-receipt.json").write_text("[]", encoding="utf-8")
        item = arm_lab.run_catalog()["runs"][0]
        self.assertNotIn("video_path", item)

    def test_catalog_rejects_symlinked_report_outside_runs(self) -> None:
        """Gap: catalog reports are not resolved and confined beneath RUNS."""

        outside = Path(self.temp.name).parent / f"{Path(self.temp.name).name}-outside.json"
        outside.write_text(
            json.dumps(
                {
                    "case_id": self.case_id,
                    "controller": "teacher",
                    "passed": True,
                    "source_hashes": self.expected_sources,
                    "model_sha256": self.model_sha256,
                    "results": [],
                }
            ),
            encoding="utf-8",
        )
        directory = self.runs / self.case_id / "run-a"
        directory.mkdir(parents=True)
        link = directory / "teacher-evaluation.json"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("symlinks unavailable")
        self.addCleanup(outside.unlink, missing_ok=True)

        self.assertEqual(arm_lab.run_catalog()["runs"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
