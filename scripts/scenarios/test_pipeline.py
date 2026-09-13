from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))

import pipeline


SCENARIOS = ("dance", "swing", "running", "stilts", "backflip")


def option_value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


class PlanTests(unittest.TestCase):
    def make_args(self, scenario: str, output: Path, *extra: str):
        return pipeline.parse_args(
            [scenario, "--run-name", f"{scenario}-test", "--output-dir", str(output), *extra]
        )

    def test_all_default_plans_finish_with_export_verify_evaluate_render(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scenario in SCENARIOS:
                with self.subTest(scenario=scenario):
                    plan = pipeline.make_plan(
                        self.make_args(scenario, root / f"{scenario}-output")
                    )
                    self.assertEqual(
                        [step["name"] for step in plan["commands"][-4:]],
                        ["export", "verify-policy", "evaluate", "render"],
                    )

    def test_custom_recipe_seed_is_preserved_unless_explicitly_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = root / "recipe.json"
            recipe.write_text(json.dumps({"experimentId": "dance", "seed": 19}))
            for arguments, expected in (([], 19), (["--seed", "23"], 23)):
                plan = pipeline.make_plan(self.make_args(
                    "dance", root / f"output-{expected}", "--recipe-json", str(recipe), *arguments,
                ))
                training = next(step for step in plan["commands"] if step["name"] == "train-1")
                self.assertEqual(option_value(training["argv"], "--seed"), str(expected))

    def test_full_evaluates_skill_and_smoke_evaluates_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scenario in SCENARIOS:
                for profile, expected in (("full", "skill"), ("smoke", "pipeline")):
                    with self.subTest(scenario=scenario, profile=profile):
                        plan = pipeline.make_plan(
                            self.make_args(
                                scenario,
                                root / f"{scenario}-{profile}",
                                "--profile",
                                profile,
                            )
                        )
                        evaluation = next(
                            step for step in plan["commands"] if step["name"] == "evaluate"
                        )
                        self.assertEqual(
                            option_value(evaluation["argv"], "--evaluation-mode"), expected
                        )

    def test_running_and_stilts_chain_fresh_bootstrap_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scenario in ("running", "stilts"):
                with self.subTest(scenario=scenario):
                    output = root / f"{scenario}-fresh"
                    plan = pipeline.make_plan(self.make_args(scenario, output))
                    training = [
                        step
                        for step in plan["commands"]
                        if step["name"].startswith("train-")
                    ]
                    self.assertEqual([step["name"] for step in training], ["train-1", "train-2"])
                    stage_output = output.resolve() / "stages" / "stage-1"
                    self.assertEqual(
                        option_value(training[0]["argv"], "--output-dir"), str(stage_output)
                    )
                    self.assertNotIn("--init-from", training[0]["argv"])
                    self.assertEqual(
                        option_value(training[1]["argv"], "--init-from"),
                        str(stage_output / f"{scenario}.safetensors"),
                    )
                    self.assertEqual(
                        option_value(training[1]["argv"], "--output-dir"), str(output.resolve())
                    )

    def test_swing_initialization_is_owned_by_fresh_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ignored = root / "ignored-runs" / "swing"
            ignored.mkdir(parents=True)
            (ignored / "swing.safetensors").write_bytes(b"ignored")
            output = root / "fresh swing output"

            plan = pipeline.make_plan(self.make_args("swing", output))
            initialization, training = plan["commands"][:2]

            self.assertEqual(initialization["name"], "initialize-swing")
            self.assertEqual(
                option_value(initialization["argv"], "--output"),
                str(output.resolve() / "initialization"),
            )
            self.assertEqual(
                option_value(training["argv"], "--init-from"),
                str(output.resolve() / "initialization" / "swing.safetensors"),
            )
            self.assertNotIn(str(ignored), " ".join(initialization["argv"] + training["argv"]))

    def test_custom_recipe_rejects_scenario_mismatch_unknown_fields_and_missing_init(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                (
                    "mismatch.json",
                    {"experimentId": "running"},
                    "recipe JSON must be an object for this scenario",
                ),
                (
                    "unknown.json",
                    {"experimentId": "dance", "surprise": True},
                    "unsupported recipe fields",
                ),
                (
                    "continuation.json",
                    {"experimentId": "running", "resumeFromCheckpoint": True},
                    "requires an explicit --init-from",
                ),
            )
            for filename, recipe, message in cases:
                with self.subTest(filename=filename):
                    path = root / filename
                    path.write_text(json.dumps(recipe))
                    scenario = "dance" if filename != "continuation.json" else "running"
                    args = self.make_args(
                        scenario,
                        root / f"output-{filename}",
                        "--recipe-json",
                        str(path),
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        pipeline.make_plan(args)

    def test_custom_recipe_with_frozen_normalization_requires_init_from(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = root / "frozen.json"
            recipe.write_text(
                json.dumps(
                    {
                        "experimentId": "dance",
                        "freezeObservationNormalization": True,
                    }
                )
            )
            args = self.make_args(
                "dance",
                root / "frozen-output",
                "--recipe-json",
                str(recipe),
            )
            with self.assertRaisesRegex(
                ValueError, "frozen observation normalization requires"
            ):
                pipeline.make_plan(args)

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "collision"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "output already exists"):
                pipeline.make_plan(self.make_args("backflip", output))


class ShellEntrypointTests(unittest.TestCase):
    def test_all_wrappers_support_help_and_dry_run_from_space_cwd(self):
        root = Path(__file__).resolve().parents[2]
        environment = {**os.environ, "MICRODUCK_STUDIO_PYTHON_DIRECT": sys.executable}
        with tempfile.TemporaryDirectory(prefix="microduck pipeline ") as directory:
            cwd = Path(directory)
            for scenario in SCENARIOS:
                script = root / "scripts" / f"train-{scenario}.sh"
                with self.subTest(scenario=scenario, mode="help"):
                    result = subprocess.run(
                        ["bash", str(script), "--help"],
                        cwd=cwd,
                        env=environment,
                        text=True,
                        capture_output=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("--dry-run", result.stdout)

                with self.subTest(scenario=scenario, mode="dry-run"):
                    output = cwd / f"{scenario} output"
                    result = subprocess.run(
                        [
                            "bash",
                            str(script),
                            "--profile",
                            "smoke",
                            "--run-name",
                            f"{scenario}-shell-test",
                            "--output-dir",
                            str(output),
                            "--dry-run",
                        ],
                        cwd=cwd,
                        env=environment,
                        text=True,
                        capture_output=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    plan = json.loads(result.stdout)
                    self.assertEqual(plan["scenario"], scenario)
                    self.assertEqual(plan["profile"], "smoke")
                    self.assertEqual(plan["output"], str(output.resolve()))
                    self.assertFalse(output.exists())


class ExecuteTests(unittest.TestCase):
    def make_plan(self, output: Path, profile: str = "full") -> dict:
        commands = [
            {
                "name": "train-1",
                "argv": ["train"],
                "recipe": {"totalTimesteps": 32},
            },
            {"name": "export", "argv": ["export"]},
            {"name": "verify-policy", "argv": ["verify-policy"]},
            {"name": "evaluate", "argv": ["evaluate"]},
            {"name": "render", "argv": ["render"]},
        ]
        return {
            "scenario": "dance",
            "profile": profile,
            "output": str(output),
            "commands": commands,
            "input_hashes": {},
            "render_seconds": 1.0,
        }

    def command_runner(self, output: Path, evaluation: dict):
        def run(step: dict, _output: Path):
            if step["name"] == "export":
                (output / "dance.onnx").write_bytes(b"policy")
            results = {
                "train-1": {"trained_steps": 32},
                "export": {"exported": True},
                "verify-policy": {"verified": True},
                "evaluate": evaluation,
                "render": {"rendered": True},
            }
            return (2 if step["name"] == "evaluate" and not evaluation["passed"] else 0,
                    results[step["name"]])

        return run

    def test_failed_full_skill_still_renders_and_returns_two(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed-skill"
            evaluation = {
                "passed": False,
                "pipeline_passed": True,
                "skill_status": "failed",
            }
            calls = []
            runner = self.command_runner(output, evaluation)

            def recording_runner(step, destination):
                result = runner(step, destination)
                if step["name"] == "evaluate":
                    evaluation["source_sha256"] = pipeline.digest(output / "dance.onnx")
                calls.append(step["name"])
                return result

            with (
                mock.patch.object(pipeline, "preflight"),
                mock.patch.object(pipeline, "run_command", side_effect=recording_runner),
                mock.patch.object(
                    pipeline,
                    "verify_media",
                    return_value={"duration": 1.0},
                ),
            ):
                code = pipeline.execute(self.make_plan(output))

            self.assertEqual(code, 2)
            self.assertEqual(calls[-1], "render")
            report = pipeline.read_json(output / "pipeline.json")
            self.assertEqual(report["status"], "evaluation_failed")
            self.assertEqual(report["skill_status"], "failed")

    def test_command_error_halts_later_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "command-error"
            calls = []

            def run(step, _output):
                calls.append(step["name"])
                if step["name"] == "verify-policy":
                    raise RuntimeError("bad verification")
                if step["name"] == "export":
                    (output / "dance.onnx").write_bytes(b"policy")
                return 0, {"trained_steps": 32}

            with (
                mock.patch.object(pipeline, "preflight"),
                mock.patch.object(pipeline, "run_command", side_effect=run),
                mock.patch.object(pipeline, "verify_media") as verify_media,
            ):
                code = pipeline.execute(self.make_plan(output))

            self.assertEqual(code, 1)
            self.assertEqual(calls, ["train-1", "export", "verify-policy"])
            verify_media.assert_not_called()
            self.assertEqual(
                pipeline.read_json(output / "pipeline.json")["status"], "failed"
            )

    def test_missing_training_evidence_fails_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "missing-training-evidence"
            calls = []

            def run(step, _output):
                calls.append(step["name"])
                return 0, {}

            with (
                mock.patch.object(pipeline, "preflight"),
                mock.patch.object(pipeline, "run_command", side_effect=run),
                mock.patch.object(pipeline, "verify_media") as verify_media,
            ):
                code = pipeline.execute(self.make_plan(output))

            self.assertEqual(code, 1)
            self.assertEqual(calls, ["train-1"])
            verify_media.assert_not_called()
            report = pipeline.read_json(output / "pipeline.json")
            self.assertEqual(report["status"], "failed")
            self.assertIn("training stopped", report["error"])

    def test_mismatched_policy_evidence_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mismatched-evidence"
            evaluation = {
                "passed": True,
                "pipeline_passed": True,
                "skill_status": "passed",
                "source_sha256": "not-the-policy-digest",
            }
            with (
                mock.patch.object(pipeline, "preflight"),
                mock.patch.object(
                    pipeline,
                    "run_command",
                    side_effect=self.command_runner(output, evaluation),
                ),
                mock.patch.object(
                    pipeline,
                    "verify_media",
                    return_value={"duration": 1.0},
                ),
            ):
                code = pipeline.execute(self.make_plan(output))

            self.assertEqual(code, 1)
            report = pipeline.read_json(output / "pipeline.json")
            self.assertEqual(report["status"], "failed")
            self.assertIn("evaluation does not match", report["error"])

    def test_successful_smoke_does_not_claim_full_skill(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke-success"
            evaluation = {
                "passed": True,
                "pipeline_passed": True,
                "skill_status": "not_assessed",
            }
            runner = self.command_runner(output, evaluation)

            def run(step, destination):
                result = runner(step, destination)
                if step["name"] == "evaluate":
                    evaluation["source_sha256"] = pipeline.digest(output / "dance.onnx")
                return result

            with (
                mock.patch.object(pipeline, "preflight"),
                mock.patch.object(pipeline, "run_command", side_effect=run),
                mock.patch.object(
                    pipeline,
                    "verify_media",
                    return_value={"duration": 1.0},
                ),
            ):
                code = pipeline.execute(self.make_plan(output, profile="smoke"))

            self.assertEqual(code, 0)
            report = pipeline.read_json(output / "pipeline.json")
            self.assertEqual(report["status"], "smoke_passed")
            self.assertEqual(report["skill_status"], "not_assessed")
            self.assertNotEqual(report["status"], "skill_passed")


if __name__ == "__main__":
    unittest.main()
