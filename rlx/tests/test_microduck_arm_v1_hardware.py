import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "hardware" / "microduck-arm-v1"
SPEC = PACKAGE / "spec" / "design.json"
GENERATOR = PACKAGE / "cad" / "generate_engineering.py"
SCAD = PACKAGE / "cad" / "microduck_arm_v1_b.scad"
DOCS = ROOT / "docs" / "microduck-arm-v1"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_canonical_v1_b_contract():
    spec = load_json(SPEC)
    arm = spec["arm"]

    assert spec["design_id"] == "microduck-single-arm-v1-b"
    assert spec["status"] == "simulation_prototype_not_fabrication_or_hardware_release"
    assert arm["pose_joint_count"] == 4
    assert arm["actuator_count"] == 5
    assert arm["motor_sku"] == "XL330-M288-T"
    assert arm["motor_mass_kg"] == 0.018
    assert arm["motor_envelope_m"] == [0.020, 0.026, 0.034]
    assert arm["link_lengths_m"] == [0.055, 0.050, 0.030]
    assert arm["gripper_pivot_half_spacing_m"] == 0.012
    assert arm["gripper_finger_length_m"] == 0.030
    assert arm["gripper_max_closure_rad"] == 0.32
    assert arm["joint_ranges_rad"][4] == [-0.32, 0]
    assert arm["servo_centers_local_m"] == {
        "yaw": [0, 0, -0.025],
        "shoulder": [0, 0, 0.018],
        "elbow": [0.045, 0, 0.020],
        "wrist": [0.025, 0, 0.020],
        "gripper": [0.007, 0, 0.022],
    }
    assert arm["required_continuous_torque_margin"] == 3.0
    assert arm["gravity_compensation"] == "not_implemented_or_assumed"


def test_replacement_masses_are_explicit_unmeasured_placeholders():
    spec = load_json(SPEC)
    assumptions = spec["replacement_mass_assumptions"]

    assert spec["replacement_mass_status"].startswith("unmeasured_")
    assert assumptions["relocated_electronics"]["mass_kg"] == 0.070
    assert assumptions["dedicated_arm_pack"]["mass_kg"] == 0.080
    assert assumptions["power_and_communications"]["mass_kg"] == 0.025
    assert assumptions["sensor_shell"]["mass_kg"] == 0.008
    assert assumptions["sensor_shell"]["pos_m"] == [-0.020, 0, 0.059]

    bom = (DOCS / "BOM.md").read_text(encoding="utf-8")
    assert "unmeasured_placeholder" in bom
    assert "not BOM facts" in bom
    assert "70 g" in bom
    assert "80 g" in bom
    assert "25 g" in bom
    assert "8 g" in bom


def test_generator_outputs_parseable_review_artifacts():
    with tempfile.TemporaryDirectory() as temporary:
        output_root = Path(temporary)
        completed = subprocess.run(
            [
                sys.executable,
                str(GENERATOR),
                "--spec",
                str(SPEC),
                "--output-root",
                str(output_root),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        receipt = json.loads(completed.stdout)

        assert receipt["status"] == "generated_review_artifacts"
        assert receipt["fabrication_release"] is False
        assert receipt["canonical_spec_sha256"] == (
            "57aed79ec22c5e651950271e089a578766dc49e7a30961353c9d3a308688a50d"
        )
        expected = {
            "cad/design_parameters.scad",
            "cad/mechanical-dimensions.svg",
            "cad/manufacture-check.json",
            "electrical/wiring.svg",
            "electrical/circuit.svg",
            "electrical/netlist.json",
        }
        if shutil.which("rsvg-convert"):
            expected.update(
                {
                    "cad/mechanical-dimensions.pdf",
                    "electrical/wiring.pdf",
                    "electrical/circuit.pdf",
                }
            )
            assert receipt["pdf_export"] == "generated"
        else:
            assert receipt["pdf_export"] == "rsvg-convert_unavailable"
        assert set(receipt["files"]) == expected

        for relative in (
            "cad/mechanical-dimensions.svg",
            "electrical/wiring.svg",
            "electrical/circuit.svg",
        ):
            tree = ET.parse(output_root / relative)
            assert tree.getroot().tag.endswith("svg")
            pdf_path = (output_root / relative).with_suffix(".pdf")
            if shutil.which("rsvg-convert"):
                assert pdf_path.read_bytes().startswith(b"%PDF")

        mechanical = (
            output_root / "cad" / "mechanical-dimensions.svg"
        ).read_text(encoding="utf-8")
        assert "55 mm shoulder-elbow nominal" in mechanical
        assert "50 mm elbow-wrist nominal" in mechanical
        assert "30 mm wrist-TCP nominal" in mechanical
        assert "left q [-0.32,0], right q [0,0.32] rad" in mechanical
        assert "right = -left; max closure 0.32 rad" in mechanical
        assert "gripper [7,0,22]" in mechanical
        assert "Kinematic sketch omits motor boxes" in mechanical
        assert "mounting interface" not in mechanical.lower() or "No horn pilot" in mechanical

        invalid = load_json(SPEC)
        del invalid["arm"]["motor_envelope_m"]
        invalid["motor"] = {"motor_envelope_m": [0.020, 0.026, 0.034]}
        invalid_path = output_root / "invalid-alias-spec.json"
        invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
        rejected = subprocess.run(
            [
                sys.executable,
                str(GENERATOR),
                "--spec",
                str(invalid_path),
                "--output-root",
                str(output_root / "invalid-output"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode == 2
        assert json.loads(rejected.stdout)["status"] == "blocked"


def test_scad_models_rotary_geared_jaws_and_no_released_hole_pattern():
    scad = SCAD.read_text(encoding="utf-8")

    assert "include <design_parameters.scad>" in scad
    assert "gripper_left_q_rad = -0.16" in scad
    assert "right_q_rad = -left_q_rad" in scad
    assert "gripper_left_range_rad" in scad
    assert "gripper_right_range_rad" in scad
    assert "jaw_center_offset_mm" in scad
    assert "jaw_length_mm" in scad
    assert "jaw_coupling_ratio" in scad
    assert "equal gears ratio -1" in scad
    assert "servo_center_wrist_local_mm" in scad
    assert "servo_center_gripper_local_mm" in scad
    assert "M5 ROTARY GRIPPER - ABOVE HAND" in scad
    assert "overall packaging and table clearance unverified" in scad

    parameters = (PACKAGE / "cad" / "design_parameters.scad").read_text(
        encoding="utf-8"
    )
    assert "servo_center_elbow_local_mm = [45, 0, 20]" in parameters
    assert "servo_center_wrist_local_mm = [25, 0, 20]" in parameters
    assert "servo_center_gripper_local_mm = [7, 0, 22]" in parameters
    assert (
        "57aed79ec22c5e651950271e089a578766dc49e7a30961353c9d3a308688a50d"
        in parameters
    )
    assert "difference(" not in scad
    assert "bolt_circle" not in scad
    assert "mount_hole" not in scad


def test_netlist_is_complete_but_unqualified_and_separate():
    netlist = load_json(PACKAGE / "electrical" / "netlist.json")

    assert netlist["notice"] == "ENGINEERING PROTOTYPE NOT FABRICATION RELEASE"
    assert netlist["release"] == {
        "fabrication_release": False,
        "electrical_release": False,
        "hardware_acceptance": "not_run",
    }
    assert netlist["architecture"]["actuator_count"] == 5
    assert netlist["architecture"]["arm_voltage_domain_v"] == 5.0
    assert (
        netlist["architecture"]["arm_power_domain"]
        == "separate_from_original_microduck_positive"
    )
    assert netlist["architecture"]["mobile_interface"] == "TBD"

    names = {net["name"] for net in netlist["nets"]}
    assert {
        "ARM_5V_SOURCE",
        "ARM_5V_FUSED",
        "ARM_5V_ENABLED",
        "ARM_GND",
        "TTL_DATA",
        "M1_VDD",
        "M2_VDD",
        "M3_VDD",
        "M4_VDD",
        "M5_VDD",
    }.issubset(names)
    assert "ORIGINAL_MICRODUCK_BATTERY:POS" in netlist["explicit_no_connects"]
    assert "ORIGINAL_MICRODUCK_BODY_BUS:VDD" in netlist["explicit_no_connects"]
    assert "USB:VBUS_TO_ARM_5V" in netlist["explicit_no_connects"]
    assert netlist["components"]["BAT1"]["part_number"] is None
    assert netlist["components"]["BMS1"]["current_rating"] is None
    assert netlist["components"]["F_MAIN"]["rating"] is None
    assert netlist["components"]["HUB1"]["status"] == "unresolved_not_cad_ready"
    assert "intentionally unspecified" in netlist["internal_source_topology"]


def test_manufacture_check_fails_closed():
    with tempfile.TemporaryDirectory() as temporary:
        completed = subprocess.run(
            [
                sys.executable,
                str(GENERATOR),
                "--spec",
                str(SPEC),
                "--output-root",
                temporary,
                "--manufacture-check",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    assert completed.returncode == 3
    result = json.loads(completed.stdout)
    assert result["status"] == "FAIL_CLOSED"

    checked_in = load_json(PACKAGE / "cad" / "manufacture-check.json")
    assert checked_in["fabrication_release"] is False
    assert checked_in["manufacture_check"] == "FAIL_CLOSED"
    assert any(
        "continuous torque evidence" in gate
        for gate in checked_in["blocking_gates"]
    )


def test_bilingual_sop_and_acceptance_make_no_hardware_pass_claim():
    readme = (DOCS / "README.md").read_text(encoding="utf-8")
    sop = (DOCS / "SOP.md").read_text(encoding="utf-8")
    acceptance = (DOCS / "HARDWARE-ACCEPTANCE.md").read_text(encoding="utf-8")
    flow_path = DOCS / "engineering-flow.svg"
    build_script = DOCS / "build-book.sh"
    flow = flow_path.read_text(encoding="utf-8")
    builder = build_script.read_text(encoding="utf-8")

    assert "Fixed-root bench mode" in readme
    assert "Floating mode" in readme
    assert "固定根部台架模式" in readme
    assert "浮动模式" in readme
    assert "build-book.sh" in readme
    assert "does not claim completed gait, hardware operation, or UI" in readme
    assert "no test in this document has been run or passed" in sop
    assert "未执行" in sop
    assert "NOT RUN - FAIL CLOSED" in acceptance
    assert "Shoulder gravity margin" in acceptance
    assert "BLOCKED - EVIDENCE MISSING" in acceptance

    assert ET.parse(flow_path).getroot().tag.endswith("svg")
    for label in (
        "Canonical spec",
        "CAD / BOM / MJCF",
        "Checks",
        "Scripted teacher",
        "Behavior cloning",
        "Residual PPO",
        "ONNX export",
        "Hardware gates",
        "BLOCKED / FAIL CLOSED",
    ):
        assert label in flow

    assert "RESULTS.en.md" in builder
    assert "RESULTS.zh-CN.md" in builder
    assert "pandoc" in builder
    assert "xelatex" in builder
    with tempfile.TemporaryDirectory() as temporary:
        isolated_docs = Path(temporary) / "docs" / "microduck-arm-v1"
        isolated_docs.mkdir(parents=True)
        isolated_builder = isolated_docs / "build-book.sh"
        shutil.copy2(build_script, isolated_builder)
        missing_results = subprocess.run(
            ["bash", str(isolated_builder)],
            check=False,
            capture_output=True,
            text=True,
        )
    assert missing_results.returncode == 3
    assert "required leader-owned results file is missing" in missing_results.stderr
