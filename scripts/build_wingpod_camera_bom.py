"""Generate a source-bound candidate BOM, not a fabrication release."""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

from microduck_arm_design.wingpod import PODS
from microduck_arm_design.wingpod_camera_soft import soft_camera_spec
from microduck_arm_experiments.tennis_return import TennisReturnEnv


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/microduck-arm-v1c/appearance/camera-v2-bom"
ARM_BOM = ROOT / "docs/microduck-arm-v1c/ENGINEERING-BOM.csv"
TENNIS = ROOT / "hardware/microduck-arm-v1c/experiments/tennis-return.json"
FIELDS = ["id", "group", "quantity", "unit", "item", "candidate_or_reference", "specification",
          "disposition", "status", "required_evidence", "source", "measured_mass_kg",
          "measured_com_m", "measured_inertia_kg_m2", "hardware_released"]


def item(identity, group, quantity, name, candidate, specification, disposition,
         evidence, source, unit="each", status="CANDIDATE_UNQUALIFIED"):
    return dict(zip(FIELDS, [identity, group, quantity, unit, name, candidate, specification,
                            disposition, status, evidence, source, None, None, None, False]))


def assemble():
    rows = []
    with ARM_BOM.open() as stream:
        inherited = list(csv.DictReader(stream))
    budget = [row for row in inherited if row["ID"].startswith("BUDGET-")]
    for row in inherited:
        if row in budget:
            continue
        quantity, *unit = row["Qty"].split(" ", 1)
        disposition = "NEW_ARM_PART_CANDIDATE"
        if row["ID"] in ("CTRL-BENCH", "PSU-BENCH"):
            disposition = "BENCH_ONLY"
        elif row["ID"] in ("PACK-ARM", "REG-SHARED"):
            disposition = "POWER_ARCHITECTURE_OPTION_NOT_ADDITIVE"
        elif row["ID"] == "F-BRANCH":
            disposition = "PROTECTION_TRADE_STUDY_NOT_SELECTED"
        rows.append(item("ARM-" + row["ID"], "arm", int(quantity), row["Function"], row["Candidate"],
                         row["Interface"], disposition, row["Required Evidence"],
                         str(ARM_BOM.relative_to(ROOT)) + "#" + row["ID"], unit[0] if unit else "each",
                         row["Status"]))
    by_id = {row["id"]: row for row in rows}
    wide = json.loads(TENNIS.read_text())["wide_gripper_candidate"]
    for side in ("L", "R"):
        jaw = by_id["ARM-JAW-" + side]
        jaw["item"] = f"Wide tennis jaw {side}: integral bridge/stem and pad carrier"
        jaw["specification"] = (
            f"75 mm open pad gap; bridge {wide['bridge_full_size_m']} m; "
            f"stem {wide['stem_full_size_m']} m; pad center X=48 mm; TCP=55 mm; "
            "existing 24 mm gear axis spacing; dimensions are simulation only"
        )
        jaw["candidate_or_reference"] = "Custom wide_candidate; replaces narrow 30 mm jaw; no extra jaw ordered"
        jaw["source"] = str(TENNIS.relative_to(ROOT)) + "#wide_gripper_candidate"
        jaw["required_evidence"] += "; wider-finger bending; ground reach; 58 g load and grip qualification"
    by_id["ARM-LINK-U"]["specification"] += "; rendered link feather is decorative, not another load-bearing link"
    by_id["ARM-LINK-F"]["specification"] += "; 55 + 50 + 55 = 160 mm geometric reach in tennis configuration"
    env = TennisReturnEnv(gripper="wide_candidate", release_mode="half_height")
    try:
        env.reset(0)
        model = env.robot.model
        actuators = [model.actuator(index).name for index in range(model.nu)]
        counts = collections.Counter()
        for index in range(model.ngeom):
            geom = model.geom(index)
            if geom.type[0] == 7 and geom.group[0] == 2 and not geom.name.startswith("arm_"):
                counts[model.mesh(geom.dataid[0]).name] += 1
    finally:
        env.close()
    for mesh, count in sorted(counts.items()):
        if mesh in ("xl330", "np_f970"):
            continue
        rows.append(item("BASE-" + mesh, "microduck_base", count, mesh,
                         "Existing MicroDuck part; matching source mesh (not a machining drawing)",
                         "Counted once per visual mesh instance; collision copies excluded",
                         "REUSE_EXISTING_ASSEMBLY_SUBPART",
                         "Inspect installed part and drawing revision; dimensions/material/fasteners need physical audit",
                         "rlx/rlx/mjlab_microduck/robot/microduck/assets/" + mesh + ".stl"))
    rows.append(item("BASE-LEG-MOTORS", "microduck_base", 10, "Five leg servos per leg",
                     "Installed DYNAMIXEL XL330 family; exact M288/M077 labels to inspect",
                     "hip yaw / hip roll / hip pitch / knee / ankle per leg; existing IDs and calibration retained",
                     "REUSE", "Photo each label; record ID/model/firmware; do not infer gearbox SKU from xl330.stl",
                     "rlx/rlx/mjlab_microduck/robot/microduck/robot_allcollisions_backlash.xml"))
    base_parts = [
        ("COMPUTE", "Onboard compute", "Radxa ZERO 3W reference", "Retain installed RAM/storage variant; CSI/USB/power budget must be audited"),
        ("STORAGE", "Boot storage", "Existing eMMC or microSD", "Use installed boot medium; not two new storage devices"),
        ("CONTROL-BOARD", "Motor interface and base power board", "Existing MicroDuck board; exact revision TBD", "Retain leg bus; no assumed spare arm current capacity or ports"),
        ("IMU", "Body IMU function", "Existing board-integrated or discrete sensor; exact SKU TBD", "Reuse existing physical sensor; simulation IMU site is not a part number"),
        ("BATTERY", "Main battery pack", "Installed pack label required; np_f970 is only a source mesh name", "Record chemistry/cell count/voltage/capacity/mass; no guessed battery substitution"),
        ("CHARGER", "Battery-compatible charger", "Existing qualified matching charger", "Off-robot charging; exact pack compatibility required"),
        ("PACK-PROTECTION", "Pack protection/BMS function", "Installed protection; topology TBD", "May be integrated in pack; do not buy a duplicate BMS by default"),
        ("BASE-HARNESS", "Leg and onboard wiring set", "Existing keyed cables/connectors", "Retain known pinout; inspect insulation/flex/strain relief"),
        ("BASE-FASTENERS", "Base screws/spacers/retainers set", "Reuse installed fasteners", "Exact thread/length/count must be measured; meshes do not define screw procurement"),
    ]
    for identity, name, candidate, specification in base_parts:
        rows.append(item("BASE-" + identity, "microduck_electronics", 1, name, candidate, specification,
                         "REUSE_OR_ASSEMBLY_INCLUDED", "Installed inventory and electrical/mechanical qualification",
                         "microduck/docs/project/media-bringup.md; local physical inventory required",
                         "set" if "set" in name else "function" if "function" in name else "each"))
    for pod in PODS:
        rows.append(item("SHELL-" + pod.name, "appearance", 1, pod.name + " cover envelope",
                         "Custom WingPod cream shell; material/process unselected",
                         f"body={pod.body}; local center={pod.center} m; XYZ envelope="
                         f"{[round(value * 2000, 3) for value in pod.half_size]} mm; fillet={pod.radius * 1000:g} mm",
                         "COSMETIC_ENVELOPE_NOT_ADDITIVE_SOLID", "Wall thickness; clips; cable exits; clearances; mass/inertia; cooling; mount strength",
                         "microduck_arm_design/wingpod.py#PODS", status="RENDER_ONLY"))
    camera_rows = [
        ("SENSORS", 2, "Active left/right RGB camera modules incl. lens", "Exact miniature sensor modules UNSELECTED", "13 mm optical center spacing design target; matched optics preferred; voltage/interface not selected", "DUAL_EYE_TARGET_BLOCKED", "Actual module drawings; PCB/lens fit; usable FOV; bandwidth; synchronization if stereo requested"),
        ("BEZEL", 1, "Cream twin-eye housing / peach cheeks / honey beak", "Custom camera chest cover; soft appearance revision", "28 mm wide x 17 mm high x 5 mm deep outer envelope; not usable internal volume; colors are render-only", "RENDER_ONLY", "PCB/lens/connector stack and cable bend envelope; mounting and removable access; qualify cosmetic finish"),
        ("MOUNT", 1, "Camera internal carrier and fixed chest mount", "Custom removable nonstructural carrier", "Fixed to arm_mount chest; must not load moving gripper or obstruct shoulder", "NEW_CANDIDATE", "Actual fastener datums; no chassis-shell bending load; retention; mass and clearances"),
        ("LENS-TRIM", 2, "Sage-gray lens trim / light baffle", "Custom cosmetic ring; no extra camera lens", "8 mm visual ring outer diameter; avoid vignette/reflections", "COSMETIC", "Optical aperture; lens movement; glare test; no unsupported IP rating"),
        ("LED", 1, "Amber status LED", "Low-current LED; exact MPN TBD", "Displayed amber bar is a render cue; real drive current must be chosen", "NEW_CANDIDATE", "LED voltage/current/brightness; privacy semantics; avoid optical contamination"),
        ("LED-DRIVER", 1, "LED current-limit/driver circuit", "Resistor/driver to match chosen logic rail", "Do not directly drive unknown load from GPIO; value and topology TBD", "NEW_CANDIDATE", "Schematic; GPIO limits; resistor power; startup and fault behavior"),
        ("LIGHTPIPE", 1, "Amber indicator light pipe", "Custom diffuser", "Matches rendered indicator; thickness and material TBD", "NEW_CANDIDATE", "Optical cross-talk; retention; fit"),
        ("DATA-CABLES", 2, "Camera data cable assemblies", "Matched FFC or USB assemblies; not interchangeable", "One per selected module; pitch/pin count/contact side/length TBD", "DUAL_EYE_TARGET_BLOCKED", "Vendor pinout; CSI lane routing or USB signal integrity; strain relief; minimum bend radius"),
        ("BRIDGE", 1, "Dual-camera host interface function", "UNSELECTED: synchronized bridge/USB stereo assembly if required", "Do not split one CSI input with a passive Y cable; assembly may include this function", "ARCHITECTURE_NOT_SELECTED", "Radxa driver support; simultaneous capture; timestamps; encoder/load test; calibration"),
        ("LOGIC-POWER", 1, "Camera protected logic-power branch", "Existing logic regulator if qualified; otherwise sized replacement TBD", "Camera supply by selected interface; do not connect to motor rail or raw battery", "CONDITIONAL_FUNCTION", "Voltage/current/transients; branch fault containment; no compute brownout; regulator heating"),
        ("HARDWARE", 1, "Camera screws/inserts/grommets/strain-relief set", "Custom matched hardware; size TBD", "No drilling or screw lengths released before real-board drawing", "NEW_CANDIDATE", "Count/threads/length; tool access; no PCB contact or pierced cables"),
        ("BENCH-IMX219", 1, "MicroDuck-compatible single-camera bench reference", "Raspberry Pi Camera Module 2 / Sony IMX219", "Reference only; not claimed to fit chest or provide two live eyes", "BENCH_ALTERNATIVE_NOT_IN_DUAL_TARGET", "Validate board/FFC/OS overlay; dimensional fit; do not order two as a proven stereo system"),
    ]
    for identity, quantity, name, candidate, specification, disposition, evidence in camera_rows:
        rows.append(item("CAM-" + identity, "camera_eyes", quantity, name, candidate, specification,
                         disposition, evidence, "microduck_arm_design/wingpod_camera.py; microduck/docs/project/media-bringup.md",
                         "set" if identity == "HARDWARE" else "function" if identity in ("BRIDGE", "LOGIC-POWER") else "each"))
    extras = [
        ("TELEMETRY", "Voltage/current/temperature instrumentation", "Calibrated measurement equipment; onboard sensor selection TBD", "Branch current + pack/logic/arm voltage + regulator temperature"),
        ("POWER-PROTECTION", "Transient/reverse-current protection and local decoupling", "TVS/blocking/clamp/capacitor topology TBD", "Coordinate with measured motor regeneration and converter response; no guessed fuse or capacitor value"),
        ("FIXTURE", "Nonconductive bench fixture and fall restraint", "Custom support allowing safe leg/arm tests", "Provide catch restraint without mistaking tether-assisted stability for free-base success"),
        ("CAM-CALIBRATION", "Camera calibration target", "Printed/measured checkerboard or Charuco target", "Intrinsic/extrinsic tests; verified square size; timing and reprojection logs"),
        ("BALL", "Yellow tennis ball", "Measured classroom test object", "Simulation nominal diameter 67 mm / mass 58 g; this is not a qualified real payload rating"),
        ("BIN", "Low classroom bin", "Custom anchored fixture; not a household garbage can", "180 x 180 mm inner plan; 100 mm walls; 4 mm wall/bottom in simulation"),
    ]
    for identity, name, candidate, specification in extras:
        rows.append(item("TEST-" + identity, "support_and_validation", 1, name, candidate, specification,
                         "BENCH_OR_FIXTURE" if identity != "POWER-PROTECTION" else "ELECTRICAL_FUNCTION_UNSELECTED",
                         "Specify equipment/ratings; review test procedure; record measured evidence", str(TENNIS.relative_to(ROOT)), "set"))
    return rows, budget, actuators, counts


def build(output):
    rows, budget, actuators, counts = assemble()
    assert len({row["id"] for row in rows}) == len(rows)
    assert len(actuators) == 15 and counts["xl330"] == 10
    assert all(row["quantity"] > 0 and row["hardware_released"] is False for row in rows)
    assert sum(row["quantity"] for row in rows if row["id"] in {f"ARM-M{index}" for index in range(1, 6)}) == 5
    output.mkdir(parents=True, exist_ok=True)
    with (output / "BOM.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    sources = [ARM_BOM, TENNIS, ROOT / "microduck_arm_design/wingpod.py",
               ROOT / "microduck_arm_design/wingpod_camera.py",
               ROOT / "microduck_arm_design/wingpod_camera_soft.py", Path(__file__).resolve()]
    document = {
        "design": "WingPod Camera v2 Soft / tennis-wide-gripper", "date": "2026-09-13",
        "status": "CANDIDATE_ENGINEERING_BOM_NOT_PURCHASE_OR_FABRICATION_RELEASE",
        "scope": "One converted existing MicroDuck; reuse rows are not new purchases; options/functions are not additive",
        "hardware_released": False, "parts": rows, "actuator_order": actuators,
        "arm_candidate_A": ["XL330-M288-T"] * 5,
        "arm_candidate_B": ["XL330-M288-T"] * 3 + ["XL330-M077-T"] * 2,
        "selected_arm_candidate": "A simulation default; hardware selection pending qualification",
        "camera": soft_camera_spec(), "inherited_budget_not_parts": budget,
        "budget_limit_note": "160 g arm budget is not a measured wide-gripper + shell + camera total",
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    }
    (output / "BOM.json").write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    text = ["# Full candidate parts list / 完整候选物料表", "",
            "Generated from local engineering BOM and model. NOT a released purchase order. Units and options matter.",
            "源自工程BOM及仿真模型；不是已放行采购单。复用件、可选架构、功能项不可全部相加。", "",
            "| ID | Qty | Unit | Item | Candidate/reference | Specification | Disposition |",
            "| --- | ---: | --- | --- | --- | --- | --- |"]
    for row in rows:
        text.append("| " + " | ".join(str(row[key]).replace("|", "/").replace("\n", " ") for key in
                                       ("id", "quantity", "unit", "item", "candidate_or_reference", "specification", "disposition")) + " |")
    (output / "PARTS.md").write_text("\n".join(text) + "\n")
    summary = {"part_rows": len(rows), "groups": dict(collections.Counter(row["group"] for row in rows)),
               "leg_actuators": 10, "arm_actuators": 5, "total_actuators": 15,
               "target_camera_modules": 2, "unique_ids": True, "hardware_released": False}
    (output / "consistency-check.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.output)
