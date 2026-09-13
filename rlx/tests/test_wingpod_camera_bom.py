import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.build_wingpod_camera_bom import build


def test_camera_bom_matches_model_and_preserves_unknowns(tmp_path):
    build(tmp_path)
    bom = json.loads((tmp_path / "BOM.json").read_text())
    parts = {part["id"]: part for part in bom["parts"]}
    assert len(parts) == len(bom["parts"])
    assert parts["BASE-LEG-MOTORS"]["quantity"] == 10
    assert sum(parts[f"ARM-M{number}"]["quantity"] for number in range(1, 6)) == 5
    assert len(bom["actuator_order"]) == 15
    assert parts["CAM-SENSORS"]["quantity"] == 2
    assert bom["camera"]["hardware_model"] is None
    assert bom["camera"]["version"] == "WingPod Camera v2 Soft"
    assert "Cream" in parts["CAM-BEZEL"]["item"]
    assert "Sage-gray" in parts["CAM-LENS-TRIM"]["item"]
    assert all(part["measured_mass_kg"] is None for part in parts.values())
    assert all(part["measured_com_m"] is None for part in parts.values())
    assert all(part["measured_inertia_kg_m2"] is None for part in parts.values())
    assert all(part["hardware_released"] is False for part in parts.values())
    assert "np_f970" not in parts
    assert "source mesh name" in parts["BASE-BATTERY"]["candidate_or_reference"]
    assert "POWER_ARCHITECTURE_OPTION" in parts["ARM-PACK-ARM"]["disposition"]
    assert "BENCH_ALTERNATIVE" in parts["CAM-BENCH-IMX219"]["disposition"]
    assert not any(identity.startswith("ARM-BUDGET-") for identity in parts)
    for side in ("L", "R"):
        assert "75 mm" in parts[f"ARM-JAW-{side}"]["specification"]
        assert "replaces narrow" in parts[f"ARM-JAW-{side}"]["candidate_or_reference"]
    assert bom["arm_candidate_A"].count("XL330-M288-T") == 5
    assert bom["arm_candidate_B"].count("XL330-M077-T") == 2
    with (tmp_path / "BOM.csv").open(encoding="utf-8-sig") as stream:
        exported = list(csv.DictReader(stream))
    assert {row["id"] for row in exported} == set(parts)
    for row in exported:
        assert row["measured_mass_kg"] == ""
        assert int(row["quantity"]) == parts[row["id"]]["quantity"]
