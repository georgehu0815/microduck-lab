import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "spec" / "md-arm-t1.json"
BOM_PATH = ROOT / "bom.json"
NETLIST_PATH = ROOT / "netlists" / "arm-ttl-netlist.json"
GENERATOR = ROOT / "scripts" / "generate.py"


class HardwarePackageTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        self.bom = json.loads(BOM_PATH.read_text(encoding="utf-8"))
        self.netlist = json.loads(NETLIST_PATH.read_text(encoding="utf-8"))

    def test_authoritative_candidate_and_design_boundaries(self):
        candidate = self.spec["manufacturer_candidate"]
        self.assertEqual(candidate["manufacturer"], "ROBOTIS")
        self.assertEqual(candidate["model"], "XL330-M288-T")
        self.assertEqual(candidate["model_number"]["value"], 1200)
        self.assertEqual(
            candidate["body_envelope_w_h_d_mm"]["value"],
            [20.0, 34.0, 26.0],
        )
        self.assertEqual(candidate["input_voltage_v"]["recommended"], 5.0)
        self.assertEqual(candidate["stall_at_5v"]["current_a"], 1.47)
        self.assertEqual(candidate["connector"]["pins"], [
            {"pin": 1, "label": "GND"},
            {"pin": 2, "label": "VDD"},
            {"pin": 3, "label": "DATA"},
        ])
        self.assertTrue(all(
            source["url"].startswith("https://")
            and source["accessed"] == "2026-09-12"
            for source in candidate["sources"]
        ))

        geom = self.spec["design_geometry"]
        self.assertEqual(
            [
                geom["upper_arm_axis_distance_mm"],
                geom["forearm_axis_distance_mm"],
                geom["wrist_to_tcp_mm"],
            ],
            [65.0, 55.0, 30.0],
        )
        self.assertEqual(geom["maximum_geometric_reach_mm"], 150.0)
        self.assertEqual(geom["gripper_opening_mm"]["maximum"], 30.0)
        self.assertEqual(self.spec["payload"]["target_g"], 20.0)
        self.assertEqual(self.spec["payload"]["verification"], "unverified")
        self.assertEqual(
            self.spec["interfaces"]["servo_mount"]["status"],
            "blocked",
        )

    def test_station_layout_distinguishes_history_simulation_and_fit(self):
        spacing = self.spec["design_geometry"]["dual_arm_base_spacing_mm"]
        self.assertEqual(
            {
                "nominal": spacing["nominal"],
                "adjustable_minimum": spacing["adjustable_minimum"],
                "adjustable_maximum": spacing["adjustable_maximum"],
            },
            {
                "nominal": 220.0,
                "adjustable_minimum": 180.0,
                "adjustable_maximum": 240.0,
            },
        )
        self.assertEqual(
            spacing["fact_class"],
            "historical_project_design_candidate",
        )
        self.assertEqual(
            spacing["simulation_layouts"]["arms-handover-v1"][
                "base_center_spacing"
            ],
            150.0,
        )
        self.assertEqual(
            spacing["simulation_layouts"]["arms-co-carry-v1"][
                "base_center_spacing"
            ],
            242.0,
        )
        requirement = spacing["physical_station_requirement"]
        self.assertEqual(
            [
                requirement["minimum_adjustable_center_spacing"],
                requirement["maximum_adjustable_center_spacing"],
            ],
            [150.0, 242.0],
        )
        self.assertEqual(requirement["verification"], "unverified")
        self.assertEqual(requirement["fit_claim"], "none")

    def test_bom_forbids_original_power_and_checks_variant(self):
        servo = next(
            item for item in self.bom["items"]
            if item["id"] == "ACT-XL330-M288-T"
        )
        self.assertEqual(servo["manufacturer_part_number"], "XL330-M288-T")
        self.assertEqual(servo["quantity_per_arm"], 6)
        self.assertTrue(any(
            "model number 1200" in check
            for check in servo["compatibility_checks"]
        ))
        forbidden = " ".join(self.bom["forbidden_connections"])
        self.assertIn("original Microduck battery", forbidden)
        self.assertIn("original Microduck actuator bus", forbidden)
        self.assertIn("U2D2 TTL port used as a servo power source", forbidden)

    def test_pin_labeled_netlist_and_protected_branches(self):
        self.assertEqual(
            self.netlist["ports"]["U2D2_TTL"],
            {
                "pin_1": "GND",
                "pin_2": "N/C",
                "pin_3": "DATA",
                "note": (
                    "The U2D2-specific official pinout labels TTL pin 2 N/C. "
                    "ROBOTIS also states U2D2 does not supply DYNAMIXEL power. "
                    "External VDD is injected only by protected actuator branches."
                ),
            },
        )
        names = {net["name"] for net in self.netlist["nets"]}
        self.assertTrue({
            "ARM_5V_RAW",
            "ARM_5V_FUSED",
            "ARM_5V_ENABLED",
            "ARM_GND",
            "TTL_DATA",
            "J1_VDD",
            "J2_VDD",
            "J3_VDD",
            "J4_VDD",
            "J5_VDD",
            "G_VDD",
        }.issubset(names))
        self.assertIn(
            "U2D2_TTL:2_N/C",
            self.netlist["explicit_no_connects"],
        )

    def test_stall_sum_is_capacity_basis_not_payload_claim(self):
        stall_sum = (
            self.spec["design_geometry"]["active_actuator_count_per_arm"]
            * self.spec["manufacturer_candidate"]["stall_at_5v"]["current_a"]
        )
        self.assertTrue(math.isclose(stall_sum, 8.82))
        architecture = self.spec["power_architecture"]
        self.assertEqual(architecture["candidate_capacity_a"], 10.0)
        self.assertIn("stall", architecture["capacity_basis"].lower())

    def test_generator_creates_parseable_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(GENERATOR),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            receipt = json.loads(completed.stdout)
            self.assertEqual(receipt["status"], "generated")
            expected = {
                "md-arm-t1-mechanical.svg",
                "md-arm-t1-wiring.svg",
                "md-arm-t1-link-blanks.stl",
                "xl330-body-envelope.stl",
                "md-arm-t1-catch-tray.stl",
                "manifest.json",
                "component-consistency.json",
                "shared-xl330.scad",
                "shared-xl330-comparison.svg",
                "shared-xl330-preview.xml",
            }
            self.assertEqual(set(receipt["files"]), expected)

            for svg_name in (
                "md-arm-t1-mechanical.svg",
                "md-arm-t1-wiring.svg",
            ):
                tree = ET.parse(output / svg_name)
                self.assertTrue(tree.getroot().tag.endswith("svg"))

            mechanical = (output / "md-arm-t1-mechanical.svg").read_text()
            self.assertIn("65 mm axis distance", mechanical)
            self.assertIn("55 mm axis distance", mechanical)
            self.assertIn("30 mm to TCP", mechanical)
            self.assertIn("mounting interface BLOCKED", mechanical)
            self.assertIn("Handover simulation", mechanical)
            self.assertIn("150 mm base centers", mechanical)
            self.assertIn("Historical candidate", mechanical)
            self.assertIn("220 mm base centers", mechanical)
            self.assertIn("adjustment candidate 180-240 mm", mechanical)
            self.assertIn("Co-carry simulation", mechanical)
            self.assertIn("242 mm base centers", mechanical)
            self.assertIn("adjust 150-242 mm, or redesign and revalidate", mechanical)
            self.assertIn(
                "SIMULATION LAYOUT ONLY - PHYSICAL FIT UNVERIFIED",
                mechanical,
            )
            self.assertNotIn("mounting hole diameter", mechanical)

            wiring = (output / "md-arm-t1-wiring.svg").read_text()
            self.assertIn("pin 1 GND", wiring)
            self.assertIn("pin 2 N/C", wiring)
            self.assertIn("pin 3 DATA", wiring)
            self.assertIn("original Microduck battery positive", wiring)

            for stl_name in (
                "md-arm-t1-link-blanks.stl",
                "xl330-body-envelope.stl",
                "md-arm-t1-catch-tray.stl",
            ):
                stl = (output / stl_name).read_text()
                self.assertTrue(stl.startswith("solid "))
                self.assertIn("facet normal", stl)
                self.assertTrue(stl.rstrip().endswith(stl.splitlines()[0].replace(
                    "solid ", "endsolid ", 1
                )))

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertFalse(manifest["fabrication_release"])
            self.assertEqual(manifest["servo_mount_status"], "blocked")
            self.assertEqual(manifest["payload_verification"], "unverified")
            self.assertEqual(
                manifest["station_layout_contract"],
                {
                    "historical_candidate_mm": {
                        "nominal": 220.0,
                        "adjustable_minimum": 180.0,
                        "adjustable_maximum": 240.0,
                    },
                    "simulation_base_center_spacing_mm": {
                        "arms-handover-v1": 150.0,
                        "arms-co-carry-v1": 242.0,
                    },
                    "required_physical_adjustment_mm": {
                        "minimum": 150.0,
                        "maximum": 242.0,
                    },
                    "fit_claim": "none",
                    "verification": "unverified",
                },
            )
            self.assertEqual(len(manifest["outputs"]), 9)
            self.assertTrue(all(
                len(item["sha256"]) == 64 and item["bytes"] > 100
                for item in manifest["outputs"]
            ))


if __name__ == "__main__":
    unittest.main()
