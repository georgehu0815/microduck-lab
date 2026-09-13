import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "hardware/md-arm-t1/scripts/shared_components.py"
MODULE_SPEC = importlib.util.spec_from_file_location("shared_components", MODULE_PATH)
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)


class SharedComponentTests(unittest.TestCase):
    def test_xml_inventory_and_variant_boundary(self):
        audit = MODULE.audit_components()
        self.assertEqual(len(audit["motor_instances"]), 15)
        self.assertEqual(audit["actuator_count"], 14)
        self.assertEqual(audit["backlash_joint_count"], 14)
        self.assertIsNone(audit["microduck_exact_variant"])
        self.assertFalse(audit["fabrication_release"])
        self.assertEqual(audit["arm_candidate"], "XL330-M288-T")

    def test_mesh_dimensions_are_not_squeezed_into_body_envelope(self):
        audit = MODULE.audit_components()
        for actual, expected in zip(audit["mesh"]["bounds_size_xyz_mm"], [29, 20, 34]):
            self.assertAlmostEqual(actual, expected, places=4)
        self.assertEqual(audit["manufacturer_body_w_h_d_mm"], [20, 34, 26])
        self.assertEqual(audit["mesh"]["unit_scale_to_mm"], 1000)
        self.assertEqual(audit["mesh"]["triangle_count"], 4126)

    def test_shared_outputs_preserve_source_and_cad_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            MODULE.write_outputs(output)
            audit = json.loads((output / "component-consistency.json").read_text())
            self.assertEqual(
                audit["mesh"]["sha256"], MODULE.audit_components()["mesh"]["sha256"]
            )
            scad = (output / "shared-xl330.scad").read_text()
            self.assertIn("scale([1000, 1000, 1000])", scad)
            self.assertNotIn("26 / 29", scad)
            self.assertIn("module microduck_xl330_reference()", scad)
            svg = ET.parse(output / "shared-xl330-comparison.svg").getroot()
            uses = svg.findall(".//{http://www.w3.org/2000/svg}use")
            self.assertEqual(len(uses), 6)
            self.assertEqual(
                [item.attrib["href"] for item in uses[:3]],
                [item.attrib["href"] for item in uses[3:]],
            )
        cad = (ROOT / "hardware/md-arm-t1/cad/md_arm_t1.scad").read_text()
        self.assertIn("include <../generated/shared-xl330.scad>", cad)
        self.assertIn("microduck_xl330_reference();", cad)
        self.assertNotIn("servo_w_mm = 20;", cad)

    def test_invalid_stl_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.stl"
            path.write_bytes(b"not an STL")
            with self.assertRaises(ValueError):
                MODULE.read_triangles(path)


if __name__ == "__main__":
    unittest.main()
