# MD-Arm-T1 hardware package

Status: bounded design output, not fabrication approval or safety certification.

This directory owns the editable and generated hardware artifacts for one
tabletop MD-Arm-T1 arm. A dual-arm station uses two electrically independent
copies of the 5 V arm branch and two U2D2 adapters.

The earlier 220 mm nominal / 180–240 mm adjustable base-spacing concept is
retained as design history, but it does not cover the current simulation
layouts: handover uses 150 mm and co-carry uses 242 mm between base centers.
A common physical station must therefore adjust over at least 150–242 mm, or
use separate/redesigned platforms followed by updated simulation and physical
validation. These simulation dimensions do not prove actual arm, clamp, tray,
cable, or collision-free workspace fit.

## Source of truth

The supplied `robot_allcollisions_backlash.xml` references 15 instances of one
`xl330.stl`, but only 14 active policy actuators. It does not identify M288 versus
M077. `scripts/shared_components.py` audits that XML, all mesh instances and
the Onshape part provenance. The arm CAD now imports the same original motor
mesh through generated `shared-xl330.scad` instead of drawing a different box.
The source mesh is in meters: uniform scale 1000 converts it to CAD millimeters.
Its measured XYZ bounds are 29 x 20 x 34 mm; the published body envelope is
W20 x H34 x D26 mm. These are distinct representations: do not squeeze the
29 mm reference into 26 mm or infer manufacturing tolerances from the STL.

`generated/shared-xl330-comparison.svg` shows identical front/side/top views
for body and arm. `generated/shared-xl330-preview.xml` loads one mesh for two
visual-only MuJoCo instances; it is not the trained arm model.
`cad/md_arm_t1.scad` supports `render_part="shared_motor"`. Its existing assembly
remains a three-reference kinematic sketch, not a complete six-servo assembly.
The prototype physics and training evidence are unchanged; fitted body motor
parameters and backlash are not automatically valid for the arm.

- `spec/md-arm-t1.json`: manufacturer facts, project design inputs, and blocked
  unknowns. Each value is classified by `fact_class`.
- `bom.json`: machine-readable per-arm BOM and compatibility gates.
- `netlists/arm-ttl-netlist.json`: pin-labeled power and TTL communication
  netlist.
- `cad/md_arm_t1.scad`: editable OpenSCAD concept geometry. It contains no
  servo mounting holes. Its `station_layouts` mode shows the 150 mm and 242 mm
  simulation base-center layouts as conceptual envelopes.
- `scripts/generate.py`: standard-library-only generator for review drawings,
  a wiring drawing, ASCII STL envelope/link blanks, catch tray, and hashes.
- `generated/`: checked-in outputs from the generator.

## Generate and test

```bash
python3 hardware/md-arm-t1/scripts/generate.py
python3 -m unittest discover -s hardware/md-arm-t1/tests -p 'test_*.py'
```

OpenSCAD is optional and was not available when these outputs were generated.
The Python generator intentionally creates only non-precision body envelopes,
link blanks, and a catch tray. It does not create a servo bracket or horn
adapter.

## Fabrication stop

Do not fabricate a servo mating interface from these files. Before a bracket
can be released:

1. Download the official ROBOTIS XL330 PDF/DWG/STEP linked in the spec.
2. Confirm the purchased variant is exactly `XL330-M288-T`.
3. Measure the actual servo, horn, fasteners, cable bend clearance, and mating
   parts.
4. Define process-specific fits and tolerances, then perform a physical fit
   coupon review.
5. Update `interfaces.servo_mount.status` from `blocked` only through a
   separate reviewed hardware revision.

The dual-arm station is also blocked until an adjustment mechanism covering
150–242 mm is designed and measured, or scenario-specific platforms are
designed and the simulation layouts are revalidated. The current drawings are
not evidence that either scenario physically fits.

The proposed 20 g payload remains unverified. The arm must first be tested
unloaded, at reduced reach and speed, above the catch tray.
