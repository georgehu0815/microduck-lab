# Full candidate parts list / 完整候选物料表

Generated from local engineering BOM and model. NOT a released purchase order. Units and options matter.
源自工程BOM及仿真模型；不是已放行采购单。复用件、可选架构、功能项不可全部相加。

| ID | Qty | Unit | Item | Candidate/reference | Specification | Disposition |
| --- | ---: | --- | --- | --- | --- | --- |
| ARM-M1 | 1 | each | Base-yaw actuator | A/B: XL330-M288-T | J1 output and TTL/VDD/GND; provisional ID 21 | NEW_ARM_PART_CANDIDATE |
| ARM-M2 | 1 | each | Shoulder-pitch actuator | A/B: XL330-M288-T | J2 output and TTL/VDD/GND; provisional ID 22 | NEW_ARM_PART_CANDIDATE |
| ARM-M3 | 1 | each | Elbow-pitch actuator | A/B: XL330-M288-T | J3 output and TTL/VDD/GND; provisional ID 23 | NEW_ARM_PART_CANDIDATE |
| ARM-M4 | 1 | each | Wrist-pitch actuator | A: XL330-M288-T; B: XL330-M077-T | J4 output and TTL/VDD/GND; provisional ID 24 | NEW_ARM_PART_CANDIDATE |
| ARM-M5 | 1 | each | Gripper actuator | A: XL330-M288-T; B: XL330-M077-T | Driven jaw gear and TTL/VDD/GND; provisional ID 25 | NEW_ARM_PART_CANDIDATE |
| ARM-IF-TRUNK | 1 | each | Arm cartridge to trunk load interface | TBD part number | Mount position [15 0 40] mm in inherited model; physical datum and fastener pattern TBD | NEW_ARM_PART_CANDIDATE |
| ARM-BR-J1 | 1 | each | Base-yaw bearing bracket | TBD part number | Supports J1 shaft reaction independently of motor output bearing | NEW_ARM_PART_CANDIDATE |
| ARM-BR-J2 | 1 | each | Shoulder clevis or dual-side bracket | TBD part number | Transfers J2 moment from upper link into J1 carrier | NEW_ARM_PART_CANDIDATE |
| ARM-BR-J3 | 1 | each | Elbow clevis or dual-side bracket | TBD part number | Transfers forearm load into upper link and J3 actuator | NEW_ARM_PART_CANDIDATE |
| ARM-BR-J4 | 1 | each | Wrist carrier | TBD part number | Supports independent wrist shaft and gripper module | NEW_ARM_PART_CANDIDATE |
| ARM-LINK-U | 1 | each | Shoulder-to-elbow structural link | TBD part number | 55 mm nominal joint-axis distance; rendered link feather is decorative, not another load-bearing link | NEW_ARM_PART_CANDIDATE |
| ARM-LINK-F | 1 | each | Elbow-to-wrist structural link | TBD part number | 50 mm nominal joint-axis distance; 55 + 50 + 55 = 160 mm geometric reach in tennis configuration | NEW_ARM_PART_CANDIDATE |
| ARM-WR-CARRIER | 1 | each | Independent wrist transmission carrier | TBD part number | J4 shaft/bearings and M4 horn interface; not the jaw transmission | NEW_ARM_PART_CANDIDATE |
| ARM-HORN-J1 | 1 | each | Servo output horn or hub for J1 | TBD exact manufacturer part | Motor spline to J1 shaft/carrier | NEW_ARM_PART_CANDIDATE |
| ARM-HORN-J2 | 1 | each | Servo output horn or hub for J2 | TBD exact manufacturer part | Motor spline to upper link | NEW_ARM_PART_CANDIDATE |
| ARM-HORN-J3 | 1 | each | Servo output horn or hub for J3 | TBD exact manufacturer part | Motor spline to forearm link | NEW_ARM_PART_CANDIDATE |
| ARM-HORN-J4 | 1 | each | Servo output horn or hub for J4 | TBD exact manufacturer part | Motor spline to wrist shaft | NEW_ARM_PART_CANDIDATE |
| ARM-SHAFT-J1 | 1 | each | Base-yaw structural shaft | TBD part number | BR-J1 bearings; HORN-J1 coupling; axial retention | NEW_ARM_PART_CANDIDATE |
| ARM-SHAFT-J2 | 1 | each | Shoulder structural shaft | TBD part number | BR-J2 bearings; HORN-J2 coupling; axial retention | NEW_ARM_PART_CANDIDATE |
| ARM-SHAFT-J3 | 1 | each | Elbow structural shaft | TBD part number | BR-J3 bearings; HORN-J3 coupling; axial retention | NEW_ARM_PART_CANDIDATE |
| ARM-SHAFT-J4 | 1 | each | Wrist structural shaft | TBD part number | WR-CARRIER bearings; HORN-J4 coupling; axial retention | NEW_ARM_PART_CANDIDATE |
| ARM-BRG-J1 | 2 | each | Base-yaw radial/axial reaction bearings | TBD exact part | SHAFT-J1 and BR-J1 seats | NEW_ARM_PART_CANDIDATE |
| ARM-BRG-J2 | 2 | each | Shoulder radial/axial reaction bearings | TBD exact part | SHAFT-J2 and BR-J2 seats | NEW_ARM_PART_CANDIDATE |
| ARM-BRG-J3 | 2 | each | Elbow radial/axial reaction bearings | TBD exact part | SHAFT-J3 and BR-J3 seats | NEW_ARM_PART_CANDIDATE |
| ARM-BRG-J4 | 2 | each | Wrist radial/axial reaction bearings | TBD exact part | SHAFT-J4 and WR-CARRIER seats | NEW_ARM_PART_CANDIDATE |
| ARM-SPACER-J1 | 1 | set | Base-yaw bearing and horn spacing | TBD part number | J1 axial stack | NEW_ARM_PART_CANDIDATE |
| ARM-SPACER-J2 | 1 | set | Shoulder bearing and horn spacing | TBD part number | J2 axial stack | NEW_ARM_PART_CANDIDATE |
| ARM-SPACER-J3 | 1 | set | Elbow bearing and horn spacing | TBD part number | J3 axial stack | NEW_ARM_PART_CANDIDATE |
| ARM-SPACER-J4 | 1 | set | Wrist bearing and horn spacing | TBD part number | J4 axial stack | NEW_ARM_PART_CANDIDATE |
| ARM-RET-J1-J4 | 1 | set | Shaft nuts/circlips/retainers | TBD part numbers | J1-J4 axial retention | NEW_ARM_PART_CANDIDATE |
| ARM-FAST-STRUCT | 1 | set | Structural screws/nuts/inserts/washers | TBD part numbers | Mount/brackets/links/covers | NEW_ARM_PART_CANDIDATE |
| ARM-GR-GEAR-L | 1 | each | Driven gripper gear | TBD part number | Candidate m=1 mm; z=24; 24 mm equal-gear center distance | NEW_ARM_PART_CANDIDATE |
| ARM-GR-GEAR-R | 1 | each | Passive gripper gear | TBD part number | Candidate m=1 mm; z=24; ratio -1; 24 mm center distance | NEW_ARM_PART_CANDIDATE |
| ARM-GR-SHAFT-L | 1 | each | Driven jaw shaft | TBD part number | GR-GEAR-L; left jaw; bearings; M5 transmission | NEW_ARM_PART_CANDIDATE |
| ARM-GR-SHAFT-R | 1 | each | Passive jaw shaft | TBD part number | GR-GEAR-R; right jaw; bearings | NEW_ARM_PART_CANDIDATE |
| ARM-GR-BRG-L | 2 | each | Driven jaw shaft bearings | TBD exact part | GR-SHAFT-L and gripper housing | NEW_ARM_PART_CANDIDATE |
| ARM-GR-BRG-R | 2 | each | Passive jaw shaft bearings | TBD exact part | GR-SHAFT-R and gripper housing | NEW_ARM_PART_CANDIDATE |
| ARM-GR-TRANS | 1 | each | Independent M5-to-driven-jaw transmission | TBD horn/coupler part numbers | M5 spline to GR-SHAFT-L; separate from wrist shaft | NEW_ARM_PART_CANDIDATE |
| ARM-JAW-L | 1 | each | Wide tennis jaw L: integral bridge/stem and pad carrier | Custom wide_candidate; replaces narrow 30 mm jaw; no extra jaw ordered | 75 mm open pad gap; bridge [0.012, 0.031, 0.003] m; stem [0.033, 0.008, 0.003] m; pad center X=48 mm; TCP=55 mm; existing 24 mm gear axis spacing; dimensions are simulation only | NEW_ARM_PART_CANDIDATE |
| ARM-JAW-R | 1 | each | Wide tennis jaw R: integral bridge/stem and pad carrier | Custom wide_candidate; replaces narrow 30 mm jaw; no extra jaw ordered | 75 mm open pad gap; bridge [0.012, 0.031, 0.003] m; stem [0.033, 0.008, 0.003] m; pad center X=48 mm; TCP=55 mm; existing 24 mm gear axis spacing; dimensions are simulation only | NEW_ARM_PART_CANDIDATE |
| ARM-PAD-L | 1 | each | Left rectangular contact pad | TBD material/part | 24 x 5 x 8 mm modeled envelope on JAW-L | NEW_ARM_PART_CANDIDATE |
| ARM-PAD-R | 1 | each | Right rectangular contact pad | TBD material/part | 24 x 5 x 8 mm modeled envelope on JAW-R | NEW_ARM_PART_CANDIDATE |
| ARM-HARNESS-ARM | 1 | set | Five-actuator VDD/GND/DATA harness | TBD part numbers | TTL multidrop and protected 5 V branches | NEW_ARM_PART_CANDIDATE |
| ARM-CTRL-BENCH | 1 | each | Bench communication controller | U2D2 or equivalent study | Host USB to arm TTL only | BENCH_ONLY |
| ARM-CTRL-MOBILE | 1 | each | Mobile arm controller | TBD | Robot compute to arm TTL and safety chain | NEW_ARM_PART_CANDIDATE |
| ARM-PSU-BENCH | 1 | each | External regulated 5 V bench source | TBD exact model | Default qualification source through protection and disconnect | BENCH_ONLY |
| ARM-REG-SHARED | 1 | each | Shared-main-pack arm regulator | TBD buck or buck-boost | Robot pack to protected 5 V arm domain | POWER_ARCHITECTURE_OPTION_NOT_ADDITIVE |
| ARM-PACK-ARM | 1 | each | Dedicated arm pack | TBD chemistry/cells/capacity | Pack/BMS/charger to protected arm regulator | POWER_ARCHITECTURE_OPTION_NOT_ADDITIVE |
| ARM-F-MAIN | 1 | each | Main arm feeder protection | TBD exact part | Source to arm disconnect/regulator | NEW_ARM_PART_CANDIDATE |
| ARM-F-BRANCH | 5 | each | Per-actuator branch protection | TBD exact parts | Protected 5 V to M1-M5 | PROTECTION_TRADE_STUDY_NOT_SELECTED |
| ARM-SW-DISCONNECT | 1 | each | Manual DC-rated arm disconnect | TBD exact part | Removes arm actuator VDD without commanding motion | NEW_ARM_PART_CANDIDATE |
| ARM-ESTOP | 1 | each | Physical emergency-stop function | TBD architecture/part | Independent motion-authority removal chain | NEW_ARM_PART_CANDIDATE |
| BASE-ankle_left | 1 | each | ankle_left | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-ankle_right | 1 | each | ankle_right | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-banana_pcb_locker | 1 | each | banana_pcb_locker | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-bearing_roll | 2 | each | bearing_roll | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-foot_left | 1 | each | foot_left | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-foot_right | 1 | each | foot_right | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-hip_l | 2 | each | hip_l | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-left_shell | 1 | each | left_shell | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-leg | 2 | each | leg | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-power_support | 1 | each | power_support | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-right_shell | 1 | each | right_shell | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-seeed_bearing__configuration__22x16x4 | 8 | each | seeed_bearing__configuration__22x16x4 | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-seeed_bearing__configuration_default | 2 | each | seeed_bearing__configuration_default | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-sole_left | 1 | each | sole_left | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-sole_right | 1 | each | sole_right | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-trunk_base | 1 | each | trunk_base | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-upper_leg_left | 1 | each | upper_leg_left | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-upper_leg_right | 1 | each | upper_leg_right | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-upper_leg_rigidity_plate | 2 | each | upper_leg_rigidity_plate | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-yaw2roll | 2 | each | yaw2roll | Existing MicroDuck part; matching source mesh (not a machining drawing) | Counted once per visual mesh instance; collision copies excluded | REUSE_EXISTING_ASSEMBLY_SUBPART |
| BASE-LEG-MOTORS | 10 | each | Five leg servos per leg | Installed DYNAMIXEL XL330 family; exact M288/M077 labels to inspect | hip yaw / hip roll / hip pitch / knee / ankle per leg; existing IDs and calibration retained | REUSE |
| BASE-COMPUTE | 1 | each | Onboard compute | Radxa ZERO 3W reference | Retain installed RAM/storage variant; CSI/USB/power budget must be audited | REUSE_OR_ASSEMBLY_INCLUDED |
| BASE-STORAGE | 1 | each | Boot storage | Existing eMMC or microSD | Use installed boot medium; not two new storage devices | REUSE_OR_ASSEMBLY_INCLUDED |
| BASE-CONTROL-BOARD | 1 | each | Motor interface and base power board | Existing MicroDuck board; exact revision TBD | Retain leg bus; no assumed spare arm current capacity or ports | REUSE_OR_ASSEMBLY_INCLUDED |
| BASE-IMU | 1 | function | Body IMU function | Existing board-integrated or discrete sensor; exact SKU TBD | Reuse existing physical sensor; simulation IMU site is not a part number | REUSE_OR_ASSEMBLY_INCLUDED |
| BASE-BATTERY | 1 | each | Main battery pack | Installed pack label required; np_f970 is only a source mesh name | Record chemistry/cell count/voltage/capacity/mass; no guessed battery substitution | REUSE_OR_ASSEMBLY_INCLUDED |
| BASE-CHARGER | 1 | each | Battery-compatible charger | Existing qualified matching charger | Off-robot charging; exact pack compatibility required | REUSE_OR_ASSEMBLY_INCLUDED |
| BASE-PACK-PROTECTION | 1 | function | Pack protection/BMS function | Installed protection; topology TBD | May be integrated in pack; do not buy a duplicate BMS by default | REUSE_OR_ASSEMBLY_INCLUDED |
| BASE-BASE-HARNESS | 1 | set | Leg and onboard wiring set | Existing keyed cables/connectors | Retain known pinout; inspect insulation/flex/strain relief | REUSE_OR_ASSEMBLY_INCLUDED |
| BASE-BASE-FASTENERS | 1 | set | Base screws/spacers/retainers set | Reuse installed fasteners | Exact thread/length/count must be measured; meshes do not define screw procurement | REUSE_OR_ASSEMBLY_INCLUDED |
| SHELL-shoulder_saddle | 1 | each | shoulder_saddle cover envelope | Custom WingPod cream shell; material/process unselected | body=arm_mount; local center=(0, 0, -0.008) m; XYZ envelope=[50.0, 50.0, 8.0] mm; fillet=3 mm | COSMETIC_ENVELOPE_NOT_ADDITIVE_SOLID |
| SHELL-root | 1 | each | root cover envelope | Custom WingPod cream shell; material/process unselected | body=arm_mount; local center=(0, 0, -0.025) m; XYZ envelope=[26.0, 32.0, 40.0] mm; fillet=6 mm | COSMETIC_ENVELOPE_NOT_ADDITIVE_SOLID |
| SHELL-shoulder | 1 | each | shoulder cover envelope | Custom WingPod cream shell; material/process unselected | body=arm_yaw; local center=(0, 0, 0.018) m; XYZ envelope=[26.0, 32.0, 40.0] mm; fillet=6 mm | COSMETIC_ENVELOPE_NOT_ADDITIVE_SOLID |
| SHELL-upper_knuckle | 1 | each | upper_knuckle cover envelope | Custom WingPod cream shell; material/process unselected | body=arm_upper; local center=(0.045, 0, 0.02) m; XYZ envelope=[26.0, 32.0, 40.0] mm; fillet=6 mm | COSMETIC_ENVELOPE_NOT_ADDITIVE_SOLID |
| SHELL-forearm_knuckle | 1 | each | forearm_knuckle cover envelope | Custom WingPod cream shell; material/process unselected | body=arm_forearm; local center=(0.025, 0, 0.02) m; XYZ envelope=[26.0, 32.0, 40.0] mm; fillet=6 mm | COSMETIC_ENVELOPE_NOT_ADDITIVE_SOLID |
| SHELL-palm | 1 | each | palm cover envelope | Custom WingPod cream shell; material/process unselected | body=arm_hand; local center=(0.007, 0, 0.022) m; XYZ envelope=[26.0, 32.0, 40.0] mm; fillet=6 mm | COSMETIC_ENVELOPE_NOT_ADDITIVE_SOLID |
| SHELL-backpack | 1 | each | backpack cover envelope | Custom WingPod cream shell; material/process unselected | body=assumed_dedicated_arm_pack; local center=(0, 0, 0) m; XYZ envelope=[36.0, 30.0, 22.0] mm; fillet=3 mm | COSMETIC_ENVELOPE_NOT_ADDITIVE_SOLID |
| SHELL-electronics | 1 | each | electronics cover envelope | Custom WingPod cream shell; material/process unselected | body=assumed_relocated_electronics; local center=(0, 0, 0) m; XYZ envelope=[42.0, 26.0, 10.0] mm; fillet=2 mm | COSMETIC_ENVELOPE_NOT_ADDITIVE_SOLID |
| SHELL-power | 1 | each | power cover envelope | Custom WingPod cream shell; material/process unselected | body=assumed_power_and_communications; local center=(0, 0, 0) m; XYZ envelope=[30.0, 22.0, 10.0] mm; fillet=2 mm | COSMETIC_ENVELOPE_NOT_ADDITIVE_SOLID |
| SHELL-face | 1 | each | face cover envelope | Custom WingPod cream shell; material/process unselected | body=assumed_sensor_shell; local center=(0, 0, 0) m; XYZ envelope=[38.0, 34.0, 26.0] mm; fillet=6 mm | COSMETIC_ENVELOPE_NOT_ADDITIVE_SOLID |
| CAM-SENSORS | 2 | each | Active left/right RGB camera modules incl. lens | Exact miniature sensor modules UNSELECTED | 13 mm optical center spacing design target; matched optics preferred; voltage/interface not selected | DUAL_EYE_TARGET_BLOCKED |
| CAM-BEZEL | 1 | each | Cream twin-eye housing / peach cheeks / honey beak | Custom camera chest cover; soft appearance revision | 28 mm wide x 17 mm high x 5 mm deep outer envelope; not usable internal volume; colors are render-only | RENDER_ONLY |
| CAM-MOUNT | 1 | each | Camera internal carrier and fixed chest mount | Custom removable nonstructural carrier | Fixed to arm_mount chest; must not load moving gripper or obstruct shoulder | NEW_CANDIDATE |
| CAM-LENS-TRIM | 2 | each | Sage-gray lens trim / light baffle | Custom cosmetic ring; no extra camera lens | 8 mm visual ring outer diameter; avoid vignette/reflections | COSMETIC |
| CAM-LED | 1 | each | Amber status LED | Low-current LED; exact MPN TBD | Displayed amber bar is a render cue; real drive current must be chosen | NEW_CANDIDATE |
| CAM-LED-DRIVER | 1 | each | LED current-limit/driver circuit | Resistor/driver to match chosen logic rail | Do not directly drive unknown load from GPIO; value and topology TBD | NEW_CANDIDATE |
| CAM-LIGHTPIPE | 1 | each | Amber indicator light pipe | Custom diffuser | Matches rendered indicator; thickness and material TBD | NEW_CANDIDATE |
| CAM-DATA-CABLES | 2 | each | Camera data cable assemblies | Matched FFC or USB assemblies; not interchangeable | One per selected module; pitch/pin count/contact side/length TBD | DUAL_EYE_TARGET_BLOCKED |
| CAM-BRIDGE | 1 | function | Dual-camera host interface function | UNSELECTED: synchronized bridge/USB stereo assembly if required | Do not split one CSI input with a passive Y cable; assembly may include this function | ARCHITECTURE_NOT_SELECTED |
| CAM-LOGIC-POWER | 1 | function | Camera protected logic-power branch | Existing logic regulator if qualified; otherwise sized replacement TBD | Camera supply by selected interface; do not connect to motor rail or raw battery | CONDITIONAL_FUNCTION |
| CAM-HARDWARE | 1 | set | Camera screws/inserts/grommets/strain-relief set | Custom matched hardware; size TBD | No drilling or screw lengths released before real-board drawing | NEW_CANDIDATE |
| CAM-BENCH-IMX219 | 1 | each | MicroDuck-compatible single-camera bench reference | Raspberry Pi Camera Module 2 / Sony IMX219 | Reference only; not claimed to fit chest or provide two live eyes | BENCH_ALTERNATIVE_NOT_IN_DUAL_TARGET |
| TEST-TELEMETRY | 1 | set | Voltage/current/temperature instrumentation | Calibrated measurement equipment; onboard sensor selection TBD | Branch current + pack/logic/arm voltage + regulator temperature | BENCH_OR_FIXTURE |
| TEST-POWER-PROTECTION | 1 | set | Transient/reverse-current protection and local decoupling | TVS/blocking/clamp/capacitor topology TBD | Coordinate with measured motor regeneration and converter response; no guessed fuse or capacitor value | ELECTRICAL_FUNCTION_UNSELECTED |
| TEST-FIXTURE | 1 | set | Nonconductive bench fixture and fall restraint | Custom support allowing safe leg/arm tests | Provide catch restraint without mistaking tether-assisted stability for free-base success | BENCH_OR_FIXTURE |
| TEST-CAM-CALIBRATION | 1 | set | Camera calibration target | Printed/measured checkerboard or Charuco target | Intrinsic/extrinsic tests; verified square size; timing and reprojection logs | BENCH_OR_FIXTURE |
| TEST-BALL | 1 | set | Yellow tennis ball | Measured classroom test object | Simulation nominal diameter 67 mm / mass 58 g; this is not a qualified real payload rating | BENCH_OR_FIXTURE |
| TEST-BIN | 1 | set | Low classroom bin | Custom anchored fixture; not a household garbage can | 180 x 180 mm inner plan; 100 mm walls; 4 mm wall/bottom in simulation | BENCH_OR_FIXTURE |
