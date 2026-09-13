---
title: "MicroDuck Arm v1-C Engineering Baseline"
subtitle: "Mechanical, electrical, controls, and hardware review"
date: "2026-09-13"
lang: en
---

# Release statement

> **ENGINEERING CONFIGURATION BASELINE - NOT FABRICATION READY**

This book defines the review baseline for the MicroDuck single-arm v1-C
cartridge. It is not a manufacturing drawing set, wiring release, qualified
power architecture, mobile controller design, safety certification, or record
of completed physical tests.

The baseline fails closed:

- fabrication release: **BLOCKED**;
- powered bench release: **BLOCKED pending staged authorization**;
- robot/mobile release: **BLOCKED**;
- physical evaluations: **NOT RUN**;
- candidate A/B qualification: **NOT RUN**.

Actual outcomes belong in the separate leader-owned `RESULTS.en.md`. A missing
result, blank evidence field, simulation result, estimate, or v1-B result never
becomes a v1-C pass.

# 1. Authority and provenance

The current configuration source is
`hardware/microduck-arm-v1c/spec/design.json`. The configuration validator,
environment, model adapter, and safety-state model are:

- `microduck_arm_v1c/config.py`;
- `microduck_arm_v1c/env.py`;
- `microduck_arm_v1c/model.py`;
- `microduck_arm_v1c/safety.py`.

Vendor-source retrieval and hashes are recorded in
`hardware/microduck-arm-v1c/vendor/SOURCE-MANIFEST.md`. The retained Pololu
D24V60F5/D24V90F5 geometry files are references only; they do not approve a
regulator or its project-specific integration. ROBOTIS drawing links were
located, but their advertised binary drawings were not retained because the
retrieval returned landing-page HTML.

The older `docs/microduck-arm-v1/` package is v1-B reference material. v1-C
inherits nominal kinematic lengths and the legacy simulation geometry where
the current specification says so. It does not inherit v1-B evaluations,
release status, old floor-level task success, or old power decisions.

Official candidate references:

- <https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/>
- <https://emanual.robotis.com/docs/en/dxl/x/xl330-m077/>
- <https://www.pololu.com/product/2866/resources>
- <https://mujoco.readthedocs.io/en/stable/XMLreference.html>

# 2. v1-C configuration

## 2.1 Morphology and control contract

The robot model has ten leg joints plus five arm actuators. The arm has four
pose joints and one gripper actuator:

| Joint | Function | Action index | Provisional bus ID |
| --- | --- | ---:| ---:|
| J1 / M1 | base yaw | 10 | 21 |
| J2 / M2 | shoulder pitch | 11 | 22 |
| J3 / M3 | elbow pitch | 12 | 23 |
| J4 / M4 | wrist pitch | 13 | 24 |
| J5 / M5 | gripper drive | 14 | 25 |

IDs `21-25` are configuration placeholders only. The real IDs, physical pin
mapping, firmware, actuator identity, and calibration are unverified. Real
motor I/O is not implemented or approved by this baseline.

The simulation control period is 20 ms with 2 ms physics steps. The arm slew
screening limit is `0.4 rad/s`; the leg limit is `6 rad/s`. These are
simulation screening values, not qualified hardware speed or acceleration
limits.

## 2.2 Candidate configurations

| Candidate | M1 | M2 | M3 | M4 | M5 | Motor mass |
| --- | --- | --- | --- | --- | --- | ---:|
| A, default pending qualification | M288 | M288 | M288 | M288 | M288 | 5 x 18 g |
| B, research candidate | M288 | M288 | M288 | M077 | M077 | 5 x 18 g |

The two candidate motor types have the same documented 18 g mass:

| Candidate motor | Gear ratio | Stall torque at 5 V | No-load speed at 5 V | Continuous rating |
| --- | ---:| ---:| ---:| --- |
| XL330-M288-T | 288.4 | 0.52 N m | 103 rpm | unknown |
| XL330-M077-T | 77.5 | 0.215 N m | 383 rpm | unknown |

Stall torque and no-load speed are not continuous operating points. Candidate B
is not automatically superior in energy, output inertia, backdrive behavior,
tracking, or disturbance response. Release requires measured output inertia and
backdrive evidence, matched-task energy and thermal tests, a qualified
continuous torque-speed envelope, and A/B tracking and disturbance tests.

# 3. Geometry and changed task scene

![Mechanical dimension overview](../../hardware/microduck-arm-v1c/engineering/mechanical-dimension-overview.svg)

Nominal review geometry:

| Parameter | Baseline value | Release meaning |
| --- | ---:| --- |
| shoulder-to-elbow | 55 mm | kinematic axis distance only |
| elbow-to-wrist | 50 mm | kinematic axis distance only |
| wrist-to-TCP | 30 mm | kinematic distance only |
| summed geometric reach | 135 mm | not a rated load radius or safe workspace |
| shoulder pivot | local `[0, 0, 18]` mm | v1-C change, aligned to shoulder motor envelope center |
| motor clearance envelope | 20 x 26 x 34 mm | not a mounting drawing |
| jaw pivot spacing | 24 mm total | gear-center candidate only |
| jaw length | 30 mm nominal | real section and mounting TBD |
| modeled pad size | 24 x 5 x 8 mm | physical pad properties unmeasured |

The current task scene is `raised_tabletop_v1_c`:

- tabletop top: `z = 145 mm`;
- object half-height: `6 mm`;
- initial object center: `[115, -18, 151] mm`;
- goal center: `[125, 24, 151] mm`;
- transfer waypoint height: `z = 190 mm`.

This is a changed scenario. The earlier 95 mm or floor-level arrangement
created a low target with mount-collision risk. The raised-table scene is still
provisional simulation geometry, not a released physical table, fixture, or
collision-clearance result.

The rectangular pad model uses MuJoCo half-sizes `[12, 2.5, 4] mm` with
simulation contact parameters `solref=".004 1"` and
`solimp=".95 .99 .001"`. Those values do not identify a real elastomer,
durometer, friction coefficient, compression curve, contact stiffness,
adhesive, wear life, or grip rating.

# 4. Mechanical architecture and real load path

![Structural load path](../../hardware/microduck-arm-v1c/engineering/load-path.svg)

The required physical load path is:

**payload -> contact pads -> rigid jaws -> jaw shafts -> jaw bearings ->
gripper housing -> independent wrist shaft and bearings -> forearm -> elbow
shaft and bearings -> upper link -> shoulder shaft and bearings -> base-yaw
shaft and bearings -> trunk interface -> fixture or robot structure.**

Motor output splines and horns transmit actuator torque into this path. They
must not be treated as complete radial, axial, bending, or overturning support
without explicit manufacturer evidence for the measured external loads.

## 4.1 Joint hardware required before release

Every J1-J4 assembly requires a controlled definition for:

- structural shaft material, diameter, shoulders, runout, and fatigue;
- two-plane radial/axial reaction support or a justified equivalent;
- bearing manufacturer part number, static/dynamic ratings, fits,
  preload/endplay, lubrication, and contamination protection;
- exact servo horn/hub part number, spline identity, pilot, bolt circle,
  center screw, allowable torque, and fit;
- spacers or shims controlling axial stack without bearing side-load;
- shaft retainers, nuts, circlips, threads, locking, and pull-off evidence;
- bracket and link material, process, section, tolerances, and proof load;
- structural fastener part numbers, grades, lengths, thread engagement,
  tightening torque, locking, witness marking, access, and reuse rule;
- cable passages, bend radius, strain relief, flex life, and full-motion
  clearance.

All these part numbers remain `TBD`. The motor envelope and inherited visual
mesh must not be used to infer hole positions or mating faces.

## 4.2 Independent wrist and jaw transmission

The wrist is its own structural joint. J4 requires an independent shaft,
bearing pair, carrier, horn interface, and axial retention. The gripper housing
attaches to the wrist output.

The gripper is a separate mechanism. M5 drives one jaw shaft through an
explicit horn/coupler/transmission. The second jaw shaft is passive and driven
by the equal external gear pair. Each jaw shaft requires its own support and
retention. The design must not silently combine the wrist shaft, driven jaw
shaft, and passive jaw shaft.

## 4.3 Gear definition

The current kinematic candidate is:

- module: `1 mm`;
- driven gear: `24 teeth`;
- passive gear: `24 teeth`;
- nominal center distance:
  `m(z1 + z2)/2 = 1 mm(24 + 24)/2 = 24 mm`;
- ratio: `-1`;
- left driven jaw range: `[-0.32, 0] rad`;
- passive jaw relationship: equal magnitude, opposite rotation.

This is not a gear release. Material, pressure angle, face width, profile
shift, root fillet, quality grade, backlash, center-distance tolerance,
lubrication, wear, allowable tooth load, hub geometry, shaft fit, retention,
guarding, and debris behavior are unknown. The mechanism remains blocked until
those items and physical cycle/load evidence are complete.

# 5. Engineering BOM and mass baseline

The controlled row-level BOM is `ENGINEERING-BOM.csv`. It includes ID,
quantity, function, candidate, revision, mass, CoM, interface, status, required
evidence, owner, and gate for motors, links, brackets, bearings, shafts, horns,
spacers, retainers, fasteners, independent wrist parts, jaw transmission,
gears, jaws, pads, harness, controllers, and power/safety hardware.

## 5.1 Corrected mass target

| Category | Target or estimate |
| --- | ---:|
| five motors | 90 g |
| links | 20 g |
| brackets | 10 g |
| gripper | 10 g |
| trunk interface | 10 g |
| wiring and fasteners | 10 g |
| reserve | 10 g |
| **controlled total** | **160 g** |

The user-provided category budget summed to 170 g. The controlled v1-C target
is corrected to 160 g. Release requires a closed as-built mass ledger at or
below 160 g; the reserve cannot become negative or conceal missing items.

The 70 g relocated-electronics, 80 g dedicated-pack, 25 g power/communication,
and 8 g sensor-shell values inherited from v1-B are simulation placeholders
outside the v1-C cartridge BOM. They are not purchased parts or measured
packaging.

## 5.2 Inertia estimates

The following diagonal body-frame tensors are box estimates from the current
specification. They are neither measured principal inertias nor a released
assembled inertia model.

| Component | Mass | Box size (m) | Estimated diagonal inertia `(Ixx, Iyy, Izz)` kg m2 |
| --- | ---:| --- | --- |
| M1 | 0.018 kg | 0.020 x 0.026 x 0.034 | `(2.748e-6, 2.334e-6, 1.614e-6)` |
| M2 | 0.018 kg | 0.020 x 0.026 x 0.034 | `(2.748e-6, 2.334e-6, 1.614e-6)` |
| M3 | 0.018 kg | 0.020 x 0.026 x 0.034 | `(2.748e-6, 2.334e-6, 1.614e-6)` |
| M4 | 0.018 kg | 0.020 x 0.026 x 0.034 | `(2.748e-6, 2.334e-6, 1.614e-6)` |
| M5 | 0.018 kg | 0.020 x 0.026 x 0.034 | `(2.748e-6, 2.334e-6, 1.614e-6)` |
| links | 0.020 kg | 0.090 x 0.012 x 0.008 | `(3.46667e-7, 1.3606667e-5, 1.374e-5)` |
| brackets | 0.010 kg | 0.050 x 0.040 x 0.012 | `(1.453333e-6, 2.203333e-6, 3.416667e-6)` |
| gripper | 0.010 kg | 0.040 x 0.030 x 0.020 | `(1.083333e-6, 1.666667e-6, 2.083333e-6)` |
| interface | 0.010 kg | 0.040 x 0.030 x 0.008 | `(8.03333e-7, 1.386667e-6, 2.083333e-6)` |
| wiring/fasteners | 0.010 kg | 0.080 x 0.020 x 0.010 | `(4.16667e-7, 5.416667e-6, 5.666667e-6)` |
| reserve | 0.010 kg | none | no inertia estimate |

The principal-inertia release gate is explicitly failed. Before S6, weigh the
complete assembly, measure CoM in at least three orientations, identify each
component pose, and derive or measure the assembled inertia with uncertainty.

# 6. Load, torque, and structural margin

For each joint:

```text
tau_g,j = sum_i(m_i * g * horizontal_lever_arm_i_about_joint_j)
tau_required_continuous,j >= 3 * max_released_pose(tau_g,j)
```

The structural and actuator review must then include acceleration, reflected
inertia, contact loads, cable forces, friction, backlash impact, disturbance,
and thermal duty cycle. The factor of three is a minimum continuous-torque
gate, not permission to divide stall torque by three.

A deliberately conservative bound places the full 160 g cartridge and 20 g
payload at the full 135 mm geometric reach:

```text
3 * 0.18 kg * 9.80665 m/s2 * 0.135 m = 0.715 N m
```

That bound is not the actual shoulder requirement because real mass is
distributed. It does show that neither candidate has release evidence: M288
lists 0.52 N m stall torque, M077 lists 0.215 N m stall torque, and qualified
continuous ratings are unknown. The simulation force limit of `0.1 N m` is
also only a screening assumption.

# 7. Power, wiring, and controller architecture

![Power and data architecture](../../hardware/microduck-arm-v1c/engineering/wiring-architecture.svg)

## 7.1 Bench default

The default qualification source is an external regulated 5 V bench supply
with current limiting, enclosure, calibrated measurement, main protection,
manual DC-rated disconnect, and individually reviewed actuator branches. No
exact source, fuse, connector, wire, switch, or branch-protection part is
selected.

## 7.2 Shared-pack study

A shared robot-pack topology may be studied only through an independently
protected regulator. The robot pack's minimum, maximum, transient, load-step,
and regenerative behavior are unknown. Therefore a buck-boost converter may be
required; a buck-only converter must not be assumed from nominal voltage.

The retained Pololu D24V60F5/D24V90F5 mechanical files are vendor references,
not qualification of voltage range, current, thermal performance, dynamics,
mounting, or reverse-current behavior for this robot.

## 7.3 Separate-pack study

A separate arm pack remains conditional on exact chemistry, cell count,
capacity, fault current, BMS, charger, enclosure, connector, transport
controls, runtime, thermal behavior, regulator, and regenerative protection.
No pack is qualified.

## 7.4 Prohibited connections and actions

- Never intentionally short the source or perform short-circuit tests on the
  robot.
- Never reverse polarity.
- Never connect arm VDD to robot battery positive, body VDD, or USB VBUS.
- Never permit the arm source or USB interface to reverse-power the robot.
- Never connect arm DATA to the body actuator bus without an approved
  architecture.
- Never infer power delivery from U2D2 enumeration.

There is no approved mobile controller. U2D2 or equivalent is only a bench
communication study.

# 8. Safety and control gates

The current safety code is a deterministic simulation/HIL state model with no
device I/O. It models:

- a 100 ms command deadline;
- a separate 100 ms software-heartbeat deadline;
- monotonic sequence and timestamp checks;
- replay detection;
- latched E-stop, overcurrent, and undervoltage faults;
- fail-closed motion authority.

Repeated read or heartbeat traffic cannot refresh stale command age. A bus
watchdog requests a stop but does not guarantee torque-off. Physical E-stop,
power isolation, contactor/disconnect behavior, fault detection, and real
actuator stop time are unqualified.

The arm model uses a 15-action, 64-observation v1-C contract. No approved
mobile controller implements the physical mapping. Hardware software must
preserve deterministic J1-M1 through J5-M5 mapping, actual unique IDs,
joint-sign and zero calibration, slew/current/temperature limits, watchdog,
heartbeat, and explicit arm/disarm/reset state.

# 9. Staged hardware acceptance

The complete bilingual command procedure is `HARDWARE-SOP.md`; the planned
cell-level matrix is `TEST-MATRIX.csv`.

| Stage | Scope | Current state |
| --- | --- | --- |
| S0 | configuration and evidence audit | NOT RUN |
| S1 | unpowered mechanical/load-path inspection | NOT RUN |
| S2 | isolated power-domain qualification | NOT RUN |
| S3 | torque-disabled communications and safety logic | NOT RUN |
| S4 | fixed-root single-axis dynamic motion | BLOCKED |
| S5 | fixed-root coordinated motion and payload ladder | BLOCKED |
| S6 | restrained-system disturbance/fault test and mobile decision | BLOCKED |

Payload levels are P0, P5, P10, P20, P30, P40, and P50 in grams. P0-P10 are
commissioning points. P20 is the stationary target. P30-P50 are optional
engineering overload points, not ratings; they require separate authorization
and may be omitted when unsafe.

Advancement requires traceable measurements and a reviewed pass at the prior
stage and payload. No matrix cell has passed. v1-B outcomes and current
simulation runs cannot fill physical evidence cells.

# 10. Release gates

Fabrication, powered bench motion, or mobile integration remains blocked until
the applicable owner closes all of the following:

1. Physical robot inventory and exact actuator SKUs.
2. Official mating geometry, actual measurements, fit coupons, and controlled
   drawings.
3. Complete shaft, bearing, horn, spacer, retainer, fastener, bracket, and link
   definitions with structural calculations and proof loads.
4. Independent wrist and jaw transmission release.
5. Gear material, pressure angle, face width, backlash, strength, wear,
   retention, and guarding.
6. Measured pad dimensions, material, friction, compliance/contact stiffness,
   retention, and wear.
7. Closed mass ledger at or below 160 g, measured CoM, and measured or
   validated full inertia.
8. Qualified continuous torque-speed-thermal envelopes meeting the 3x gate.
9. Candidate A/B tracking, energy, thermal, backdrive, and disturbance
   comparison.
10. Qualified source/pack, BMS/charger where applicable, converter,
    reverse-current behavior, fuses, disconnect, connectors, wire, grounding,
    and transient/thermal evidence.
11. Physical mapping and calibration of unique bus IDs; provisional IDs do not
    count.
12. Independent watchdog, software heartbeat, physical E-stop, reset, and
    real stop-response validation.
13. Selected and reviewed mobile controller.
14. Sensor/estimator latency, attachment compliance, dynamic balance,
    collision, dropped-load, tip-over, and human-access validation.

# 11. Stop condition

This baseline is complete when it enables review without implying fabrication
or test success. It intentionally stops before part selection, machining,
wiring, powered motion, or mobile release. The next authoritative records are
controlled drawings, selected-part evidence, completed S/P test records, and
the leader-owned results files.
