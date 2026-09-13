---
title: WingPod Camera v2 Kickstarter Campaign
description: Campaign-ready copy for introducing the open, characterful WingPod Camera v2 robotics concept to early backers
author: Microduck Lab community
ms.date: 2026-09-13
ms.topic: overview
keywords:
  - WingPod Camera v2
  - Kickstarter
  - robotics
  - open source hardware
  - robot vision
estimated_reading_time: 10
---

## Give your robot eyes with character

**WingPod Camera v2 is a compact twin-eye perception concept for makers who
want robots to feel expressive, understandable, and ready to explore.**

Most robot cameras look like an afterthought: a bare board, a loose cable, or a
black box bolted onto an otherwise thoughtful design. WingPod starts from a
different idea. A perception module can communicate where a robot is looking,
fit its personality, and remain open enough for its community to study,
modify, and improve.

We have already built the digital design, integrated it with the Microduck arm
concept, and recorded reproducible simulation evidence. This campaign would
fund the difficult next step: turning a promising 28 x 17 x 5 mm appearance
envelope into a tested physical prototype and a documented maker platform.

> [!IMPORTANT]
> WingPod Camera v2 is currently a simulation-validated engineering concept,
> not finished camera hardware. The sensor, electronics, enclosure, power,
> calibration, certifications, final price, and delivery schedule require
> development and validation funded by the campaign.

![WingPod Camera v2 integrated with the Microduck arm concept](../../artifacts/microduck-arm-v1c/appearance/wingpod-camera-v2/hero.png)

## A small face for a bigger idea

WingPod places two visible optical apertures and an amber status indicator on
the robot's fixed chest. The paired-eye layout gives observers an immediate
visual cue: this is where the robot perceives its environment.

The design combines:

* A proposed 28 x 17 x 5 mm camera-housing envelope
* Two visible optical channels with 13 mm proposed center spacing
* A 70-degree synthetic vertical field of view used in simulation
* Fixed-chest placement that does not move with the arm or gripper
* A graphite and teal face designed to complement Microduck's cream and honey
  body language
* An amber indicator concept for visible device status
* Open engineering artifacts, including geometry, BOM candidates, videos, and
  verification records

The result is more than a sensor mount. It is a bridge between robot perception
and human understanding.

![Close view of the proposed twin-eye housing](../../artifacts/microduck-arm-v1c/appearance/wingpod-camera-v2/camera-eyes-detail.png)

## Why back WingPod

### Make perception understandable

When cameras are visible and directional, students, collaborators, and
spectators can better understand the robot's point of view. WingPod turns an
abstract perception pipeline into something people can see and discuss.

### Build on inspectable engineering

The project does not begin with a glossy render and a promise. The repository
already contains machine-readable geometry, proposed dimensions, source-bound
videos, candidate BOM exports, optical-view simulations, and explicit
verification limits. Backers can inspect what exists and see what remains.

### Learn through a complete robotics stack

WingPod sits inside the Microduck Lab ecosystem, where makers can explore
MuJoCo simulation, robot-arm tasks, RLX training workflows, recorded evidence,
and browser-based experiment review. The goal is a platform that makes sensing,
mechanics, control, and learning easier to investigate together.

### Help shape an open maker platform

Backers should not have to wait until delivery to participate. Development
updates can publish enclosure iterations, camera tradeoffs, field-of-view
tests, thermal results, calibration work, and failures worth learning from.
Community feedback can influence the final balance among compact size,
serviceability, image quality, cost, and compatibility.

## What already works

The current project includes two reviewed videos:

1. A 20-second, 960 x 720 appearance film showing the robot, orbit view, and
   twin-eye detail
2. A 50.84-second, 640 x 480 source-bound simulation replay showing a complete
   ground-ball pickup, carry, supported bin placement, release, and retreat

[Watch the 20-second appearance film](../../artifacts/microduck-arm-v1c/appearance/wingpod-camera-v2/wingpod-camera-eyes.mp4)

[Watch the complete tennis-task replay](../../artifacts/microduck-arm-v1c/appearance/wingpod-camera-v2/tennis-return-nominal-0/rollout.mp4)

![Verified phases from the tennis-task simulation replay](../../artifacts/microduck-arm-v1c/appearance/wingpod-camera-v2/tennis-return-nominal-0/sequence-contact-sheet.jpg)

The task replay reproduces 2,541 recorded control steps and 1,271 decoded video
frames. Its verification receipt reports zero error in replayed simulation time,
ball position, recorded joint position, and action round-trip values at the
declared tolerances.

These results establish reproducibility for one simulated action replay. They
do not establish physical fit, autonomous visual control, stereo depth,
all-condition task performance, or production readiness.

## What this campaign will build

Campaign funding would move WingPod through four evidence gates.

### Gate 1: Select the optical architecture

We will compare real miniature camera options against the proposed enclosure,
interface, bandwidth, synchronization, image quality, availability, and cost.
If no credible two-channel solution fits, we will revise the enclosure rather
than hide the mismatch.

Deliverables:

* Selected sensor and interface architecture
* Measured optical and electronics envelopes
* Bench capture from one channel, followed by both intended channels
* Updated BOM with traceable supplier parts
* Published tradeoff report

### Gate 2: Engineer the physical enclosure

We will redesign the appearance envelope around measured hardware, cable bend
radii, fasteners, wall thickness, assembly access, ventilation, and robot
clearance.

Deliverables:

* Printable prototype enclosure files
* Assembly and service instructions
* Cable-routing design
* Fit and full-motion clearance evidence
* Measured mass and center-of-mass update

### Gate 3: Validate power, thermal behavior, and imaging

Camera hardware must work for sustained sessions, not only a launch-day photo.
We will test power integrity, temperature, stream stability, timestamps, field
of view, arm occlusion, and calibration if the selected architecture supports
stereo use.

Deliverables:

* Sustained capture and thermal test report
* Measured power data
* Example images and video from physical hardware
* Visibility review across representative robot poses
* Calibration workflow where applicable

### Gate 4: Prepare the maker release

After the design passes the earlier gates, we will package the files and
instructions needed for backers to assemble, operate, and extend WingPod.

Deliverables:

* Versioned hardware design package
* Firmware and host-side examples within the selected architecture
* Assembly, bring-up, and troubleshooting guides
* Safety and operating limits
* Final BOM and sourcing notes

## Designed for curious builders

WingPod is intended for:

* Robotics students who want to connect perception theory to visible hardware
* Makers building expressive robots and interactive installations
* Educators teaching camera geometry, calibration, control, and robot learning
* Researchers who value reproducible artifacts and honest evidence boundaries
* Microduck enthusiasts who want to experiment with a characterful arm and
  perception concept

Compatibility beyond the documented Microduck concept must be validated.
Universal mounting should be treated as a future design goal, not a current
promise.

## Proposed rewards

Final prices, quantities, shipping regions, taxes, and included components must
be confirmed before publication.

| Level | Suggested reward | Campaign price |
| --- | --- | ---: |
| Community supporter | Backer updates, name on the supporter page, and digital wallpapers | `[PRICE]` |
| Digital builder | Design files, BOM, assembly guide, and development archive after release | `[PRICE]` |
| Beta maker | Early prototype enclosure kit with direct feedback access; electronics inclusion must be specified | `[PRICE]` |
| Complete developer kit | Validated WingPod hardware package, cables, enclosure, and software access; exact contents pending architecture selection | `[PRICE]` |
| Education lab pack | Multiple kits, curriculum materials, and a remote workshop for one class or lab | `[PRICE]` |
| Founding collaborator | Limited design review session, recognition, and early integration support within a written scope | `[PRICE]` |

> [!NOTE]
> Do not offer a complete developer kit until the campaign team has a supplier
> quotation, build yield estimate, packaging plan, shipping model, and tested
> prototype matching the reward description.

## Responsible stretch goals

Stretch goals should expand documentation and compatibility only after the
base hardware is funded and its schedule has adequate margin.

* Additional enclosure colors and printable faceplate options
* A documented general-purpose mounting adapter
* Classroom lessons for camera geometry and robot perception
* Additional host-platform examples
* A public dataset captured from the selected hardware

Autonomous vision, depth accuracy, object tracking, weather resistance, and
regulatory certifications should not be offered as stretch goals without a
validated technical plan and budget.

## Development timeline

Replace the relative schedule with calendar dates after funding, supplier lead
times, prototype capacity, and certification requirements are confirmed.

| Phase | Target window after funds clear | Exit evidence |
| --- | --- | --- |
| Architecture and sourcing | Months 1-2 | Sensor decision, supplier quotes, measured stack |
| Electrical and mechanical prototypes | Months 3-4 | Working capture hardware and enclosure revisions |
| Integration and validation | Months 5-6 | Motion, thermal, power, imaging, and assembly reports |
| Pilot build | Months 7-8 | Documented pilot yield and corrected production files |
| Backer production | Months 9-10 | Inspected units and completed software package |
| Fulfillment | Month 11 onward | Packed rewards and region-specific shipment tracking |

This schedule is a planning template, not a delivery commitment. The final
campaign must include supplier lead-time margin and disclose whether compliance
testing is required for the offered configuration and shipping regions.

## How funds will be used

Publish percentages only after the final reward mix and quotations are known.
The budget should account for:

* Camera, interface, cable, and electronics engineering
* Mechanical design and prototype iterations
* Test fixtures, calibration targets, and validation equipment
* Pilot manufacturing and yield loss
* Documentation, software, and educational materials
* Packaging, freight, fulfillment, duties, and replacement inventory
* Kickstarter and payment-processing fees
* Taxes, compliance work, and contingency

Backers deserve to know which costs scale per unit and which fund one-time
engineering. We will publish that distinction before launch.

## Risks and challenges

### Fitting two real cameras

The proposed face is smaller than two complete Raspberry Pi Camera Module 2
boards and the reference compute platform exposes one direct CSI input. The
physical prototype may require smaller remote-head modules, an interface bridge,
or a larger enclosure. A passive splitter is not a credible dual-camera plan.

### Field of view and arm occlusion

Simulation already shows that the arm can obscure part of the view in some
poses. Sensor angle and placement may change after full-motion visibility tests.

### Power and heat

Two sustained image streams can increase compute, bandwidth, power, and thermal
loads. Final capture modes will be based on measured stability rather than a
sensor's maximum advertised resolution.

### Robot balance and durability

The current camera geometry is render-only and has no assigned physical mass.
The complete prototype must be weighed and modeled before robot-motion claims
can be made. Enclosure retention, impacts, cable flex, and service access also
require physical tests.

### Supply and fulfillment

Camera modules and interface components can change revision or availability.
The final campaign should name qualified substitutions, maintain schedule
margin, and avoid ordering production quantities before the pilot build passes.

We will report setbacks with the same evidence discipline used for successes.
That transparency is part of the product.

## Frequently asked questions

### Is WingPod Camera v2 finished hardware

No. It is a detailed appearance and integration concept with simulation
evidence. Kickstarter funding would support sensor selection, physical
prototypes, validation, documentation, and a maker release.

### Does it provide stereo depth

Not yet. The current design uses two simulated RGB viewpoints. A real sensor
stack, synchronization method, calibration, and depth performance have not been
selected or validated.

### Does the robot use these images to control the arm

Not in the current replay. The verified tennis video is an action replay, and
the camera styling does not participate in control. Visual autonomy would be a
separate development effort.

### Is a Microduck included

Only if a final reward explicitly says so. The current campaign concept is for
WingPod development and related maker materials. Reward contents must list
every included component before launch.

### Is this an official Pollen Robotics product

No. This is a community project and is not affiliated with or endorsed by
Pollen Robotics. Microduck names and references identify the compatibility and
research context.

### Will the files be open source

The project is being developed in an open repository. The campaign team must
state the exact licenses and release timing for hardware, firmware, software,
media, and documentation before launch.

## Join the build

WingPod Camera v2 asks a practical question with a playful face: what if robot
perception hardware were designed to be understood, inspected, and improved by
the people around it?

Back WingPod to help move the project from verified pixels to tested physical
hardware. Follow the engineering. Challenge the assumptions. Help build a pair
of robot eyes that earns its place on the robot.

* [Explore the Microduck Lab repository](https://github.com/georgehu0815/microduck-lab)
* [Open Microduck Training Studio](https://georgehu0815.github.io/microduck-lab/)
* [Review the WingPod Camera v2 page](https://georgehu0815.github.io/microduck-lab/wingpod/)
* Contact: [bochuxt7@gmail.com](mailto:bochuxt7@gmail.com)

## Founder launch checklist

Remove this section before pasting the campaign into Kickstarter.

* [ ] Confirm the campaign entity, team biographies, and relevant experience
* [ ] Set the funding goal from quotations and a unit-economics model
* [ ] Finalize reward prices, quantities, contents, and early-bird limits
* [ ] Define shipping regions, taxes, duties, return handling, and warranty terms
* [ ] Replace the relative timeline with dated commitments and schedule margin
* [ ] State the exact open source licenses and release timing
* [ ] Add physical prototype photos before offering physical rewards
* [ ] Obtain supplier quotations and document component substitutions
* [ ] Confirm applicable radio, electrical, battery, product, and import rules
* [ ] Review every image and claim for third-party rights and trademark context
* [ ] Add a campaign video with captions and clear simulation labels
* [ ] Proofread the final Kickstarter preview on desktop and mobile