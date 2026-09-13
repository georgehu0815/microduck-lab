/*
MicroDuck single arm v1-B editable parametric engineering concept.

ENGINEERING PROTOTYPE NOT FABRICATION RELEASE

This model contains:
- five motor clearance envelopes, not mounting interfaces;
- four pose axes: base yaw, shoulder, elbow, wrist;
- one rotary gripper actuator;
- equal counter-rotating jaw gears with ratio -1;
- solid support and link blanks labelled for later mating machining.

This model intentionally contains no servo holes, horn bolt circle, pilot,
bearing seat, fastener depth, cable gland, tolerance, or released gear tooth
geometry. The source mesh is about 29 mm deep while the manufacturer body
envelope is 26 mm deep; neither is authority for a mating interface.
Canonical servo centers are visual placement inputs only. Overall packaging,
self-interference, cable routing, and table clearance remain unverified.
*/

$fn = 64;

include <design_parameters.scad>

render_part = "assembly"; // assembly, gripper, support_blanks, motor_envelope
pose_deg = [0, 18, -32, 14]; // base, shoulder, elbow, wrist review pose
gripper_left_q_rad = -0.16;

link_width_mm = 14;
link_thickness_mm = 5;
support_wall_mm = 3;
support_clearance_mm = 1.5; // visualization only; not a manufacturing fit
gear_pitch_radius_mm = jaw_center_offset_mm;
gear_visual_teeth = 16; // visual coupling only; module and profile TBD

module engineering_notice(label) {
    // Console labels remain visible when reviewing individual modules.
    echo(str("ENGINEERING PROTOTYPE NOT FABRICATION RELEASE: ", label));
}

module motor_envelope(label = "MOTOR") {
    engineering_notice(str(label, " clearance envelope only"));
    color([0.70, 0.74, 0.78, 0.70])
        cube(motor_envelope_mm, center = true);
}

module motor_at(center_mm, label = "MOTOR", orientation_deg = [0, 0, 0]) {
    translate(center_mm)
        rotate(orientation_deg)
            motor_envelope(label);
}

module link_blank(axis_distance_mm, label = "LINK BLANK") {
    engineering_notice(str(label, " - TBD mating machining"));
    color([0.88, 0.48, 0.12])
        hull() {
            cylinder(h = link_thickness_mm, d = link_width_mm, center = true);
            translate([axis_distance_mm, 0, 0])
                cylinder(h = link_thickness_mm, d = link_width_mm, center = true);
        }
}

module dual_side_support_blank(label = "SUPPORT BLANK") {
    engineering_notice(str(label, " - solid, no holes"));
    inner_y = motor_envelope_mm[0] / 2 + support_clearance_mm;
    height = motor_envelope_mm[1] + 2 * support_wall_mm;
    color([0.96, 0.78, 0.25], 0.80) {
        translate([0, inner_y + support_wall_mm / 2, 0])
            cube([motor_envelope_mm[2] + 8, support_wall_mm, height], center = true);
        translate([0, -inner_y - support_wall_mm / 2, 0])
            cube([motor_envelope_mm[2] + 8, support_wall_mm, height], center = true);
        translate([0, 0, -height / 2 + support_wall_mm / 2])
            cube(
                [
                    motor_envelope_mm[2] + 8,
                    2 * (inner_y + support_wall_mm),
                    support_wall_mm
                ],
                center = true
            );
    }
}

module visual_equal_gear(label = "GEAR") {
    engineering_notice(str(label, " visual pitch geometry; teeth TBD"));
    color([0.35, 0.55, 0.72]) {
        cylinder(h = 4, r = gear_pitch_radius_mm - 1, center = true);
        for (tooth = [0 : gear_visual_teeth - 1])
            rotate([0, 0, tooth * 360 / gear_visual_teeth])
                translate([gear_pitch_radius_mm - 0.5, 0, 0])
                    cube([3, 2.2, 4], center = true);
    }
}

module jaw_blank(label = "JAW BLANK") {
    engineering_notice(str(label, " - rotary, fingertip and retention TBD"));
    color([0.20, 0.68, 0.46])
        translate([jaw_length_mm / 2, 0, 0])
            cube([jaw_length_mm, 5, 6], center = true);
}

module geared_gripper(left_q_rad = gripper_left_q_rad) {
    engineering_notice("equal gears ratio -1; axes y +/-12 mm, parallel +Z");
    assert(
        jaw_coupling_ratio == -1,
        "canonical gripper coupling must remain right = -left"
    );
    assert(
        left_q_rad >= gripper_left_range_rad[0]
        && left_q_rad <= gripper_left_range_rad[1],
        "left gripper q outside canonical [-0.32, 0] rad range"
    );
    right_q_rad = -left_q_rad;
    assert(
        right_q_rad >= gripper_right_range_rad[0]
        && right_q_rad <= gripper_right_range_rad[1],
        "right gripper q outside canonical [0, 0.32] rad range"
    );
    // Looking down +Z: left hinge is at +Y and has q in [-0.32, 0].
    // Right hinge is at -Y, q_right = -q_left in [0, 0.32].
    translate([0, jaw_center_offset_mm, 0])
        rotate([0, 0, left_q_rad * 180 / PI]) {
            visual_equal_gear("DRIVEN GEAR A");
            jaw_blank("JAW A");
        }
    translate([0, -jaw_center_offset_mm, 0])
        rotate([0, 0, right_q_rad * 180 / PI]) {
            visual_equal_gear("FOLLOWER GEAR B");
            jaw_blank("JAW B");
        }
    motor_at(
        servo_center_gripper_local_mm,
        "M5 ROTARY GRIPPER - ABOVE HAND",
        [0, 90, 0]
    );
}

module arm_cartridge() {
    engineering_notice(str("arm cartridge target ", arm_cartridge_target_g, " g - unverified"));
    engineering_notice("overall packaging and table clearance unverified");

    // M1 base yaw envelope and 20 g target trunk blank.
    color([0.95, 0.78, 0.25], 0.85)
        translate([0, 0, -22])
            cube([42, 42, 10], center = true);
    motor_at(servo_center_yaw_local_mm, "M1 BASE YAW");

    rotate([0, 0, pose_deg[0]])
        translate([0, 0, 0]) {
            dual_side_support_blank("SHOULDER SUPPORT BLANK");
            rotate([0, pose_deg[1], 0]) {
                motor_at(
                    servo_center_shoulder_local_mm,
                    "M2 SHOULDER",
                    [90, 0, 0]
                );
                link_blank(shoulder_elbow_mm, "UPPER LINK BLANK");
                motor_at(
                    servo_center_elbow_local_mm,
                    "M3 ELBOW IN UPPER-ARM FRAME",
                    [90, 0, 0]
                );

                translate([shoulder_elbow_mm, 0, 0])
                    rotate([0, pose_deg[2], 0]) {
                        dual_side_support_blank("ELBOW SUPPORT BLANK");
                        link_blank(elbow_wrist_mm, "FOREARM LINK BLANK");
                        motor_at(
                            servo_center_wrist_local_mm,
                            "M4 WRIST IN FOREARM FRAME",
                            [90, 0, 0]
                        );

                        translate([elbow_wrist_mm, 0, 0])
                            rotate([0, pose_deg[3], 0]) {
                                dual_side_support_blank("WRIST SUPPORT BLANK");
                                geared_gripper();
                            }
                    }
            }
        }
}

if (render_part == "assembly") {
    arm_cartridge();
} else if (render_part == "gripper") {
    geared_gripper();
} else if (render_part == "support_blanks") {
    dual_side_support_blank();
    translate([45, 0, 0])
        link_blank(shoulder_elbow_mm);
    translate([45, 24, 0])
        link_blank(elbow_wrist_mm);
} else if (render_part == "motor_envelope") {
    motor_envelope();
}
