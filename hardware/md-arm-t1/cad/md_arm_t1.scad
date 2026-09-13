/*
MD-Arm-T1 editable concept CAD.

This source is intentionally non-fabrication-ready:
- no XL330 mounting holes
- no horn bolt circle or pilot
- no bearing seats
- no base clamp interface

Import the official ROBOTIS XL330 STEP and complete a measured tolerance review
before creating any mating feature.
*/

$fn = 48;

include <../generated/shared-xl330.scad>

render_part = "assembly"; // assembly, station_layouts, link_blanks, catch_tray

upper_arm_axis_mm = 65;
forearm_axis_mm = 55;
wrist_to_tcp_mm = 30;
link_width_mm = 12;
link_thickness_mm = 5;


tray_width_mm = 250;
tray_depth_mm = 200;
tray_base_mm = 2.4;
tray_wall_height_mm = 20;
tray_wall_mm = 2.4;

historical_station_nominal_mm = 220;
historical_station_adjustable_min_mm = 180;
historical_station_adjustable_max_mm = 240;
handover_sim_base_spacing_mm = 150;
co_carry_sim_base_spacing_mm = 242;
required_station_adjustable_min_mm = 150;
required_station_adjustable_max_mm = 242;

module servo_envelope() {
    color([0.75, 0.75, 0.78, 0.65])
        microduck_xl330_reference();
}

module link_blank(axis_distance_mm) {
    // Blank stock only; both ends remain solid until interface release.
    hull() {
        translate([0, 0, 0])
            cylinder(h=link_thickness_mm, d=link_width_mm, center=true);
        translate([axis_distance_mm, 0, 0])
            cylinder(h=link_thickness_mm, d=link_width_mm, center=true);
    }
}

module arm_envelope() {
    color("gray") translate([0, 0, servo_h_mm / 2]) servo_envelope();
    color("orange") translate([0, 0, servo_h_mm])
        rotate([0, 0, 18]) link_blank(upper_arm_axis_mm);
    color("gray") translate([61.82, 20.09, servo_h_mm]) servo_envelope();
    color("gold") translate([61.82, 20.09, servo_h_mm])
        rotate([0, 0, -22]) link_blank(forearm_axis_mm);
    color("gray") translate([112.82, -0.50, servo_h_mm]) servo_envelope();
    color("lightgreen") translate([112.82, -0.50, servo_h_mm])
        rotate([0, 0, 8]) link_blank(wrist_to_tcp_mm);
}

module catch_tray() {
    union() {
        cube([tray_width_mm, tray_depth_mm, tray_base_mm]);
        cube([tray_width_mm, tray_wall_mm, tray_wall_height_mm]);
        translate([0, tray_depth_mm - tray_wall_mm, 0])
            cube([tray_width_mm, tray_wall_mm, tray_wall_height_mm]);
        cube([tray_wall_mm, tray_depth_mm, tray_wall_height_mm]);
        translate([tray_width_mm - tray_wall_mm, 0, 0])
            cube([tray_wall_mm, tray_depth_mm, tray_wall_height_mm]);
    }
}

module station_layout(base_spacing_mm) {
    // Base blocks are conceptual envelopes only. No clamp or mounting holes.
    translate([0, -base_spacing_mm / 2, 0])
        color("gray") cube([46, 46, 8], center=true);
    translate([0, base_spacing_mm / 2, 0])
        color("gray") cube([46, 46, 8], center=true);
}

if (render_part == "assembly") {
    arm_envelope();
} else if (render_part == "shared_motor") {
    microduck_xl330_reference();
} else if (render_part == "station_layouts") {
    station_layout(handover_sim_base_spacing_mm);
    translate([100, 0, 0])
        station_layout(co_carry_sim_base_spacing_mm);
} else if (render_part == "link_blanks") {
    link_blank(upper_arm_axis_mm);
    translate([0, 25, 0]) link_blank(forearm_axis_mm);
    translate([0, 50, 0]) link_blank(wrist_to_tcp_mm);
} else if (render_part == "catch_tray") {
    catch_tray();
}
