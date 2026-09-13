servo_w_mm = 20.0;
servo_h_mm = 34.0;
servo_d_mm = 26.0;
module microduck_xl330_reference() {
    rotate([0, 0, 90])
        translate([-0.0, -0.0, 7.499999832361937])
            scale([1000, 1000, 1000])
                import("../../../rlx/rlx/mjlab_microduck/robot/microduck/assets/xl330.stl");
}
