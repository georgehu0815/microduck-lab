/*
Generated from hardware/microduck-arm-v1/spec/design.json.
Canonical spec SHA-256: 57aed79ec22c5e651950271e089a578766dc49e7a30961353c9d3a308688a50d.
ENGINEERING PROTOTYPE NOT FABRICATION RELEASE
Do not hand-edit; regenerate with cad/generate_engineering.py.
*/
design_revision = "engineering-prototype-1";
motor_model = "XL330-M288-T";
motor_count = 5;
motor_mass_g = 18;
motor_envelope_mm = [20, 34, 26];
servo_center_yaw_local_mm = [0, 0, -25];
servo_center_shoulder_local_mm = [0, 0, 18];
servo_center_elbow_local_mm = [45, 0, 20];
servo_center_wrist_local_mm = [25, 0, 20];
servo_center_gripper_local_mm = [7, 0, 22];
shoulder_elbow_mm = 55;
elbow_wrist_mm = 50;
wrist_tcp_mm = 30;
jaw_center_offset_mm = 12;
jaw_length_mm = 30;
jaw_coupling_ratio = -1;
gripper_max_closure_rad = 0.32;
gripper_left_range_rad = [-0.32, 0];
gripper_right_range_rad = [0, 0.32];
trunk_mount_target_g = 20;
arm_cartridge_target_g = 160;
