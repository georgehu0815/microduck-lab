from __future__ import annotations

import pytest

from microduck_arm_v1c.config import (
    BASE_ENV_MODULE,
    BASE_MODEL_MODULE,
    CASES,
    DESIGN_ID,
    ROOT,
    SPEC_PATH,
    VENDOR_SOURCE_MANIFEST,
    load_design,
    sha256,
)


def test_canonical_api_and_default_candidate():
    design = load_design()

    assert DESIGN_ID == "microduck-single-arm-v1-c"
    assert SPEC_PATH == ROOT / "hardware/microduck-arm-v1c/spec/design.json"
    assert len(sha256(SPEC_PATH)) == 64
    assert len(CASES) == 6
    assert design["candidate"] == "A"
    assert design["candidate_configuration"]["label"] == "5M288"
    assert design["simulation_base"]["model_module"] == BASE_MODEL_MODULE
    assert design["simulation_base"]["env_module"] == BASE_ENV_MODULE


def test_joint_motor_action_and_provisional_bus_mapping():
    design = load_design()

    for index, item in enumerate(design["actuator_mapping"]):
        assert item == {
            "joint_id": f"J{index + 1}",
            "motor_id": f"M{index + 1}",
            "joint_name": design["joint_order"][index + 10],
            "action_index": index + 10,
            "provisional_bus_id": index + 21,
        }
    assert design["bus_id_policy"]["actual_id_evidence_status"] == "unverified"
    assert design["bus_id_policy"]["physical_pin_mapping_status"] == "unreleased"
    assert not design["bus_id_policy"]["real_io_implemented"]


def test_candidate_motor_assignments_and_unknown_ratings():
    candidate_a = load_design("A")
    candidate_b = load_design("B")

    assert list(candidate_a["arm"]["motor_skus_by_motor"].values()) == [
        "XL330-M288-T"
    ] * 5
    assert list(candidate_b["arm"]["motor_skus_by_motor"].values()) == [
        "XL330-M288-T",
        "XL330-M288-T",
        "XL330-M288-T",
        "XL330-M077-T",
        "XL330-M077-T",
    ]
    assert candidate_a["arm"]["motor_total_mass_kg"] == pytest.approx(0.09)
    assert candidate_b["arm"]["motor_total_mass_kg"] == pytest.approx(0.09)
    for motor in candidate_a["motor_catalog"].values():
        assert motor["continuous_torque_rating_nm"] is None
        assert motor["continuous_speed_rating_rpm"] is None
        assert "stall torque" in motor["rating_note"]
        assert "no-load speed" in motor["rating_note"]
    policy = candidate_a["candidate_comparison_policy"]
    assert not policy["automatic_rotor_inertia_superiority"]
    assert not policy["automatic_energy_superiority"]


def test_corrected_complete_cartridge_mass_budget():
    design = load_design()
    bom = design["engineering_bom"]
    categories = bom["categories"]

    assert bom["original_user_sum_kg"] == pytest.approx(0.17)
    assert bom["target_total_kg"] == pytest.approx(0.16)
    assert sum(item["mass_kg"] for item in categories.values()) == pytest.approx(
        0.16
    )
    assert categories["servos"]["mass_kg"] == pytest.approx(0.09)
    assert categories["links"]["mass_kg"] == pytest.approx(0.02)
    assert categories["reserve"]["interface"] == (
        "not_a_separate_electronics_allowance"
    )


def test_full_placeholder_inertia_is_explicit_and_fails_release():
    design = load_design()
    mass_properties = design["mass_properties"]

    assert mass_properties["principal_inertia_release_gate"] == "fail"
    assert mass_properties["status"] == (
        "analytical_box_estimate_not_CAD_or_measured"
    )
    assert sum(
        component["mass_kg"] for component in mass_properties["components"]
    ) == pytest.approx(0.16)
    for component in mass_properties["components"]:
        tensor = component["inertia_tensor_kg_m2"]
        if tensor is None:
            assert component["status"].startswith("unqualified_")
        else:
            assert component["status"] == (
                "analytical_box_estimate_not_CAD_or_measured"
            )
            assert len(tensor) == 3
            assert all(len(row) == 3 for row in tensor)
            assert all(tensor[index][index] > 0 for index in range(3))


def test_equal_gear_candidate_has_24mm_jaw_center_distance():
    gear = load_design()["arm"]["gripper_gear_candidate"]

    calculated = (
        gear["module_m"] * (gear["left_teeth"] + gear["right_teeth"]) / 2
    )
    assert calculated == pytest.approx(0.024)
    assert gear["module_m"] * gear["left_teeth"] == pytest.approx(0.024)
    assert gear["ratio"] == -1.0
    assert gear["status"] == "kinematic_candidate_not_physical_release"


def test_v1_c_geometry_and_raised_table_scene_are_explicit():
    design = load_design()
    arm = design["arm"]
    control = design["control"]
    scene = design["simulation_scene"]

    assert arm["shoulder_pivot_local_m"] == [0, 0, 0.018]
    assert arm["shoulder_pivot_local_m"] == arm["servo_centers_local_m"]["shoulder"]
    assert control["arm_joint_slew_limit_rad_s"] == pytest.approx(0.4)
    assert control["leg_joint_slew_limit_rad_s"] == pytest.approx(6.0)
    assert control["slew_limit_status"] == (
        "simulation_screening_only_not_hardware_qualified"
    )
    assert scene["scenario_id"] == "raised_tabletop_v1_c"
    assert scene["fixture_classification"] == (
        "raised_table_fixture_screening_not_original_task"
    )
    assert scene["surface_top_m"] == pytest.approx(0.145)
    assert scene["tabletop_top_z_m"] == pytest.approx(0.145)
    assert scene["object_initial_center_m"] == pytest.approx([0.115, -0.018, 0.151])
    assert scene["goal_center_m"] == pytest.approx([0.125, 0.024, 0.151])
    assert scene["transfer_waypoint_z_m"] == pytest.approx(0.190)
    assert scene["object_initial_center_m"][2] == pytest.approx(
        scene["tabletop_top_z_m"] + scene["object_half_height_m"]
    )
    assert not scene["old_floor_level_success_transferable"]
    assert scene["distinct_from"] == "v1_b_floor_level_task_scene"


def test_contact_pad_and_teacher_feedforward_contract():
    arm = load_design()["arm"]
    pads = arm["contact_pads"]
    compensation = arm["gravity_compensation"]

    assert pads == {
        "count": 2,
        "shape": "analytical_box",
        "full_size_m": [0.024, 0.005, 0.008],
        "center_local_m": [0.023, 0, 0],
        "mass_each_kg": 0.005,
        "solref": [0.004, 1],
        "solimp": [0.95, 0.99, 0.001],
        "status": "unmeasured_contact_screening",
    }
    assert compensation["formula"] == (
        "clip(qfrc_bias / simulation_kp, -0.04, 0.04)"
    )
    assert compensation["applies_to_action_indices"] == [10, 11, 12, 13]
    assert compensation["maximum_position_offset_rad"] == pytest.approx(0.04)
    assert compensation["state_source"] == "simulator_truth_qfrc_bias"
    assert compensation["hardware_status"] == "unqualified"
    assert compensation["not_direct_torque_feedforward"]


def test_simulation_mass_ledger_is_derived_not_target_forced():
    design = load_design()
    ledger = design["simulation_mass_ledger"]

    assert sum(ledger["components_kg"].values()) == pytest.approx(0.166)
    assert ledger["modeled_arm_geoms_including_mount_kg"] == pytest.approx(0.166)
    assert ledger["modeled_arm_geoms_excluding_mount_kg"] == pytest.approx(0.146)
    assert ledger["engineering_target_kg"] == pytest.approx(0.160)
    assert ledger["target_enforcement"] == (
        "not_forced_to_match_simulation_geom_mass"
    )
    assert design["arm"]["mount_mass_target_kg"] == pytest.approx(0.020)
    assert "separate_from_0.010_kg_engineering_interface_budget" in (
        design["arm"]["mount_mass_target_status"]
    )


def test_vendor_sources_keep_ratings_separate_from_shop_references():
    design = load_design()
    provenance = design["source_provenance"]

    assert VENDOR_SOURCE_MANIFEST.is_file()
    assert provenance["vendor_source_manifest"] == (
        "hardware/microduck-arm-v1c/vendor/SOURCE-MANIFEST.md"
    )
    assert provenance["manufacturer_rating_authority"] == (
        "official_ROBOTIS_eManual_model_pages"
    )
    assert not provenance["shop_rating_values_ingested"]
    assert "must_not_be_merged" in provenance["shop_reference_policy"]
    for motor in design["motor_catalog"].values():
        assert motor["rating_source_class"] == "official_manufacturer_emanual"
        assert "emanual.robotis.com" in motor["rating_source_url"]


def test_power_release_and_safety_gates_fail_closed():
    design = load_design()
    power = design["power"]

    assert power["default"]["topology"] == "bench_external_regulated_supply"
    shared = power["alternatives"]["shared_main_pack"]
    assert shared["status"] == "research_candidate_not_automatically_approved"
    assert "buck_boost" in shared["unknown_pack_voltage_rule"]
    assert power["alternatives"]["standalone_pack"]["bms_required"]
    assert design["arm"]["required_continuous_torque_margin"] == 3.0
    assert design["safety_gates"]["software_heartbeat"][
        "separate_from_command_watchdog"
    ]
    assert not design["safety_gates"]["real_io"]["implemented"]
    assert design["release_states"]["relationship"].startswith(
        "orthogonal_evidence_axes"
    )


@pytest.mark.parametrize("candidate", ["", "C", "a", None, 1])
def test_invalid_candidate_fails_closed(candidate):
    with pytest.raises(ValueError, match="Unknown v1-C candidate"):
        load_design(candidate)


def test_load_returns_independent_normalized_copies():
    first = load_design()
    first["arm"]["link_lengths_m"][0] = 999

    second = load_design()
    assert second["arm"]["link_lengths_m"] == [0.055, 0.05, 0.03]
