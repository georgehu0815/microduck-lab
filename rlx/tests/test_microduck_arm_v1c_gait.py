from __future__ import annotations

import mujoco
import numpy as np
import pytest

from microduck_arm_v1c.env import IntegratedArmEnv
from microduck_arm_v1c.gait import GaitPhase, IndependentWalkingController


def _single_support(env):
    active = set()
    force = np.zeros(6)
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        names = {
            env.model.geom(contact.geom1).name,
            env.model.geom(contact.geom2).name,
        }
        if "floor" not in names:
            continue
        mujoco.mj_contactForce(env.model, env.data, index, force)
        if force[0] > 0.01:
            bodies = {
                int(env.model.geom_bodyid[contact.geom1]),
                int(env.model.geom_bodyid[contact.geom2]),
            }
            active.update(bodies & env.foot_body_ids)
    return next(iter(active)) if len(active) == 1 else None


def test_reset_and_leg_target_contract_are_deterministic():
    env = IntegratedArmEnv(
        case="stowed_arm_walking", mode="free", candidate="A", max_steps=5
    )
    env.reset(seed=0)
    controller = IndependentWalkingController()

    first = controller.leg_targets(env)
    assert first.shape == (10,)
    assert np.isfinite(first).all()
    assert np.all(first >= env.joint_limits[:10, 0])
    assert np.all(first <= env.joint_limits[:10, 1])

    controller.reset()
    second = controller.leg_targets(env)
    np.testing.assert_array_equal(first, second)
    assert controller.active_phase == GaitPhase.CLEAR_OBSTACLE
    assert controller.provenance["leg_targets_only"] is True


@pytest.mark.parametrize("leg_action_scale_rad", [0.5, 1.0])
def test_controller_clears_table_and_meets_original_walking_success(
    leg_action_scale_rad,
):
    env = IntegratedArmEnv(
        case="stowed_arm_walking", mode="free", candidate="A", max_steps=400
    )
    env.reset(seed=0)
    assert env.model.opt.noslip_iterations == 10
    env.action_scales[:10] = leg_action_scale_rad
    controller = IndependentWalkingController()

    longest_support_s = 0.0
    current_support = None
    current_ticks = 0
    max_translation_m = 0.0
    max_tilt_rad = 0.0
    max_arm_error_rad = 0.0
    max_raw_requested_offset_rad = 0.0
    saturated_frames = 0
    transition = None
    info = None
    for _ in range(400):
        target = env.home.copy()
        target[:10] = controller.leg_targets(env)
        if controller.phase == GaitPhase.ADVANCE and transition is None:
            transition = (
                float(env.data.xpos[env.trunk_id, 1] - env.trunk_start[1]),
                controller.last_clearance_distance_m,
            )
        max_raw_requested_offset_rad = max(
            max_raw_requested_offset_rad,
            float(
                np.max(
                    np.abs(controller.last_raw_targets - env.home[:10])
                )
            ),
        )
        saturated_frames += controller.last_saturated_joint_count > 0
        action = env.normalize_targets(target)
        _, _, terminated, truncated, info = env.step(action)
        support = _single_support(env)
        if support is not None and support == current_support:
            current_ticks += 1
        else:
            longest_support_s = max(longest_support_s, current_ticks * env.dt)
            current_support = support
            current_ticks = 1 if support is not None else 0
        max_translation_m = max(
            max_translation_m,
            float(
                np.linalg.norm(
                    env.data.xpos[env.trunk_id, :2] - env.trunk_start[:2]
                )
            ),
        )
        max_tilt_rad = max(max_tilt_rad, info["diagnostics"]["tilt_rad"])
        max_arm_error_rad = max(
            max_arm_error_rad,
            float(
                np.max(
                    np.abs(
                        env.data.qpos[env.qpos_indices[10:]]
                        - env.home[10:]
                    )
                )
            ),
        )
        if terminated or truncated:
            break
    longest_support_s = max(longest_support_s, current_ticks * env.dt)

    assert longest_support_s >= 0.10
    assert max_tilt_rad < 0.25
    assert max_translation_m > 0.05
    assert max_arm_error_rad < 0.10
    assert info is not None
    assert info["success"]
    assert info["failure_reason"] is None
    assert info["diagnostics"]["forward_displacement_m"] >= 0.05
    assert len(env.support_events) >= 3
    assert controller.phase == GaitPhase.ADVANCE
    assert transition is not None
    assert transition[0] <= controller.clearance_y_m
    assert transition[1] >= controller.clearance_distance_m

    # The legacy policy routinely requests a wider range than this controller
    # can safely execute on v1-C, so saturation is explicit in the controller
    # rather than an accidental side effect of the environment encoding.
    assert saturated_frames / env.steps > 0.50
    assert max_raw_requested_offset_rad > 1.0
