from __future__ import annotations

import numpy as np
import pytest

from rlx.environments import arm


@pytest.mark.parametrize("boundary", [5.0, 8.0, 14.0, 17.0])
def test_co_carry_cartesian_phase_boundary_has_no_target_jump(boundary):
    environment = arm.ArmEnv("arms-co-carry-v1")
    try:
        environment.reset(seed=80307, options={"position_noise": 0.002})
        environment.steps = round(boundary / arm.CONTROL_DT) - 1
        before = environment.teacher_targets()
        environment.steps += 1
        after = environment.teacher_targets()
        for arm_index in range(2):
            assert np.linalg.norm(after[arm_index][0] - before[arm_index][0]) < 0.000001
            assert before[arm_index][1] == 0.012
            assert after[arm_index][1] == (0.030 if boundary == 17.0 else 0.012)
    finally:
        environment.close()


def test_co_carry_trajectory_preserves_offsets_and_bounded_speed():
    environment = arm.ArmEnv("arms-co-carry-v1")
    try:
        environment.reset(seed=80609, options={"friction_scale": 0.9})
        previous = None
        for step in range(round(5 / arm.CONTROL_DT), round(17 / arm.CONTROL_DT) + 1):
            environment.steps = step
            targets = environment.teacher_targets()
            np.testing.assert_allclose(targets[1][0] - targets[0][0], [0, 0.08, 0], atol=1e-12)
            if previous is not None:
                speed = np.linalg.norm(targets[0][0] - previous) / arm.CONTROL_DT
                assert speed <= 0.021
            previous = targets[0][0]
        np.testing.assert_allclose(targets[0][0], environment.goal + environment.grasp_offsets[0], atol=1e-12)
        assert targets[0][1] == 0.030
    finally:
        environment.close()
