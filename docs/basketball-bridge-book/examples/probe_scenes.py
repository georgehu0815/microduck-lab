"""Compile both real scenes and check four transitions, without claiming mastery."""

import json

import numpy as np

from rlx.environments.basketball import BasketballEnv
from rlx.environments.bridge import BridgeEnv


def main():
    results = []
    for name, environment_class in (("basketball", BasketballEnv), ("bridge", BridgeEnv)):
        environment = environment_class(actuator="xml", random_yaw=False)
        try:
            observation, _ = environment.reset(seed=101)
            assert observation.shape == (61,)
            assert environment.action_space.shape == (14,)
            assert environment.model.nu == 14
            for _ in range(4):
                observation, reward, terminated, truncated, _ = environment.step(
                    np.zeros(14, dtype=np.float32)
                )
                assert np.isfinite(observation).all()
                assert np.isfinite(reward)
                if terminated or truncated:
                    raise AssertionError("probe ended before four transitions")
            results.append({
                "scene": name,
                "nq": environment.model.nq,
                "nv": environment.model.nv,
                "actuators": environment.model.nu,
                "tendons": environment.model.ntendon,
                "finite_transitions": 4,
                "actuator": "xml",
                "scope": "construction only, zero action, no learned policy",
            })
        finally:
            environment.close()
    print(json.dumps({"passed": True, "results": results}, indent=2))


if __name__ == "__main__":
    main()
