"""WingPod render sidecar: decoration never enters the dynamics model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools

import mujoco
import numpy as np

from microduck_arm_experiments.tennis_return import TennisReturnEnv


CREAM = (0.94, 0.91, 0.82, 1.0)
HONEY = (1.0, 0.66, 0.16, 1.0)
INK = (0.16, 0.20, 0.22, 1.0)
WHITE = (1.0, 0.99, 0.94, 1.0)
VISUAL_ARRAYS = frozenset({"geom_rgba", "mat_rgba"})


def physics_fingerprint(model):
    digest = hashlib.sha256()
    for name in sorted(dir(model)):
        value = getattr(model, name)
        if isinstance(value, np.ndarray) and name not in VISUAL_ARRAYS:
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
    for name in ("nq", "nv", "nu", "nbody", "njnt", "ngeom"):
        digest.update(f"{name}={getattr(model, name)}".encode())
    for name in sorted(dir(model.opt)):
        if not name.startswith("_"):
            value = getattr(model.opt, name)
            if isinstance(value, (int, float, np.ndarray)):
                digest.update(name.encode())
                digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Pod:
    name: str
    body: str
    center: tuple
    half_size: tuple
    radius: float
    color: tuple = CREAM


PODS = (
    Pod("shoulder_saddle", "arm_mount", (0, 0, -.008), (.025, .025, .004), .003),
    Pod("root", "arm_mount", (0, 0, -.025), (.013, .016, .020), .006),
    Pod("shoulder", "arm_yaw", (0, 0, .018), (.013, .016, .020), .006),
    Pod("upper_knuckle", "arm_upper", (.045, 0, .020), (.013, .016, .020), .006),
    Pod("forearm_knuckle", "arm_forearm", (.025, 0, .020), (.013, .016, .020), .006),
    Pod("palm", "arm_hand", (.007, 0, .022), (.013, .016, .020), .006),
    Pod("backpack", "assumed_dedicated_arm_pack", (0, 0, 0), (.018, .015, .011), .003),
    Pod("electronics", "assumed_relocated_electronics", (0, 0, 0), (.021, .013, .005), .002),
    Pod("power", "assumed_power_and_communications", (0, 0, 0), (.015, .011, .005), .002),
    Pod("face", "assumed_sensor_shell", (0, 0, 0), (.019, .017, .013), .006),
)


@dataclass(frozen=True)
class Detail:
    name: str
    body: str
    kind: str
    center: tuple
    size: tuple
    color: tuple
    rotate_x_deg: float = 0


def design_details():
    details = [
        Detail("upper_feather", "arm_upper", "ellipsoid", (.025, 0, .004), (.026, .008, .009), HONEY),
        Detail("forearm_feather", "arm_forearm", "ellipsoid", (.023, 0, .004), (.023, .008, .009), HONEY),
        Detail("beak", "assumed_sensor_shell", "ellipsoid", (.019, 0, -.004), (.004, .008, .003), HONEY),
    ]
    for pod in PODS[2:6]:
        for side in (-1, 1):
            center = (pod.center[0], side * .0165, pod.center[2])
            details.append(Detail(f"{pod.name}_ring_{side}", pod.body, "cylinder", center, (.006, .0007, 0), INK, 90))
            center = (pod.center[0], side * .0173, pod.center[2])
            details.append(Detail(f"{pod.name}_cap_{side}", pod.body, "cylinder", center, (.0045, .0002, 0), CREAM, 90))
    for side in (-1, 1):
        details.append(Detail(f"eye_{side}", "assumed_sensor_shell", "ellipsoid",
                              (.0189, side * .008, .004), (.0015, .0025, .0035), INK))
        details.append(Detail(f"eye_glint_{side}", "assumed_sensor_shell", "ellipsoid",
                              (.0202, side * .008 - .0005, .005), (.0005, .0007, .0009), WHITE))
    return tuple(details)


DETAILS = design_details()


def appearance_spec():
    return {
        "name": "MicroDuck WingPod / 奶油小鸭",
        "status": "RENDER_ONLY_PHYSICAL_SHELL_UNVERIFIED",
        "units": "metres",
        "pods": [asdict(pod) for pod in PODS],
        "details": [asdict(detail) for detail in DETAILS],
        "existing_geometry_treatment": "Palette only; jaw bridges and stems honey, contact pads unchanged",
        "wall_candidate_m": .0012,
        "physical_mass_kg": None,
        "physical_clearance_verified": False,
        "hardware_release": False,
    }


def apply_palette(model):
    before = physics_fingerprint(model)
    for index in range(model.nmat):
        name = model.mat(index).name
        if "material" not in name:
            continue
        color = CREAM
        if any(word in name for word in ("sole", "xl330", "np_f970")):
            color = INK
        elif "foot_" in name:
            color = HONEY
        model.mat_rgba[index] = color
    for index in range(model.ngeom):
        name = model.geom(index).name
        if name.startswith("arm_pad"):
            continue
        if name.startswith("assumed_") or name.endswith("_motor") or name == "arm_mount_plate":
            model.geom_rgba[index, 3] = 0
        elif name.startswith("arm_"):
            model.geom_rgba[index] = HONEY if any(word in name for word in ("link", "stem", "bridge")) else INK
    if physics_fingerprint(model) != before:
        raise ValueError("Appearance changed the physical model")
    return before


class WingPodOverlay:
    def __init__(self, model):
        self.model = model
        self.body_ids = {pod.body: model.body(pod.body).id for pod in PODS}
        for name in ("arm_upper", "arm_forearm", "arm_hand"):
            self.body_ids[name] = model.body(name).id

    def _geom(self, scene, data, body, kind, center, size, color, rotation=None):
        if scene.ngeom >= scene.maxgeom:
            raise ValueError("Insufficient decorative scene capacity")
        body_id = self.body_ids[body]
        basis = data.xmat[body_id].reshape(3, 3)
        orientation = basis if rotation is None else basis @ rotation
        position = data.xpos[body_id] + basis @ np.asarray(center)
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, kind, np.array(size, dtype=np.float64), position,
                           orientation.ravel(), np.array(color, dtype=np.float32))
        geom.shininess = .25
        geom.specular = .18
        scene.ngeom += 1

    def _rounded(self, scene, data, pod):
        inner = np.array(pod.half_size) - pod.radius
        if np.any(inner <= 0):
            raise ValueError("Fillet must be smaller than every half dimension")
        center = np.array(pod.center)
        for axis in range(3):
            size = np.array(pod.half_size)
            size[axis] = inner[axis]
            for other in range(3):
                if other != axis:
                    size[other] = inner[other]
            size[axis] += pod.radius
            self._geom(scene, data, pod.body, mujoco.mjtGeom.mjGEOM_BOX, center, size, pod.color)
        for axis in range(3):
            others = [other for other in range(3) if other != axis]
            orientation = np.eye(3)[:, [1, 2, 0] if axis == 0 else [2, 0, 1] if axis == 1 else [0, 1, 2]]
            for signs in itertools.product((-1, 1), repeat=2):
                offset = center.copy()
                for other, sign in zip(others, signs):
                    offset[other] += sign * inner[other]
                self._geom(scene, data, pod.body, mujoco.mjtGeom.mjGEOM_CAPSULE,
                           offset, (pod.radius, inner[axis], 0), pod.color, orientation)

    def update(self, scene, data):
        for pod in PODS:
            self._rounded(scene, data, pod)
        for detail in DETAILS:
            angle = np.deg2rad(detail.rotate_x_deg)
            rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                                 [0, np.sin(angle), np.cos(angle)]])
            kind = mujoco.mjtGeom.mjGEOM_ELLIPSOID if detail.kind == "ellipsoid" else mujoco.mjtGeom.mjGEOM_CYLINDER
            self._geom(scene, data, detail.body, kind, detail.center, detail.size, detail.color, rotation)


class WingPodEnv(TennisReturnEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.physical_hash = apply_palette(self.robot.model)
        self.overlay = WingPodOverlay(self.robot.model)

    def render(self):
        robot = self.robot
        if robot._renderer is None:
            robot._renderer = mujoco.Renderer(robot.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.18, .03, .10]
        camera.distance, camera.azimuth, camera.elevation = .95, 125, -28
        robot._renderer.update_scene(robot.data, camera=camera)
        self.overlay.update(robot._renderer.scene, robot.data)
        return robot._renderer.render().copy()
