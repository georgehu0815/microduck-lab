from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

TRUSTED_RECIPES = frozenset({"backflip", "dance", "running", "stilts", "swing", "basketball", "bridge"})
DRAWING_RECIPE = "drawing"
DRAWING_CONTRACT_VERSION = "microduck-drawing-v1"
DRAWING_OBSERVATION_DIM = 83
DRAWING_ACTION_DIM = 15
BRUSH_CONTRACT_VERSION = "microduck-brush-v2"
BRUSH_OBSERVATION_DIM = 93
BRUSH_ACTION_DIM = 15


def studio_runs_root() -> Path:
    override = os.environ.get("MICRODUCK_STUDIO_RUNS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "rlx" / "runs" / "studio"


def is_studio_policy_path(path: str | Path) -> bool:
    policy_path = Path(path).expanduser().resolve()
    try:
        relative = policy_path.relative_to(studio_runs_root())
    except ValueError:
        return False
    return (
        len(relative.parts) == 3
        and relative.parts[0] in TRUSTED_RECIPES | {DRAWING_RECIPE}
        and (
            relative.parts[0] != DRAWING_RECIPE
            or relative.parts[2] == "policy.onnx"
        )
        and policy_path.suffix == ".onnx"
    )


def _validate_onnx_value(
    value: Any,
    *,
    expected_width: int,
    kind: str,
    policy_path: Path,
) -> None:
    import onnx

    tensor_type = value.type.tensor_type
    if tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError(f"Studio ONNX {policy_path} {kind} must use float32")
    dimensions = tensor_type.shape.dim
    if len(dimensions) != 2:
        raise ValueError(
            f"Studio ONNX {policy_path} {kind} must have rank 2, got {len(dimensions)}"
        )
    width = dimensions[1]
    if not width.HasField("dim_value") or width.dim_value != expected_width:
        raise ValueError(
            f"Studio ONNX {policy_path} {kind} width must be {expected_width}"
        )


def _drawing_contract_dimensions(
    metadata: dict[str, Any] | None,
    policy_path: Path,
) -> tuple[int, int]:
    if metadata is None:
        raise ValueError(
            f"Studio drawing ONNX {policy_path} requires embedded rlx_metadata"
        )
    contract_version = metadata.get("contract_version")
    if contract_version == DRAWING_CONTRACT_VERSION:
        return DRAWING_OBSERVATION_DIM, DRAWING_ACTION_DIM
    if contract_version == BRUSH_CONTRACT_VERSION:
        return BRUSH_OBSERVATION_DIM, BRUSH_ACTION_DIM
    raise ValueError(
        f"Studio drawing policy {policy_path} has unsupported contract_version "
        f"{contract_version!r}"
    )


def _validate_onnx_contract(
    model: Any,
    policy_path: Path,
    metadata: dict[str, Any] | None,
) -> None:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = [
        value for value in model.graph.input if value.name not in initializer_names
    ]
    outputs = list(model.graph.output)
    recipe = policy_path.parent.parent.name
    if recipe == "basketball":
        if [value.name for value in inputs] != ["obs", "h_in", "c_in"] or [value.name for value in outputs] != ["actions", "h_out", "c_out"]:
            raise ValueError("basketball ONNX requires obs/h_in/c_in and actions/h_out/c_out")
        for value in inputs[1:] + outputs[1:]:
            tensor = value.type.tensor_type
            if tensor.elem_type != 1 or [dimension.dim_value for dimension in tensor.shape.dim] != [1, 1, 256]:
                raise ValueError("basketball recurrent state must be float32 [1,1,256]")
        _validate_onnx_value(inputs[0], expected_width=61, kind="input", policy_path=policy_path)
        _validate_onnx_value(outputs[0], expected_width=14, kind="output", policy_path=policy_path)
        return
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"Studio ONNX {policy_path} must have exactly one input and one output"
        )
    observation_dim, action_dim = (
        _drawing_contract_dimensions(metadata, policy_path)
        if recipe == DRAWING_RECIPE
        else (61, 14)
    )
    _validate_onnx_value(
        inputs[0],
        expected_width=observation_dim,
        kind="input",
        policy_path=policy_path,
    )
    _validate_onnx_value(
        outputs[0],
        expected_width=action_dim,
        kind="output",
        policy_path=policy_path,
    )


def _load_onnx_metadata(policy_path: Path) -> dict[str, Any] | None:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "Studio policy metadata requires the installed 'onnx' dependency"
        ) from exc
    try:
        model = onnx.load(policy_path, load_external_data=False)
    except Exception as exc:
        raise ValueError(f"cannot read Studio ONNX {policy_path}: {exc}") from exc
    properties = {item.key: item.value for item in model.metadata_props}
    encoded = properties.get("rlx_metadata")
    metadata = None
    if encoded is not None:
        try:
            metadata = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid rlx_metadata in {policy_path}: {exc.msg}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"rlx_metadata in {policy_path} must be a JSON object")
    _validate_onnx_contract(model, policy_path, metadata)
    return metadata


def _sidecar_candidates(policy_path: Path) -> tuple[tuple[Path, Path], ...]:
    return (
        (
            policy_path.with_suffix(".safetensors.json"),
            policy_path.with_suffix(".safetensors"),
        ),
        (
            policy_path.with_suffix(".mlx.npz.json"),
            policy_path.with_suffix(".mlx.npz"),
        ),
    )


def _load_sidecar_metadata(policy_path: Path) -> tuple[dict[str, Any], Path]:
    for sidecar_path, checkpoint_path in _sidecar_candidates(policy_path):
        if not sidecar_path.is_file():
            continue
        if not checkpoint_path.is_file():
            raise ValueError(
                f"Studio metadata sidecar has no matching checkpoint: {sidecar_path}"
            )
        policy_mtime = policy_path.stat().st_mtime_ns
        newer = [
            path
            for path in (checkpoint_path, sidecar_path)
            if path.stat().st_mtime_ns > policy_mtime
        ]
        if newer:
            names = ", ".join(path.name for path in newer)
            raise ValueError(
                f"Studio checkpoint metadata is newer than {policy_path.name}: {names}; "
                "re-export the ONNX before loading it"
            )
        try:
            payload = json.loads(sidecar_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read Studio metadata {sidecar_path}: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
            raise ValueError(
                f"Studio metadata {sidecar_path} must contain a metadata object"
            )
        return payload["metadata"], sidecar_path
    expected = ", ".join(path.name for path, _ in _sidecar_candidates(policy_path))
    raise ValueError(
        f"Studio ONNX {policy_path} has no rlx_metadata and no matching sidecar "
        f"({expected})"
    )


def _path_stat_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _metadata_cache_key(policy_path: Path) -> tuple[Any, ...]:
    policy_signature = _path_stat_signature(policy_path)
    if policy_signature is None:
        raise FileNotFoundError(f"Studio policy does not exist: {policy_path}")
    related = tuple(
        (
            str(sidecar_path),
            _path_stat_signature(sidecar_path),
            str(checkpoint_path),
            _path_stat_signature(checkpoint_path),
        )
        for sidecar_path, checkpoint_path in _sidecar_candidates(policy_path)
    )
    return str(policy_path), policy_signature, related


def _load_metadata_with_source(
    policy_path: Path,
) -> tuple[dict[str, Any], Path | None]:
    if not policy_path.is_file():
        raise FileNotFoundError(f"Studio policy does not exist: {policy_path}")
    if not is_studio_policy_path(policy_path):
        raise ValueError(
            f"Studio policy must be root/recipe/run/*.onnx under {studio_runs_root()}: "
            f"{policy_path}"
        )
    embedded = _load_onnx_metadata(policy_path)
    recipe = policy_path.relative_to(studio_runs_root()).parts[0]
    if recipe == DRAWING_RECIPE and embedded is None:
        raise ValueError(
            f"Studio drawing ONNX {policy_path} requires embedded rlx_metadata"
        )
    metadata, source_path = (
        (embedded, None)
        if embedded is not None
        else _load_sidecar_metadata(policy_path)
    )
    relative = policy_path.relative_to(studio_runs_root())
    recipe = relative.parts[0]
    if metadata.get("recipe") != recipe:
        raise ValueError(
            f"Studio metadata recipe {metadata.get('recipe')!r} does not match "
            f"directory recipe {recipe!r}"
        )
    expected_actuators = {"xml", "bam"} if recipe == "basketball" else {"xml"}
    if metadata.get("actuator") not in expected_actuators:
        raise ValueError(
            f"Studio policy {policy_path} requires actuator in {sorted(expected_actuators)} metadata"
        )
    if not isinstance(metadata.get("recipe_options"), dict):
        raise ValueError(f"Studio policy {policy_path} is missing recipe_options")
    if not isinstance(metadata.get("reward_weights"), dict):
        raise ValueError(f"Studio policy {policy_path} is missing reward_weights")
    max_episode_s = metadata.get("max_episode_s")
    if (
        isinstance(max_episode_s, bool)
        or not isinstance(max_episode_s, (int, float))
        or not math.isfinite(max_episode_s)
        or max_episode_s <= 0
    ):
        raise ValueError(f"Studio policy {policy_path} has invalid max_episode_s")
    if recipe == DRAWING_RECIPE:
        _validate_drawing_metadata(metadata, policy_path)
    return metadata, source_path


def _drawing_module(contract_version: str):
    module_name = {
        DRAWING_CONTRACT_VERSION: "drawing",
        BRUSH_CONTRACT_VERSION: "brush",
    }.get(contract_version)
    if module_name is None:
        raise ValueError(f"unsupported drawing contract_version {contract_version!r}")
    try:
        return importlib.import_module(f"rlx.environments.{module_name}")
    except ImportError:
        _add_sibling_rlx_to_path()
        return importlib.import_module(f"rlx.environments.{module_name}")


def _validate_drawing_metadata(metadata: dict[str, Any], policy_path: Path) -> None:
    contract_version = metadata.get("contract_version")
    drawing = _drawing_module(contract_version)
    observation_dim, action_dim = _drawing_contract_dimensions(metadata, policy_path)
    expected = {
        "actuator": "xml",
        "recipe_options": {},
        "reward_weights": drawing.REWARD_WEIGHTS,
        "contract_version": contract_version,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
    }
    if contract_version == BRUSH_CONTRACT_VERSION:
        expected["max_episode_s"] = 120.0
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Studio drawing policy {policy_path} requires {key}={value!r}"
            )
    if isinstance(metadata.get("max_episode_s"), bool) or not isinstance(
        metadata.get("max_episode_s"), float
    ):
        raise ValueError(
            f"Studio drawing policy {policy_path} requires float max_episode_s"
        )


@lru_cache(maxsize=128)
def _cached_metadata_with_source(
    policy_path_text: str,
    cache_key: tuple[Any, ...],
) -> tuple[dict[str, Any], Path | None]:
    del cache_key
    return _load_metadata_with_source(Path(policy_path_text))


def _metadata_with_source(policy_path: Path) -> tuple[dict[str, Any], Path | None]:
    metadata, source_path = _cached_metadata_with_source(
        str(policy_path),
        _metadata_cache_key(policy_path),
    )
    return deepcopy(metadata), source_path


def studio_metadata(path: str | Path) -> dict[str, Any]:
    policy_path = Path(path).expanduser().resolve()
    metadata, _ = _metadata_with_source(policy_path)
    return metadata


def studio_policy_signature(path: str | Path) -> tuple[Any, ...]:
    policy_path = Path(path).expanduser().resolve()
    metadata, source_path = _metadata_with_source(policy_path)
    policy_stat = policy_path.stat()
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    metadata_hash = hashlib.sha256(encoded).hexdigest()
    if source_path is None:
        source_signature: tuple[Any, ...] = ("onnx",)
    else:
        source_stat = source_path.stat()
        source_signature = (
            str(source_path),
            source_stat.st_mtime_ns,
            source_stat.st_size,
        )
    return (
        str(policy_path),
        policy_stat.st_mtime_ns,
        policy_stat.st_size,
        *source_signature,
        metadata_hash,
    )


def discover_studio_policies() -> list[dict[str, Any]]:
    root = studio_runs_root()
    if not root.is_dir():
        return []
    policies = []
    for policy_path in root.glob("*/*/*.onnx"):
        try:
            metadata = studio_metadata(policy_path)
            if metadata["recipe"] != DRAWING_RECIPE:
                _load_recipe_factory()
            _recipe_kwargs(metadata)
            relative = policy_path.resolve().relative_to(root)
            recipe, run_name, artifact = relative.parts
            policies.append(
                {
                    "group": "studio",
                    "id": f"studio:{recipe}/{run_name}",
                    "label": run_name,
                    "path": str(policy_path.resolve()),
                    "mtime": policy_path.stat().st_mtime,
                    "artifact": artifact,
                    "recipe": metadata["recipe"],
                }
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            continue
    return sorted(policies, key=lambda item: (-item["mtime"], item["id"]))


def _add_sibling_rlx_to_path() -> None:
    checkout = Path(__file__).resolve().parents[3] / "rlx"
    if checkout.is_dir() and str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))


def _load_recipe_factory() -> Callable[..., Any]:
    try:
        module = importlib.import_module("rlx.environments.microduck_recipes")
    except ImportError:
        _add_sibling_rlx_to_path()
        try:
            module = importlib.import_module("rlx.environments.microduck_recipes")
        except ImportError as exc:
            raise RuntimeError(
                "Studio policy playback requires the sibling rlx checkout"
            ) from exc
    return module.make_single_recipe_env


def _validate_dance_options(options: dict[str, Any]) -> dict[str, Any]:
    allowed = {"dance_clip", "dance_clip_sha256", "dance_pose_sigma"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"unsupported dance recipe options: {sorted(unknown)}")
    clip_value = options.get("dance_clip")
    clip_hash = options.get("dance_clip_sha256")
    if clip_value is None and clip_hash is None:
        result = {}
    elif not isinstance(clip_value, str) or not isinstance(clip_hash, str):
        raise ValueError("dance metadata requires dance_clip and dance_clip_sha256")
    else:
        clip_path = Path(clip_value).expanduser().resolve()
        if not clip_path.is_file():
            raise FileNotFoundError(f"Studio dance clip does not exist: {clip_path}")
        actual_hash = hashlib.sha256(clip_path.read_bytes()).hexdigest()
        if actual_hash != clip_hash:
            raise ValueError(
                f"Studio dance clip hash mismatch for {clip_path}: "
                f"expected {clip_hash}, got {actual_hash}"
            )
        result = {"dance_clip": clip_path}
    if "dance_pose_sigma" in options:
        result["dance_pose_sigma"] = options["dance_pose_sigma"]
    return result


def _validate_backflip_protocol() -> dict[str, Any]:
    try:
        module = importlib.import_module("rlx.environments.backflip")
    except ImportError:
        _add_sibling_rlx_to_path()
        module = importlib.import_module("rlx.environments.backflip")
    current = module.protocol_metadata()
    return current


def _recipe_kwargs(metadata: dict[str, Any]) -> dict[str, Any]:
    recipe = metadata["recipe"]
    options = metadata["recipe_options"]
    if recipe == DRAWING_RECIPE:
        if options:
            raise ValueError(
                f"unsupported drawing recipe options: {sorted(options)}"
            )
        return {}
    if recipe in {"basketball", "bridge"}:
        if options:
            raise ValueError(f"unsupported {recipe} recipe options: {sorted(options)}")
        return {}
    if recipe == "dance":
        return _validate_dance_options(options)
    if recipe == "running":
        unknown = set(options) - {"locomotion_forward_command"}
        if unknown:
            raise ValueError(f"unsupported running recipe options: {sorted(unknown)}")
        return dict(options)
    if recipe == "stilts":
        required = {"stilt_height_cm", "stilt_blend", "stilt_mass_kg"}
        allowed = required | {"locomotion_forward_command"}
        missing = required - set(options)
        unknown = set(options) - allowed
        if missing or unknown:
            raise ValueError(
                f"invalid stilts recipe options; missing={sorted(missing)}, "
                f"unsupported={sorted(unknown)}"
            )
        return dict(options)
    if recipe == "swing":
        required = {
            "swing_initial_angle_deg",
            "swing_initial_rate_rad_s",
            "swing_planar_actions",
        }
        missing = required - set(options)
        unknown = set(options) - required
        if missing or unknown:
            raise ValueError(
                f"invalid swing recipe options; missing={sorted(missing)}, "
                f"unsupported={sorted(unknown)}"
            )
        return dict(options)
    current = _validate_backflip_protocol()
    for key in ("backflip_protocol_version", "stand_policy_sha256"):
        if options.get(key) != current.get(key):
            raise ValueError(
                f"Studio Backflip {key} does not match the current composite controller"
            )
    return {"backflip_mode": "showcase"}


def make_studio_env(policy_path: str | Path, seed: int):
    metadata = studio_metadata(policy_path)
    if metadata["recipe"] == DRAWING_RECIPE:
        contract_version = metadata["contract_version"]
        drawing = _drawing_module(contract_version)
        env_class = (
            drawing.BrushEnv
            if contract_version == BRUSH_CONTRACT_VERSION
            else drawing.DrawingEnv
        )
        env = env_class(
            seed=seed,
            max_episode_s=metadata["max_episode_s"],
            assistance=0,
        )
        env.studio_recipe = DRAWING_RECIPE
        return env
    if metadata["recipe"] == "basketball":
        _add_sibling_rlx_to_path()
        from rlx.environments.basketball import BasketballEnv
        env = BasketballEnv(seed=seed, actuator=metadata["actuator"], hold=0,
                            command=(0, 0, 0), random_yaw=False,
                            max_episode_s=max(60.0, metadata["max_episode_s"]))
        env.studio_recipe = "basketball"
        return env
    factory = _load_recipe_factory()
    env = factory(
        metadata["recipe"],
        seed=seed,
        actuator=metadata["actuator"],
        domain_rand=False,
        obs_noise=False,
        action_delay=False,
        random_yaw=False,
        max_episode_s=metadata["max_episode_s"],
        weight_overrides=metadata["reward_weights"],
        **_recipe_kwargs(metadata),
    ).unwrapped
    env.studio_recipe = metadata["recipe"]
    return env


__all__ = [
    "discover_studio_policies",
    "is_studio_policy_path",
    "make_studio_env",
    "studio_metadata",
    "studio_policy_signature",
    "studio_runs_root",
]
