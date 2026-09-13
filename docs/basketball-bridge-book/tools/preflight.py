"""Inspect the book's local reproduction prerequisites without importing MLX."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[3]
REQUIRED = (
    "microduck_local/src/microduck_local/contract.py",
    "rlx/rlx/mjlab_microduck/robot/microduck/robot_allcollisions.xml",
    "microduck/policies/alpha_walking.onnx",
    "microduck-playground/artifacts/basketball/checkpoint.pt",
    "microduck-playground/artifacts/basketball/policy.onnx",
    "microduck-playground/src/mjlab_microduck/robot/assets/basketball/basketball.obj",
    "microduck-playground/src/mjlab_microduck/robot/assets/basketball/basketball.png",
)
PACKAGES = (
    "numpy", "torch", "mlx", "mujoco", "onnx", "onnxruntime", "Pillow",
    "matplotlib", "gymnasium", "imageio", "imageio-ffmpeg", "pytest",
    "stable-baselines3", "microduck-local", "rlx",
)


def inventory():
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    files = []
    for name in REQUIRED:
        path = ROOT / name
        files.append({
            "path": name,
            "exists": path.is_file(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        })
    return {
        "scope": "local assets and metadata only; not a training or graphics test",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "commands": {name: shutil.which(name) for name in (
            "uv", "node", "npm", "pandoc", "xelatex", "pdfinfo", "pdftotext", "pdftoppm",
        )},
        "files": files,
        "passed": all(item["exists"] for item in files) and all(packages.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inventory()
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
