from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import zipfile

from .config import ROOT, sha256
from .validation import provenance, write_json


def package_engineering():
    artifact_root = ROOT / "artifacts/microduck-arm-v1"
    evidence = json.loads((artifact_root / "verified-evidence/all-videos-index.json").read_text())
    if evidence.get("provenance") != provenance() or evidence.get("integrity_passed") is not True or evidence.get("count") != 12:
        raise ValueError("Current twelve-video evidence is required before packaging")
    for video in evidence["videos"]:
        if sha256(ROOT / video["path"]) != video["sha256"]:
            raise ValueError("Audited video changed before packaging")
    for language in ("en", "zh-CN"):
        if not (ROOT / f"docs/microduck-arm-v1/MicroDuck-Arm-v1-B-Engineering.{language}.pdf").is_file():
            raise ValueError("Both engineering PDFs are required")
    dependencies = ("numpy", "mujoco", "gymnasium", "torch", "stable-baselines3", "onnx", "onnxruntime", "imageio", "imageio-ffmpeg", "pillow", "pytest", "matplotlib")
    requirements = "\n".join(f"{name}=={importlib.metadata.version(name)}" for name in dependencies) + "\n"
    directories = ["microduck_arm_v1", "hardware/microduck-arm-v1", "docs/microduck-arm-v1", "artifacts/microduck-arm-v1", "rlx/rlx/mjlab_microduck/robot/microduck/assets"]
    excluded = {"__pycache__", "attempts", "evaluation-attempt-1", "evaluation-attempt-2", "negative-controls"}
    files = set()
    for directory in directories:
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and not (set(path.parts) & excluded) and path.suffix not in (".zip", ".pyc") and path.name not in ("package-manifest.json", "package-receipt.json"):
                files.add(path)
    files.add(artifact_root / "learning-smoke/ppo/ppo.zip")
    files.update((ROOT / "rlx/tests").glob("test_microduck_arm_v1_*.py"))
    files.update(ROOT / name for name in ("rlx/LICENSE", "rlx/rlx/mjlab_microduck/LICENSE", "rlx/rlx/mjlab_microduck/robot/microduck/robot_allcollisions_backlash.xml"))
    manifest = {"scope": "engineering_validation_prototype_not_fabrication_release", "provenance": provenance(),
                "files": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(files)}, "hardware_release": False,
                "runtime_note": "Python3.12; dependencies not bundled. Recreate environment. Existing learning checkpoints enforce original absolute source paths; see learning-smoke/REPLAY.md before relocation."}
    write_json(artifact_root / "package-manifest.json", manifest)
    output = artifact_root / "MicroDuck-Arm-v1-B-engineering-prototype.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(ROOT)))
        archive.write(artifact_root / "package-manifest.json", "package-manifest.json")
        archive.writestr("requirements-simulation.txt", requirements)
        archive.writestr("START-HERE.txt", "MicroDuck Arm v1-B engineering-validation prototype / 工程验证原型\n\nStart: docs/microduck-arm-v1/README.md\nResults: docs/microduck-arm-v1/RESULTS.en.md and RESULTS.zh-CN.md\nVideos: artifacts/microduck-arm-v1/verified-evidence/videos.html\n\n137 software tests pass; integrated free-base cases 0/12; fixture cases 2/8.\nNOT FABRICATION READY. NO HARDWARE RELEASE. Do not connect or actuate hardware from this archive.\n仅工程验证资料，不可直接加工或上机。实物验证未完成。\n\nSource/mesh repository layout is preserved. Original licenses are included.\nPython dependencies, system ffmpeg, PDF tooling, and robot hardware are NOT bundled.\nLearning checkpoint provenance is bound to original absolute source paths; see learning-smoke/REPLAY.md.\n")
    with zipfile.ZipFile(output) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"Corrupt archive entry: {corrupt}")
    receipt = {"path": str(output.relative_to(ROOT)), "sha256": sha256(output), "bytes": output.stat().st_size, "source_files": len(files), "zip_crc_passed": True, "hardware_release": False}
    write_json(artifact_root / "package-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(package_engineering(), indent=2))
