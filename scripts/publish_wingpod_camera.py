"""Publish verified WingPod artifacts into local static viewer assets."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
APPEARANCE = ROOT / "artifacts/microduck-arm-v1c/appearance"
SOURCE = APPEARANCE / "wingpod-camera-v2"
BOM = ROOT / "docs/microduck-arm-v1c/appearance/camera-v2-bom"
OUTPUT = ROOT / "duck-viewer/public/static/wingpod"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(include_cases=True, soft=False):
    primary = APPEARANCE / "wingpod-camera-v2-soft" if soft else SOURCE
    files = {name: primary / name for name in ("hero.png", "camera-eyes-detail.png", "wingpod-camera-eyes.mp4", "video-verification.json")}
    files.update({name: BOM / name for name in ("BOM.csv", "BOM.json", "PARTS.md", "README.md")})
    files["CAMERA-V2.md"] = ROOT / "docs/microduck-arm-v1c/appearance/CAMERA-V2.md"
    if soft:
        files["CAMERA-V2-SOFT.md"] = ROOT / "docs/microduck-arm-v1c/appearance/CAMERA-V2-SOFT.md"
        files["camera-spec.json"] = primary / "camera-spec.json"
        files["appearance-validation.json"] = primary / "appearance-validation.json"
        appearance = json.loads(files["video-verification.json"].read_text())
        if digest(files["wingpod-camera-eyes.mp4"]) != appearance["sha256"]:
            raise ValueError("Soft appearance video changed")
        for path, expected in appearance["appearance_sources"].items():
            if digest(ROOT / path) != expected:
                raise ValueError("Soft appearance source changed")
    nominal = primary / "tennis-return-nominal-0"
    files["tennis-return.mp4"] = nominal / "rollout.mp4"
    files["sequence-contact-sheet.jpg"] = nominal / "sequence-contact-sheet.jpg"
    if soft:
        for name in ("hero.png", "camera-eyes-detail.png", "sequence-contact-sheet.jpg"):
            image = Path(name)
            files[f"{image.stem}-soft-{digest(files[name])[:12]}{image.suffix}"] = files[name]
    files["camera-task-verification.json"] = nominal / "camera-task-verification.json"
    task = json.loads(files["camera-task-verification.json"].read_text())
    if not task["success"] or not task["replay"]["zero_error"] or digest(files["tennis-return.mp4"]) != task["video"]["sha256"]:
        raise ValueError("Nominal task video is not source-verified")
    for path, expected in task["appearance_and_renderer_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError("Nominal task appearance source changed")
    if include_cases:
        files["cases.json"] = SOURCE / "task-cases/cases.json"
        catalog = json.loads(files["cases.json"].read_text())
        if catalog["summary"] != {"positivePassed": 30, "positiveTotal": 30, "controlsMatched": 2, "controlsTotal": 2}:
            raise ValueError("Case matrix is not complete")
        if len(catalog["cases"]) != 32 or len({case["id"] for case in catalog["cases"]}) != 32:
            raise ValueError("Missing or duplicate cases")
        for case in catalog["cases"]:
            name = case["id"]
            if Path(name).name != name or not case["expectedOutcomeMatched"] or not case["zeroError"]:
                raise ValueError("Invalid case")
            directory = SOURCE / "task-cases" / name
            receipt = json.loads((directory / "camera-receipt.json").read_text())
            if receipt["case"] != case or digest(directory / "rollout.mp4") != receipt["replay"]["video_sha256"]:
                raise ValueError(f"Case media or receipt changed: {name}")
            for path, expected in receipt["appearance_sources"].items():
                if digest(ROOT / path) != expected:
                    raise ValueError(f"Case appearance sources changed: {name}")
            files[case["video"]] = directory / "rollout.mp4"
            files[case["receipt"]] = directory / "camera-receipt.json"
    if not all(path.is_file() for path in files.values()):
        raise ValueError("Required publication assets are missing")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for filename, source in files.items():
        destination = OUTPUT / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        checksum = digest(source)
        if digest(destination) != checksum:
            raise ValueError("Published asset hash mismatch")
        manifest[filename] = {"sha256": checksum, "bytes": destination.stat().st_size,
                              "source": str(source.relative_to(ROOT))}
    (OUTPUT / "manifest.json").write_text(json.dumps({"design": "WingPod Camera v2 Soft" if soft else "WingPod Camera v2", "all_cases_included": include_cases,
        "primary_palette": "cream-sage-peach" if soft else "graphite",
        "case_library_palette": "original graphite v2",
        "files": manifest, "hardware_release": False, "vision_control": False}, indent=2) + "\n")
    standalone = ROOT / "duck-viewer/public/wingpod-v2/index.html"
    artifact_site = APPEARANCE / "wingpod-v2"
    artifact_site.mkdir(exist_ok=True)
    html = standalone.read_text().replace("new URL('../static/wingpod/', location.href)",
        "new URL('../../../../duck-viewer/public/static/wingpod/', location.href)")
    html = html.replace('href="../"', 'href="../../../../duck-viewer/public/wingpod-v2/index.html"')
    html = html.replace('href="../wingpod/"', 'href="../../../../duck-viewer/public/wingpod-v2/index.html"')
    local_data = {"BOM.json": json.loads((BOM / "BOM.json").read_text())}
    if include_cases:
        local_data["cases.json"] = catalog
    embedded = json.dumps(local_data, ensure_ascii=False).replace("</", "<\\/")
    html = html.replace("<!-- WINGPOD_LOCAL_DATA -->",
                        '<script type="application/json" id="wingpod-local-data">' + embedded + '</script>')
    (artifact_site / "index.html").write_text(html)
    print(json.dumps({"published_files": len(files), "all_cases": include_cases, "static_page": str(standalone), "artifact_page": str(artifact_site / "index.html")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--soft", action="store_true")
    args = parser.parse_args()
    publish(not args.base_only, args.soft)
