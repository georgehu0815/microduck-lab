"""Validate both editions, source and image provenance, and PDF text/layout."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from PIL import Image


BOOK = Path(__file__).resolve().parents[1]
ROOT = BOOK.parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_markdown(path):
    text = path.read_text()
    counts = {"python": 0, "bash": 0, "json": 0, "images": 0}
    failures = []
    for index, (language, source) in enumerate(re.findall(
        r"```(python|bash|json)\n(.*?)\n```", text, re.S
    ), 1):
        try:
            if language == "python":
                ast.parse(source)
            elif language == "json":
                json.loads(source)
            else:
                subprocess.run(["bash", "-n"], input=source, text=True,
                               capture_output=True, check=True)
            counts[language] += 1
        except (SyntaxError, ValueError, subprocess.CalledProcessError) as error:
            failures.append(f"{path.name} fence {index} ({language}): {error}")
    for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        image = BOOK / target
        if not image.is_file():
            failures.append(f"Missing image: {target}")
            continue
        with Image.open(image) as opened:
            opened.verify()
        counts["images"] += 1
    return {"file": path.name, "counts": counts, "characters": len(text), "failures": failures}


def validate_pdf(path, language):
    info = subprocess.check_output(["pdfinfo", str(path)], text=True)
    if not re.search(r"Author:\s+George Hu\s*\n", info):
        raise AssertionError("PDF author metadata does not match George Hu")
    pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    if pages < 60:
        raise AssertionError(f"Unexpectedly short PDF: {pages}")
    text = subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)
    if "Author: George Hu" not in text.split("\f")[0]:
        raise AssertionError("Cover is missing the requested author credit")
    if language == "zh" and not all(word in text for word in ("篮球", "独木桥", "目录")):
        raise AssertionError("Chinese PDF text extraction failed")
    if "PPO" not in text or "102,400" not in text:
        raise AssertionError("Core textbook text absent")
    log = (BOOK / "build" / language / "pass-3.log").read_text()
    warnings = [line for line in log.splitlines() if any(token in line for token in (
        "Overfull", "Missing character", "Float too large", "! LaTeX Error",
    ))]
    bbox = subprocess.check_output(["pdftotext", "-bbox", str(path), "-"], text=True)
    document = ET.fromstring(bbox)
    namespace = {"x": "http://www.w3.org/1999/xhtml"}
    outside = []
    for number, page in enumerate(document.findall(".//x:page", namespace), 1):
        width, height = float(page.attrib["width"]), float(page.attrib["height"])
        for word in page.findall(".//x:word", namespace):
            if (float(word.attrib["xMin"]) < 0 or float(word.attrib["yMin"]) < 0
                    or float(word.attrib["xMax"]) > width or float(word.attrib["yMax"]) > height):
                outside.append({"page": number, "text": word.text})
    return {"file": path.name, "pages": pages, "bytes": path.stat().st_size,
            "sha256": sha256(path), "extracted_characters": len(text),
            "layout_warnings": warnings, "words_outside_page": outside}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pdf", action="store_true")
    args = parser.parse_args()
    source_manifest = json.loads((BOOK / "source-manifest.json").read_text())
    for item in source_manifest:
        assert sha256(BOOK / "source" / item["path"]) == item["sha256"], item["path"]
        assert sha256(ROOT / item["path"]) == item["sha256"], "Live source changed: " + item["path"]
    image_hashes = 0
    for line in (BOOK / "assets/SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        choices = (BOOK / "assets" / name, ROOT / name, BOOK / name)
        path = next((candidate for candidate in choices if candidate.is_file()), None)
        assert path is not None, name
        assert sha256(path) == expected, name
        image_hashes += 1
    chapters = []
    pdfs = []
    for language in ("zh", "en"):
        chapters.append(validate_markdown(BOOK / f"microduck-basketball-bridge-{language}.md"))
        if not args.skip_pdf:
            pdfs.append(validate_pdf(BOOK / f"microduck-basketball-bridge-{language}.pdf", language))
    arithmetic = subprocess.check_output([sys.executable, str(BOOK / "examples/ppo_arithmetic.py")], text=True)
    ast_files = list((BOOK / "tools").glob("*.py")) + list((BOOK / "examples").glob("*.py"))
    for path in ast_files:
        ast.parse(path.read_text())
    passed = not any(chapter["failures"] for chapter in chapters)
    passed = passed and not any(item["layout_warnings"] or item["words_outside_page"] for item in pdfs)
    report = {"scope": "book syntax/provenance/PDF and executable arithmetic, not retraining",
              "passed": passed, "source_hashes": len(source_manifest), "asset_hashes": image_hashes,
              "tool_example_syntax": len(ast_files), "markdown": chapters, "pdf": pdfs,
              "arithmetic": json.loads(arithmetic)}
    output = BOOK / "verification/book-validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
