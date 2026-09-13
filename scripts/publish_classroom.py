"""Publish existing bilingual classroom decks and PDF-derived browser slides."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile
from xml.etree import ElementTree

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/classroom-ppt"
OUTPUT = ROOT / "duck-viewer/public/classroom-assets"
NAMESPACE = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slide_text(deck):
    with zipfile.ZipFile(deck) as archive:
        filenames = sorted((name for name in archive.namelist()
                            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)),
                           key=lambda name: int(re.search(r"slide(\d+)\.xml", name)[1]))
        result = []
        for filename in filenames:
            root = ElementTree.fromstring(archive.read(filename))
            paragraphs = ["".join(paragraph.itertext()).strip() for paragraph in root.findall(".//a:p", NAMESPACE)]
            paragraphs = [paragraph for paragraph in paragraphs if paragraph]
            result.append({"title": paragraphs[0] if paragraphs else f"Slide {len(result) + 1}",
                           "text": "\n".join(paragraphs)})
        return result


def publish():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifests = {language: json.loads((directory / "deck-manifest.json").read_text())
                 for language, directory in (("en", SOURCE), ("zh", SOURCE / "zh-CN"))}
    if {entry["slug"] for entry in manifests["en"]} != {entry["slug"] for entry in manifests["zh"]}:
        raise ValueError("English and Chinese lesson inventories differ")
    files = {}
    lessons = []

    def register(path, source):
        name = path.relative_to(OUTPUT).as_posix()
        files[name] = {"sha256": digest(path), "bytes": path.stat().st_size,
                       "source": source.relative_to(ROOT).as_posix()}
        return name

    discovered = set()
    for order, english in enumerate(manifests["en"], 1):
        identity = english["slug"]
        if not re.fullmatch(r"[a-z0-9-]+", identity):
            raise ValueError("Unsafe lesson slug")
        chinese = next(entry for entry in manifests["zh"] if entry["slug"] == identity)
        lesson = {"id": identity, "order": order,
                  "title": {"en": english["title"], "zh": chinese["title"]}, "editions": {}}
        videos = []
        for source_video in sorted((SOURCE / "videos").glob(f"{identity}-*.mp4")):
            video = OUTPUT / "videos" / f"{source_video.stem}-{digest(source_video)[:12]}.mp4"
            video.parent.mkdir(exist_ok=True)
            shutil.copy2(source_video, video)
            videos.append({"file": register(video, source_video), "label": source_video.stem})
        for language, entry, directory in (("en", english, SOURCE), ("zh", chinese, SOURCE / "zh-CN")):
            deck = (directory / entry["file"]).resolve()
            if not deck.is_relative_to(SOURCE.resolve()):
                raise ValueError("Deck path escapes classroom sources")
            discovered.add(deck)
            pdf = directory / "pdf" / f"{deck.stem}.pdf"
            slides = slide_text(deck)
            info = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True).stdout
            pages = int(re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)[1])
            if not pages == len(slides) == entry["slides"]:
                raise ValueError(f"PPTX/PDF/manifest slide counts differ for {deck}")
            target = OUTPUT / identity / f"{language}-{digest(deck)[:12]}-{digest(pdf)[:12]}"
            target.mkdir(parents=True, exist_ok=True)
            image_paths = [target / f"slide-{number:03d}.jpg" for number in range(1, pages + 1)]
            if not all(path.is_file() for path in image_paths):
                subprocess.run(["pdftoppm", "-scale-to", "1200", "-jpeg", "-jpegopt", "quality=85",
                                str(pdf), str(target / "render")], check=True, capture_output=True)
                rendered = sorted(target.glob("render-*.jpg"), key=lambda path: int(path.stem.split("-")[-1]))
                if len(rendered) != pages:
                    raise ValueError("PDF renderer returned an incomplete slide set")
                for temporary, destination in zip(rendered, image_paths):
                    temporary.replace(destination)
            for slide, image in zip(slides, image_paths):
                with Image.open(image) as preview:
                    preview.verify()
                slide["image"] = register(image, pdf)
            resources = {}
            for original in (deck, pdf):
                destination = target / original.name
                shutil.copy2(original, destination)
                resources[original.suffix[1:]] = register(destination, original)
            lesson["editions"][language] = {**resources, "slides": slides, "videos": videos}
            print(f"{identity}/{language}: {pages} slides published", flush=True)
        lessons.append(lesson)
    if discovered != {path.resolve() for path in SOURCE.rglob("*.pptx")}:
        raise ValueError("Classroom inventory does not include every PPTX")
    catalog = {"schemaVersion": 1, "lessons": lessons}
    temporary = OUTPUT / "catalog.pending.json"
    temporary.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(OUTPUT / "catalog.json")
    register(OUTPUT / "catalog.json", SOURCE / "deck-manifest.json")
    report = {"files": files, "lessonCount": len(lessons), "deckCount": len(lessons) * 2,
              "slideCount": sum(len(edition["slides"]) for lesson in lessons for edition in lesson["editions"].values()),
              "previewSource": "Existing PDF exports; no live PowerPoint animations. Videos are separate.",
              "sourceDecksUnmodified": True}
    (OUTPUT / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    publish()
