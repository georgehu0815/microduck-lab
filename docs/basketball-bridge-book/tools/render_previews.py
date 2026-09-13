"""Render representative pages for human inspection; not a substitute for full-page bounds checks."""

import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw


BOOK = Path(__file__).resolve().parents[1]
TARGETS = {
    "en": ("clipped policy objective", "Exact local reward", "suspended bridge", "Read real reward", "class BasketballEnv"),
    "zh": ("裁剪后的策略目标", "精确的本地奖励", "绳索", "阅读真实 reward", "class BasketballEnv"),
}


def main():
    receipts = []
    for language, needles in TARGETS.items():
        source = BOOK / f"microduck-basketball-bridge-{language}.pdf"
        text = subprocess.check_output(["pdftotext", "-layout", str(source), "-"], text=True)
        pages = text.split("\f")
        selected = {1}
        for needle in needles:
            matches = [index + 1 for index, page in enumerate(pages)
                       if needle.casefold() in page.casefold() and index > 3]
            if matches:
                selected.add(matches[0])
        output = BOOK / "verification/previews" / language
        output.mkdir(parents=True, exist_ok=True)
        rendered = []
        for number in sorted(selected):
            prefix = output / f"page-{number:03d}"
            subprocess.run(["pdftoppm", "-f", str(number), "-singlefile", "-scale-to", "1400",
                            "-png", str(source), str(prefix)], check=True, capture_output=True)
            rendered.append(prefix.with_suffix(".png"))
        cell_width, cell_height = 620, 830
        sheet = Image.new("RGB", (cell_width * 3, cell_height * ((len(rendered) + 2) // 3)), "#e8edf1")
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(rendered):
            with Image.open(path) as opened:
                opened.thumbnail((600, 785))
                left = (index % 3) * cell_width + 10
                top = (index // 3) * cell_height + 30
                sheet.paste(opened, (left, top))
                draw.text((left, top - 22), f"{language.upper()} {path.stem}", fill="#18212c")
        target = BOOK / "verification" / f"{language}-page-contact-sheet.png"
        sheet.save(target)
        receipts.append({"edition": language, "pdf_pages": sorted(selected),
                         "contact_sheet": str(target.relative_to(BOOK)),
                         "scope": "representative visual review; all pages checked separately for text bounds"})
    (BOOK / "verification/preview-pages.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
