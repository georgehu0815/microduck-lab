from pathlib import Path
import argparse

import imageio.v2 as imageio
import matplotlib
from PIL import Image, ImageDraw, ImageFont


def build_ui_panels(directory, font_path):
    source = directory / "ui/backflip-evaluation-render-desktop.png"
    if not source.exists():
        return
    with Image.open(source) as original:
        screenshot = original.convert("RGB")
    scale = screenshot.width / 428
    gutter = round(20 * scale)
    heading = round(48 * scale)
    overlap = round(24 * scale)
    segment_height = (screenshot.height + 2) // 3
    panel_height = segment_height + 2 * overlap
    sheet = Image.new(
        "RGB",
        (3 * screenshot.width + 4 * gutter, panel_height + heading + gutter),
        "#101c2a",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(font_path), round(20 * scale))
    for index, label in enumerate(("Top: verdict", "Middle: measurements", "Bottom: media evidence")):
        top = max(0, index * segment_height - overlap)
        bottom = min(screenshot.height, (index + 1) * segment_height + overlap)
        left = gutter + index * (screenshot.width + gutter)
        draw.text((left, round(12 * scale)), label, font=font, fill="#bce8cd")
        sheet.paste(screenshot.crop((0, top, screenshot.width, bottom)), (left, heading))
    sheet.save(directory / "ui-evaluation-panels.png", dpi=(300, 300))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="backflip-e2e-20260908-v5")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    source = root / "rlx/runs/studio/backflip" / args.run / "render/ep0.mp4"
    reader = imageio.get_reader(source)
    frame_rate = reader.get_meta_data()["fps"]
    moments = [(0., "Initial stance"), (0.68, "Assisted rotation"), (1., "Assisted rotation"),
               (1.32, "Policy landing"), (1.68, "Policy recovery"), (11.96, "Pretrained stand hold")]
    sheet = Image.new("RGB", (1920, 1000), "#101c2a")
    draw = ImageDraw.Draw(sheet)
    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    title = ImageFont.truetype(str(font_path), 38)
    label = ImageFont.truetype(str(font_path), 29)
    draw.text((24, 18), "Backflip showcase — actual simulated rollout", font=title, fill="white")
    try:
        for index, (moment, caption) in enumerate(moments):
            frame = Image.fromarray(reader.get_data(round(moment * frame_rate))).resize((640, 360))
            column, row = index % 3, index // 3
            position = (column * 640, 85 + row * 455)
            sheet.paste(frame, position)
            draw.text((position[0] + 16, position[1] + 375), f"{moment:.2f} s · {caption}", font=label, fill="#bce8cd")
    finally:
        reader.close()
    sheet.save(directory / "motion-evidence.png", dpi=(300, 300))
    build_ui_panels(directory, font_path)


if __name__ == "__main__":
    main()
