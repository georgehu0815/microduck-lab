"""Publish an illustrated bilingual review and fully decode its evidence media."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
DEFAULT = Path("artifacts/microduck-arm-v1c/appearance/wingpod-v1")


def font(size):
    return ImageFont.truetype(FONT, size)


def make_page(title, subtitle):
    page = Image.new("RGB", (1600, 1200), "#F4F0E8")
    drawing = ImageDraw.Draw(page)
    drawing.text((60, 35), title, fill="#253337", font=font(44))
    drawing.text((62, 105), subtitle, fill="#616963", font=font(24))
    drawing.text((60, 1140), "MicroDuck WingPod  /  外观评审 · 非制造放行  /  2026-09-13", fill="#8A5027", font=font(22))
    return page, drawing


def picture(page, path, box):
    image = Image.open(path).convert("RGB")
    image.thumbnail((box[2], box[3]))
    page.paste(image, (box[0] + (box[2] - image.width) // 2, box[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    output = args.output
    summary = json.loads((output / "replay-summary.json").read_text())
    if not summary["all_complete"] or not summary["all_zero_error"]:
        raise ValueError("Do not publish incomplete or mismatched replay")
    expected = {f"{variant}-{seed}" for variant in ("nominal", "small", "large") for seed in range(10)} | {"hold", "open_jaw"}
    if set(summary["episodes"]) != expected or summary["positive_successes"] != 30:
        raise ValueError("Incomplete expected episode matrix")
    for name, validation in summary["episodes"].items():
        directory = output / "replays" / name
        if not validation["complete"] or not validation["zero_error"]:
            raise ValueError(f"Incomplete episode: {name}")
        result = json.loads((directory / "result.json").read_text())
        if result["success"] != (name not in {"hold", "open_jaw"}):
            raise ValueError(f"Unexpected episode outcome: {name}")
        for filename, key in (("result.json", "source_result_sha256"), ("telemetry.jsonl", "source_telemetry_sha256")):
            if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != validation[key]:
                raise ValueError(f"Evidence file changed: {name}/{filename}")
    root = Path(__file__).resolve().parents[1]
    for name, digest in summary["appearance_sources"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Appearance source changed: {name}")
    video_paths = sorted((output / "replays").glob("*/rollout.mp4"))
    expected_videos = {name for name, value in summary["episodes"].items() if value["claimed_video_pass"]}
    if {path.parent.name for path in video_paths} != expected_videos:
        raise ValueError("Video file set differs from verified replay manifest")
    if expected_videos != {"nominal-0", "small-5", "large-6"}:
        raise ValueError("Publication requires the three declared representative videos")
    media = []
    for path in [output / "turntable.mp4", *video_paths]:
        reader = imageio.get_reader(path)
        frames = 0
        try:
            metadata = reader.get_meta_data()
            for frame in reader:
                if frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError(f"Invalid video frame: {path}")
                frames += 1
            if not frames:
                raise ValueError(f"Empty video: {path}")
        finally:
            reader.close()
        if path.name == "rollout.mp4":
            validation = json.loads((path.parent / "replay-validation.json").read_text())
            if frames != validation["frames_written"]:
                raise ValueError(f"Video frame count mismatch: {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != validation["video_sha256"]:
                raise ValueError(f"Video binding mismatch: {path}")
        Image.fromarray(frame).save(path.parent / "last-frame.jpg")
        media.append({"path": str(path.relative_to(output)), "frames_decoded": frames,
                      "fps": metadata["fps"], "duration_s": frames / metadata["fps"],
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (output / "media-verification.json").write_text(json.dumps({"all_decoded": True, "media": media}, indent=2) + "\n")

    pages = []
    page, drawing = make_page("奶油小鸭 / WINGPOD", "圆润、协调、保留动作功能  ·  A softer wing. The same movement.")
    picture(page, output / "before-after.jpg", (40, 200, 1520, 850))
    pages.append(page)
    page, drawing = make_page("01  整机协调 / FORM LANGUAGE", "暖瓷白主壳 · 蜂蜜黄翼片和脚掌 · 石墨色关节环")
    picture(page, output / "hero.png", (40, 180, 920, 800))
    for index, text in enumerate([
        "肩鞍：让底座与臂根衔接", "分段：不跨接运动关节", "夹爪：接触垫不包覆", "配色：身体、腿、臂统一",
        "Saddle / segmented pods", "Exposed contact surfaces", "One coherent palette",
    ]):
        drawing.text((990, 270 + index * 80), text, fill="#253337", font=font(26))
    pages.append(page)
    page, drawing = make_page("02  多视角 / PRODUCT VIEWS", "保持原关节与机身位置；不为效果图移动头部或伪造运动能力")
    for index, name in enumerate(("front", "side", "rear")):
        picture(page, output / f"{name}.png", (40 + index * 510, 260, 500, 580))
        drawing.text((100 + index * 510, 850), name.upper(), fill="#253337", font=font(28))
    pages.append(page)
    page, drawing = make_page("03  尺寸与装配意图 / DIMENSION INTENT", "轴距与功能尺寸冻结；新增饰壳仍需真实碰撞、质量与热验证")
    nodes = [(130, 430, "肩 / SHOULDER"), (590, 430, "肘 / ELBOW"), (1010, 430, "腕 / WRIST")]
    drawing.line((130, 430, 1010, 430), fill="#EBA830", width=30)
    for horizontal, vertical, label in nodes:
        drawing.ellipse((horizontal - 30, vertical - 30, horizontal + 30, vertical + 30), fill="#253337")
        drawing.text((horizontal - 65, vertical + 70), label, fill="#253337", font=font(25))
    drawing.text((285, 330), "55 mm", fill="#253337", font=font(40))
    drawing.text((750, 330), "50 mm", fill="#253337", font=font(40))
    drawing.text((1150, 380), "TCP 55 mm", fill="#253337", font=font(30))
    drawing.text((1150, 450), "开口 75 mm", fill="#253337", font=font(30))
    for index, text in enumerate([
        "10 个腿部执行器 + 4 个臂姿态执行器 + 1 个夹爪执行器 = 15",
        "电机舱饰壳候选外包络：26 × 32 × 40 mm；圆角 6 mm",
        "CAD 为分件评审排版；非整机装配图、非已验证安装包络。",
        "轴、承力件、内夹持面不变；新增卡扣、通风、线槽需后续实测设计。",
    ]):
        drawing.text((80, 670 + index * 90), text, fill="#253337", font=font(28))
    pages.append(page)
    page, drawing = make_page("04  功能验证 / FUNCTIONAL EVIDENCE", "原始 v13 动作逐步复放；不是新的 PPO 训练或真机验证")
    texts = [
        f"正例 / Positive episodes: {summary['positive_successes']} / 30",
        "负对照 / Negative controls: 2, expected failures preserved",
        f"Control steps: {summary['steps']:,}  /  Physics substeps: {summary['physics_substeps']:,}",
        "零误差字段：球位置、记录的关节角、仿真时间。",
        "Zero error: ball position, recorded joint positions, simulation time.",
        "3 段完整任务 MP4 + 1 段外观旋转 MP4，均全帧解码验证。",
        "网球任务保留原低位支承释放；不能据此宣称半桶高自由落体。",
        "保持的是原模型的功能，不是带真实新增饰壳的功能验证。",
    ]
    for index, text in enumerate(texts):
        drawing.text((80, 210 + index * 105), text, fill="#253337", font=font(29))
    pages.append(page)
    page, drawing = make_page("05  实体放行路径 / HARDWARE GATES", "本包状态：外观原型 / Visual prototype — NOT released for fabrication")
    texts = [
        "① 实物测量 → 分型、紧固、拆装、线束与散热设计",
        "② CAD 质量 / 质心 / 惯量 → 真实碰撞包络与公差扫掠",
        "③ 扭矩 / 平衡 / 温升 / 电气故障与急停检验",
        "④ 原全部任务 + 负对照 → 台架与真机闭环复验",
        "未验证：新增饰壳质量、净空、夹点、耐久、加工与装配。",
        "没有用移除碰撞、降低成功门槛或重新标记失败来获得通过。",
    ]
    for index, text in enumerate(texts):
        drawing.text((80, 220 + index * 130), text, fill="#253337", font=font(29))
    pages.append(page)
    pages[0].save(output / "wingpod-review.pdf", save_all=True, append_images=pages[1:], resolution=150)

    videos = "\n".join(
        f'<article><h3>{html.escape(item["path"])}</h3><video controls preload="metadata" src="{html.escape(item["path"])}"></video><p>{item["frames_decoded"]} frames verified · {item["duration_s"]:.1f}s</p></article>'
        for item in media
    )
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MicroDuck WingPod · 奶油小鸭</title><style>
body{{margin:0;background:#f4f0e8;color:#253337;font:17px/1.6 system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:48px 24px}}h1{{font-size:48px;margin:0}}h2{{margin-top:48px}}.tag{{color:#96601a;letter-spacing:3px}}img,video{{width:100%;border-radius:18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:24px}}article{{background:white;border-radius:24px;padding:24px}}h3{{overflow-wrap:anywhere}}a{{color:#745018}}.note{{padding:24px;border-left:5px solid #e9ad43;background:#fff8e7}}
</style><main><div class="tag">MICRODUCK / INDUSTRIAL DESIGN STUDY</div><h1>WingPod · 奶油小鸭</h1>
<p>圆润分段、统一配色、保留原动作。A softer wing. The same movement.</p>
<p class="note">外观原型，不是实体放行。Render-only redesign; added physical shells remain unverified.</p>
<img src="before-after.jpg" alt="同一姿态下原工程模型与新版外观对比">
<p><a href="wingpod-review.pdf">中英双语设计图册 / Illustrated review PDF</a> · <a href="wingpod-review-shells.scad">评审 CAD / Review CAD</a> · <a href="shell-dimensions.csv">尺寸表 / Dimensions</a> · <a href="appearance-spec.json">Design specification</a></p>
<h2>三视角 / Product views</h2><div class="grid"><img src="front.png" alt="Front"><img src="side.png" alt="Side"><img src="rear.png" alt="Rear"></div>
<h2>视频 / Verified media</h2><p>30/30 原任务正例复放成功，2 个负对照保持预期失败；3 段完整动作视频。旋转视频仅供外观展示。No new PPO or hardware tests.</p><div class="grid">{videos}</div>
<h2>证据与边界 / Evidence and limits</h2><p>原模型质量、惯量、关节、碰撞和控制保持不变。逐步零误差字段仅指球位置、记录的关节角及仿真时间，不代表全部状态或接触力逐字段比较。Zero-error replay compares ball positions, recorded joint positions and time, not every state or force field. 真实饰壳需重新验证增重、碰撞、布线与散热。原任务释放为低位支承释放，并非自由落体投球。</p>
<p><a href="replay-summary.json">逐步复放 / Physics replay</a> · <a href="appearance-validation.json">Physics fingerprint</a> · <a href="media-verification.json">完整解码与哈希 / Media hashes</a></p></main></html>'''
    (output / "index.html").write_text(document)
    print(json.dumps({"pdf_pages": len(pages), "videos": len(media), "all_decoded": True}))


if __name__ == "__main__":
    main()
