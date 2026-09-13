from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parents[2]
DIRECTORIES = (
    "docs/robot-arm-design", "hardware/md-arm-t1", "arm_control",
    "rlx/assets/md_arm_t1", "rlx/runs/arm/validated-20260912-v2", "rlx/runs/arm/preflight-v2",
    "rlx/runs/arm/recheck-20260912-v2", "rlx/runs/arm/strict-repair-20260912-v3",
    "rlx/runs/arm/strict-heldout-20260912-v3",
    "duck-viewer/app/arm",
)
SOURCES = (
    "rlx/rlx/mjlab_microduck/robot/microduck/robot_allcollisions_backlash.xml",
    "rlx/rlx/mjlab_microduck/robot/microduck/assets/xl330.stl",
    "rlx/rlx/mjlab_microduck/robot/microduck/assets/xl330.part",
    "rlx/rlx/environments/arm.py", "rlx/examples/ppo_microduck_arm.py", "rlx/examples/render_microduck_arm.py",
    "scripts/arm_lab.py", "scripts/run_arm_experiments.py", "scripts/summarize_arm_experiments.py",
    "scripts/recheck_arm_experiments.py", "scripts/build_arm_video_evidence.py",
    "scripts/train_strict_arm_repair.py", "scripts/replay_arm_repair_regressions.py",
    "scripts/summarize_strict_arm_repair.py", "scripts/verify_arm_video_evidence.cjs",
    "scripts/verify_shared_arm_components.cjs",
)


def main():
    output = ROOT / "docs/md-arm-t1-classroom.zh-CN.zip"
    files = set(ROOT / name for name in SOURCES)
    for directory in DIRECTORIES:
        files.update(path for path in (ROOT / directory).rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("START-HERE.zh-CN.txt", "先打开 docs/robot-arm-design/CLASSROOM.zh-CN.html 或 CLASSROOM.zh-CN.pdf。\n视频、报告与CAD保留项目相对目录；PDF链接可能绑定生成机器路径，请优先从HTML或文件夹打开MP4。\n此包为课堂资料与源码快照，不是独立可运行环境；复现需要原Microduck工作区及已安装依赖。\n硬件未制造或放行；禁止据此给实体电机通电。\n")
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT))
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Classroom archive CRC validation failed")
    print(output)


if __name__ == "__main__":
    main()
