"""Build both textbook editions and an exact supporting source snapshot."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


BOOK = Path(__file__).resolve().parents[1]
ROOT = BOOK.parents[1]
PRINTED = (
    "rlx/rlx/environments/basketball.py",
    "rlx/rlx/models/basketball.py",
    "rlx/examples/ppo_microduck_basketball.py",
    "rlx/rlx/environments/bridge.py",
    "rlx/rlx/environments/bridge_evaluation.py",
    "rlx/rlx/models/bridge_bootstrap.py",
    "rlx/rlx/algorithms/ppo.py",
)
SUPPORTING = (
    "rlx/examples/ppo_microduck_balance.py",
    "rlx/examples/ppo_microduck_studio.py",
    "rlx/scripts/eval_basketball_local.py",
    "rlx/rlx/environments/microduck_recipes.py",
    "rlx/rlx/environments/microduck.py",
    "rlx/rlx/models/microduck.py",
    "rlx/rlx/export/microduck_onnx.py",
    "rlx/rlx/buffers/rollout_buffer.py",
    "rlx/rlx/utils/utils.py",
    "rlx/rlx/utils/distributions.py",
    "rlx/rlx/utils/__init__.py",
    "microduck_local/src/microduck_local/contract.py",
    "microduck_local/src/microduck_local/walk_env.py",
    "microduck_local/src/microduck_local/studio_policies.py",
    "microduck_local/src/microduck_local/viz_server.py",
    "duck-viewer/lib/experiments.ts",
    "duck-viewer/lib/rlx-job.ts",
    "duck-viewer/lib/scene-layout.ts",
    "scripts/train-basketball.sh",
    "scripts/train-bridge.sh",
    "scripts/verify-balance-studio-api.mjs",
    "scripts/verify-seven-cases.mjs",
    "rlx/pyproject.toml",
    "microduck_local/pyproject.toml",
    "duck-viewer/package.json",
    "microduck-playground/artifacts/basketball/NOTICE",
    "microduck-playground/artifacts/basketball/LICENSE",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    manifest = []
    tests = sorted(ROOT.glob("rlx/tests/test_basketball*.py"))
    tests += sorted(ROOT.glob("rlx/tests/test_bridge*.py"))
    tests += [ROOT / "rlx/tests/test_balance_adapter.py"]
    paths = list(PRINTED + SUPPORTING) + [str(path.relative_to(ROOT)) for path in tests]
    for name in paths:
        source = ROOT / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = BOOK / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest.append({"path": name, "sha256": digest(source), "bytes": source.stat().st_size,
                         "printed_in_full": name in PRINTED})
    (BOOK / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(BOOK / "supporting-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(BOOK / "source-manifest.json", "source-manifest.json")
        for item in manifest:
            archive.write(BOOK / "source" / item["path"], "source/" + item["path"])
    return manifest


def source_appendix(language, manifest):
    if language == "zh":
        heading = "# 附录：完整核心源码与版本指纹\n\n"
        explanation = (
            "以下七个核心模块按本版快照完整收录；注释、导入与实现保持原样。"
            "代码源文件与 SHA-256 清单随书提供。其他适配器、完整 Studio CLI、"
            "篮球评估器、共享模型、viewer 接口和测试位于 `source/` 与 `supporting-source.zip`。"
            "快照不是独立软件包；运行需要完整工作区、机器人资产和所列依赖。"
            "复制代码请使用 Markdown 或 `.py` 文件，PDF 中的视觉换行不是额外语法。\n\n"
        )
    else:
        heading = "# Appendix: complete core source and version fingerprints\n\n"
        explanation = (
            "Seven core modules follow in full, preserving their original imports, comments and implementation. "
            "Exact files and SHA-256 manifest accompany the book. Other adapters, the complete Studio CLI, "
            "basketball evaluator, shared models, viewer interfaces and tests are in `source/` and "
            "`supporting-source.zip`. This snapshot is not a standalone package: execution requires the "
            "complete workspace, robot assets and recorded dependencies. Copy code from Markdown or `.py` "
            "files; visual line wrapping in the PDF is not extra source syntax.\n\n"
        )
    parts = [heading, explanation]
    for index, name in enumerate(PRINTED, 1):
        entry = next(item for item in manifest if item["path"] == name)
        parts.append(f"## A{index}. `{name}`\n\nSHA-256: `{entry['sha256']}`\n\n")
        parts.append("```python\n" + (BOOK / "source" / name).read_text().rstrip() + "\n```\n\n")
    parts.append("## Source inventory / 源码清单\n\n")
    parts.append("| File / 文件 | Bytes / 字节 | Printed / 全文 |\n|---|---:|---|\n")
    for entry in manifest:
        parts.append(f"| `{entry['path']}` | {entry['bytes']} | {entry['printed_in_full']} |\n")
    return "".join(parts)


def assemble(language, manifest):
    chinese = language == "zh"
    title = "Microduck 平衡实验室" if chinese else "The Microduck Balance Laboratory"
    subtitle = "篮球与悬挂独木桥：从建模到 PPO、验证与复现" if chinese else "Basketball and Suspended Bridge: Modeling, PPO, Evidence and Reproduction"
    metadata = f'''---
title: "{title}"
subtitle: "{subtitle}"
author: "George Hu"
date: "2026-09-11 · v1.0 · Evidence: 2026-09-10"
lang: {"zh-CN" if chinese else "en-US"}
documentclass: article
papersize: letter
fontsize: 11pt
geometry:
  - margin=0.78in
mainfont: PingFang SC
sansfont: PingFang SC
monofont: Menlo
colorlinks: true
toc-title: "{"目录与学习路线" if chinese else "Contents and learning route"}"
---

'''
    chapters = sorted((BOOK / "chapters" / language).glob("*.md"))
    if len(chapters) < 6:
        raise ValueError(f"Incomplete {language} chapter set: {len(chapters)}")
    source = BOOK / f"microduck-basketball-bridge-{language}.md"
    source.write_text(metadata + "\n\n".join(path.read_text().strip().replace("../../assets/", "assets/") for path in chapters)
                      + "\n\n" + source_appendix(language, manifest))
    return source


def pdf(source, language):
    build = BOOK / "build" / language
    build.mkdir(parents=True, exist_ok=True)
    tex = build / "book.tex"
    subprocess.run([
        "pandoc", source.name, "--from=markdown+tex_math_dollars+raw_tex",
        "--to=latex", "--standalone", "--toc", "--toc-depth=2", "--number-sections",
        "--syntax-highlighting=tango", "--lua-filter=print-filter.lua",
        "--include-in-header=print-header.tex", "--output", str(tex),
    ], cwd=BOOK, check=True)
    for index in range(3):
        process = subprocess.run([
            "xelatex", "-interaction=nonstopmode", "-halt-on-error",
            f"-output-directory={build}", str(tex),
        ], cwd=BOOK, capture_output=True, text=True)
        (build / f"pass-{index + 1}.log").write_text(process.stdout + process.stderr)
        if process.returncode:
            print(process.stdout[-6500:])
            raise RuntimeError(f"XeLaTeX failed: {language}, pass {index + 1}")
    target = source.with_suffix(".pdf")
    shutil.copyfile(build / "book.pdf", target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown-only", action="store_true")
    parser.add_argument("--language", choices=("en", "zh", "both"), default="both")
    args = parser.parse_args()
    manifest = snapshot()
    outputs = []
    for language in (("zh", "en") if args.language == "both" else (args.language,)):
        source = assemble(language, manifest)
        outputs.append(str(source.relative_to(BOOK)))
        if not args.markdown_only:
            outputs.append(str(pdf(source, language).relative_to(BOOK)))
    print(json.dumps({"outputs": outputs, "source_files": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
