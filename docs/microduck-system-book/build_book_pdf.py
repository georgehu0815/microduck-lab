from __future__ import annotations

import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
SOURCE = DIRECTORY / "BOOK.zh-CN.md"
OUTPUT = DIRECTORY / "BOOK.zh-CN.pdf"
DIAGRAM_DIRECTORY = DIRECTORY / "diagrams"
MERMAID_CONFIG = DIAGRAM_DIRECTORY / "mermaid-config.json"


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"required tool not found: {name}")
    return path


def verify_vector_graphics(pdfimages: str) -> None:
    result = subprocess.run(
        [pdfimages, "-list", OUTPUT.name],
        cwd=DIRECTORY,
        check=True,
        capture_output=True,
        text=True,
    )
    image_rows = [
        line for line in result.stdout.splitlines() if line.lstrip()[:1].isdigit()
    ]
    if image_rows:
        raise SystemExit("PDF contains raster images; diagrams must remain vector graphics")


def build_diagrams(mmdc: str) -> None:
    for source in sorted(DIAGRAM_DIRECTORY.glob("*.mmd")):
        output = source.with_suffix(".svg")
        subprocess.run(
            [
                mmdc,
                "-i",
                source.name,
                "-o",
                output.name,
                "-c",
                MERMAID_CONFIG.name,
                "-b",
                "white",
            ],
            cwd=DIAGRAM_DIRECTORY,
            check=True,
        )
        root = ET.parse(output).getroot()
        text_nodes = root.findall(".//{http://www.w3.org/2000/svg}text")
        html_labels = root.findall(".//{http://www.w3.org/2000/svg}foreignObject")
        if not text_nodes or html_labels:
            raise SystemExit(f"{output.name}: labels must use native SVG text")


def main() -> None:
    pandoc = require_tool("pandoc")
    mmdc = require_tool("mmdc")
    require_tool("xelatex")
    require_tool("rsvg-convert")
    pdfimages = require_tool("pdfimages")
    build_diagrams(mmdc)

    header = """\
\\usepackage{longtable}
\\usepackage{booktabs}
\\usepackage{fvextra}
\\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\\\\{\\}}
\\setlength{\\tabcolsep}{3pt}
\\setlength{\\LTleft}{0pt}
\\setlength{\\LTright}{0pt}
\\widowpenalty=10000
\\clubpenalty=10000
"""
    with tempfile.TemporaryDirectory(prefix="microduck-system-book-") as temporary:
        header_path = Path(temporary) / "header.tex"
        header_path.write_text(header, encoding="utf-8")
        command = [
            pandoc,
            SOURCE.name,
            "--output",
            OUTPUT.name,
            "--from=gfm+alerts",
            "--shift-heading-level-by=-1",
            "--pdf-engine=xelatex",
            f"--resource-path={DIRECTORY}",
            f"--include-in-header={header_path}",
            "--toc",
            "--toc-depth=3",
            "--number-sections",
            "--syntax-highlighting=tango",
            "--variable=documentclass=report",
            "--variable=classoption=openany",
            "--variable=papersize=a4",
            "--variable=geometry:margin=17mm",
            "--variable=fontsize=10pt",
            "--variable=mainfont=Arial Unicode MS",
            "--variable=sansfont=Arial Unicode MS",
            "--variable=monofont=Arial Unicode MS",
            "--variable=colorlinks=true",
            "--variable=linkcolor=blue",
            "--variable=urlcolor=blue",
            "--metadata=author:Microduck Lab",
            "--metadata=date:2026-09-12",
            "--metadata=lang:zh-CN",
            "--metadata=toc-title:目录",
        ]
        subprocess.run(command, cwd=DIRECTORY, check=True)
    verify_vector_graphics(pdfimages)
    print(f"generated {OUTPUT}")
    print("verified: PDF contains no raster images")


if __name__ == "__main__":
    main()
