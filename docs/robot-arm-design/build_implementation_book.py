from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import urlparse

import markdown


DIRECTORY = Path(__file__).resolve().parent
CHAPTERS = ("repair-v3/RESULTS.zh-CN.md", "COMPONENT-CONSISTENCY.zh-CN.md", "IMPLEMENTATION.md", "RESULTS.md", "HARDWARE.md", "POWER-AND-CAD-RESEARCH.md", "VERIFICATION.md")
STYLE = """
body{font-family:Arial,'PingFang SC','Microsoft YaHei',sans-serif;color:#16333c;line-height:1.65;max-width:1100px;margin:40px auto;padding:0 24px;background:#fff}
h1{font-size:30px;border-bottom:4px solid #147d78;padding-bottom:12px}h2{font-size:23px;margin-top:34px;color:#125f60}h3{font-size:19px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{border:1px solid #c8d8dc;text-align:left;padding:8px;overflow-wrap:anywhere}th{background:#e7f2f1}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef4f5;padding:14px;font-size:12px}code{font-family:Menlo,monospace;font-size:.9em}img{max-width:100%;height:auto}a{color:#126b78;overflow-wrap:anywhere}
blockquote{border-left:4px solid #c38d33;padding:10px 16px;background:#fff5e5;margin:20px 0}.chapter{break-before:page}.chapter:first-of-type{break-before:auto}
@media print{body{font-size:11px;margin:0;padding:0;max-width:none}h1{font-size:23px}h2{font-size:17px}h3{font-size:14px}table{font-size:9px}td,th{padding:5px}pre{font-size:9px}img,table,pre,blockquote{break-inside:avoid}h1,h2,h3{break-after:avoid}a{text-decoration:none}}
"""


def main():
    chapters = []
    for name in CHAPTERS:
        text = (DIRECTORY / name).read_text()
        rendered = markdown.markdown(text, extensions=["tables", "fenced_code"])
        parent = Path(name).parent
        if parent != Path("."):
            def rebase_link(match):
                attribute, target = match.groups()
                if urlparse(target).scheme or target.startswith(("/", "#")):
                    return match.group(0)
                return f'{attribute}="{parent.as_posix()}/{target}"'
            rendered = re.sub(r'(href|src)="([^"]+)"', rebase_link, rendered)
        chapters.append(f'<section class="chapter">{rendered}</section>')
        if name == "COMPONENT-CONSISTENCY.zh-CN.md":
            standalone = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>部件一致性核对</title><style>' + STYLE + '</style></head><body>' + rendered + '</body></html>'
            (DIRECTORY / "COMPONENT-CONSISTENCY.zh-CN.html").write_text(standalone)
    html = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Microduck 机械臂课堂实验手册</title><style>' + STYLE + '</style></head><body>' + "\n".join(chapters) + '</body></html>'
    (DIRECTORY / "CLASSROOM.zh-CN.html").write_text(html)


if __name__ == "__main__":
    main()
