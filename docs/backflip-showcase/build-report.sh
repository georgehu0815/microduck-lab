#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
"${PYTHON:-../../rlx/.venv-microduck/bin/python}" build-evidence-images.py
pandoc REPORT.md --from=gfm --to=html5 --standalone --toc --toc-depth=2 \
  --metadata title="Microduck Backflip Showcase: Design and Evidence" \
  --metadata author="George Hu" --css=print.css --output=REPORT.html
weasyprint REPORT.html REPORT.pdf
pdfinfo REPORT.pdf
