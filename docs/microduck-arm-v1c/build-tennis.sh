#!/bin/sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
for language in en zh-CN; do
  pandoc "$script_dir/TENNIS-RETURN.$language.md" \
    --from=markdown --pdf-engine=xelatex \
    --include-in-header="$script_dir/tennis-pdf-header.tex" \
    -V geometry:margin=18mm -V fontsize=10pt -V colorlinks=true \
    -V mainfont="Hiragino Sans GB" -V monofont="Hiragino Sans GB" \
    -o "$script_dir/TENNIS-RETURN.$language.pdf"
done
