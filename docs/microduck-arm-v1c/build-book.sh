#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
engineering_dir="$repo_root/hardware/microduck-arm-v1c/engineering"
output_en="$script_dir/MicroDuck-Arm-v1-C-Engineering.en.pdf"
output_zh="$script_dir/MicroDuck-Arm-v1-C-Engineering.zh-CN.pdf"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'ERROR: required command not found: %s\n' "$1" >&2
        exit 2
    fi
}

require_file() {
    if [ ! -f "$1" ]; then
        printf 'ERROR: required file not found: %s\n' "$1" >&2
        exit 3
    fi
}

require_command pandoc
require_command xelatex
require_command rsvg-convert
require_file "$script_dir/BOOK.en.md"
require_file "$script_dir/BOOK.zh-CN.md"
require_file "$engineering_dir/load-path.svg"
require_file "$engineering_dir/wiring-architecture.svg"
require_file "$engineering_dir/mechanical-dimension-overview.svg"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/microduck-arm-v1c-book.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT HUP INT TERM

for name in load-path wiring-architecture mechanical-dimension-overview
do
    rsvg-convert -f pdf \
        -o "$tmp_dir/$name.pdf" \
        "$engineering_dir/$name.svg"
done

printf '%s\n' \
    '\XeTeXlinebreaklocale "zh"' \
    '\XeTeXlinebreakskip = 0pt plus 1pt' \
    '\emergencystretch=3em' \
    '\sloppy' >"$tmp_dir/book-preamble.tex"

prepare_source() {
    input=$1
    output=$2
    sed \
        -e 's#../../hardware/microduck-arm-v1c/engineering/load-path.svg#load-path.pdf#g' \
        -e 's#../../hardware/microduck-arm-v1c/engineering/wiring-architecture.svg#wiring-architecture.pdf#g' \
        -e 's#../../hardware/microduck-arm-v1c/engineering/mechanical-dimension-overview.svg#mechanical-dimension-overview.pdf#g' \
        "$input" >"$output"
}

prepare_source "$script_dir/BOOK.en.md" "$tmp_dir/book.en.md"
prepare_source "$script_dir/BOOK.zh-CN.md" "$tmp_dir/book.zh-CN.md"
printf '\n\\newpage\n' >>"$tmp_dir/book.en.md"
cat "$script_dir/RESULTS.en.md" >>"$tmp_dir/book.en.md"
printf '\n\\newpage\n' >>"$tmp_dir/book.zh-CN.md"
cat "$script_dir/RESULTS.zh-CN.md" >>"$tmp_dir/book.zh-CN.md"

build_book() {
    source=$1
    output=$2
    mainfont=$3
    monofont=$4

    pandoc "$source" \
        --from=markdown \
        --pdf-engine=xelatex \
        --resource-path="$tmp_dir:$script_dir:$repo_root" \
        --include-in-header="$tmp_dir/book-preamble.tex" \
        -V geometry:margin=18mm \
        -V colorlinks=true \
        -V fontsize=10pt \
        -V mainfont="$mainfont" \
        -V sansfont="$mainfont" \
        -V monofont="$monofont" \
        -o "$output"
}

build_book "$tmp_dir/book.en.md" "$output_en" "Hiragino Sans GB" "Menlo"
build_book "$tmp_dir/book.zh-CN.md" "$output_zh" "Hiragino Sans GB" "Hiragino Sans GB"

printf 'Generated:\n%s\n%s\n' "$output_en" "$output_zh"
