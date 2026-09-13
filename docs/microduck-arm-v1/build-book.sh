#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
output_en="$script_dir/MicroDuck-Arm-v1-B-Engineering.en.pdf"
output_zh="$script_dir/MicroDuck-Arm-v1-B-Engineering.zh-CN.pdf"
result_en="$script_dir/RESULTS.en.md"
result_zh="$script_dir/RESULTS.zh-CN.md"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'ERROR: required command not found: %s\n' "$1" >&2
        exit 2
    fi
}

require_file() {
    if [ ! -f "$1" ]; then
        printf 'ERROR: required leader-owned results file is missing: %s\n' "$1" >&2
        exit 3
    fi
}

require_command pandoc
require_command xelatex
require_command rsvg-convert
require_file "$result_en"
require_file "$result_zh"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/microduck-arm-book.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT HUP INT TERM

rsvg-convert -f pdf -o "$tmp_dir/engineering-flow.pdf" "$script_dir/engineering-flow.svg"
printf '%s\n' \
    '\XeTeXlinebreaklocale "zh"' \
    '\XeTeXlinebreakskip = 0pt plus 1pt' \
    '\emergencystretch=3em' \
    '\sloppy' >"$tmp_dir/book-preamble.tex"

build_book() {
    language=$1
    result_file=$2
    output_file=$3
    title=$4
    source_file="$tmp_dir/book-$language.md"

    {
        printf '%s\n' '---'
        printf 'title: "%s"\n' "$title"
        printf '%s\n\n' '---'
        printf '%s\n\n' '![MicroDuck arm v1-B engineering flow](engineering-flow.pdf)'
        for input_file in \
            "$script_dir/README.md" \
            "$script_dir/BOM.md" \
            "$script_dir/SOP.md" \
            "$script_dir/HARDWARE-ACCEPTANCE.md" \
            "$result_file"
        do
            printf '\n\n\\newpage\n\n'
            cat "$input_file"
        done
    } >"$source_file"

    pandoc "$source_file" \
        --from=markdown \
        --pdf-engine=xelatex \
        --resource-path="$tmp_dir:$script_dir:$repo_root" \
        --include-in-header="$tmp_dir/book-preamble.tex" \
        -V geometry:margin=20mm \
        -V colorlinks=true \
        -V mainfont="Hiragino Sans GB" \
        -V monofont="Hiragino Sans GB" \
        -o "$output_file"
}

build_book en "$result_en" "$output_en" "MicroDuck Arm v1-B Engineering Book"
build_book zh-CN "$result_zh" "$output_zh" "MicroDuck 单臂 v1-B 工程文档"

printf 'Generated:\n%s\n%s\n' "$output_en" "$output_zh"
