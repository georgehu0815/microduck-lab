# Publication review receipt

## Scope

Read-only independent chapter/code review, executable arithmetic, construction probes,
targeted scenario tests, source and figure provenance, and PDF visual/layout checks.
This is a publication review, not a new full training run or hardware certification.

## Independent review

- No HIGH findings in the bilingual PPO, reward, command and evidence explanations.
- MEDIUM: workshop claimed saved contract results before the receipt existed.
  Resolved by retaining `contracts-junit.xml`: 119 tests, zero failures/errors/skips.
- MEDIUM: retained historical basketball acceptance report has older generator hashes
  than the current snapshot. Resolved by an explicit historical-source boundary in both
  evidence chapters. The original report is not relabeled; the current 10-second diagnostic
  is separately hash-bound and is not called a 60-second acceptance battery.

## Runtime evidence

The targeted test command is the seven-file pytest command in the workshop chapter,
with `--junitxml=docs/basketball-bridge-book/verification/contracts-junit.xml`.
The first restricted-sandbox attempt had 118 passes and a Metal import failure.
The authorized Metal-runtime retry passed all 119 tests. Existing PyTorch ONNX
deprecation/tracing warnings remain; the tests include recurrent export parity checks.

`scene-probes.json` records actual model construction and four finite zero-action
transitions per scene, with XML actuators. It proves construction, not BAM fidelity or skill.
`ppo-arithmetic.json` records the pure-Python numerical exercise, not measured training.
The new deterministic BAM reward diagnostic records 500 steps, no resets, true timeout
at 10 seconds, and no termination; its exact settings/hash are in `assets/raw/`.

## Visual corrections

Chinese font lookup requires access to macOS system fonts/font caches. The build uses
existing fonts and native XeTeX line breaking, without installing a new TeX package.
Mixed Chinese code blocks use a CJK-capable font; actual English/Python source uses Menlo.
The over-wide acceptance equation was split into aligned lines. Long path tokens and
TOC page-number widths were corrected. A visual inspection found stretched charts;
the final build preserves image aspect ratios. PDF headings omit duplicate manually
written subsection numbers while Markdown preserves chapter-source headings.

The representative-page list is in `preview-pages.json`. Automated validation checks
all PDF pages for text outside page bounds and checks build logs for missing glyphs,
overfull boxes and oversized floats. See the final `book-validation.json` for exact
file hashes and page counts. Representative pages are inspected separately for readable
equations, Chinese text, reward tables, charts, genuine screenshots and source listings.

## Remaining boundaries

- No clean-machine dependency installation was performed for the book.
- Historical full training was not rerun; source actors/assets remain prerequisites.
- Original basketball training reward history is absent, visibly marked rather than invented.
- Accepted basketball rolling/steering and full bridge crossing remain unsolved.
- This book does not assert that unrelated workspace test suites are clean.
