**RINEX Split & Merge CLI**

This tool splits RINEX observation files at gaps in observation epochs and merges adjacent files across boundaries when they align at the sampling interval. It preserves the original header for each split segment and removes duplicate headers when merging.

Features
- Split within files at observation time gaps (v2 and v3).
- Merge across files when adjacent at the sampling interval.
- Preserve header for each split; remove duplicate header on merge.
- Concise structured logging of operations.
- Works on large files via streaming line processing.

Quick Start
- Requirements: Python 3.9+.
- Run once without installing (isolated):
  - `pipx run git+https://github.com/EthanOConnor/rinex_script.git -- --help`
- Install as a user app (recommended):
  - `pipx install git+https://github.com/EthanOConnor/rinex_script.git`
- Or install into current environment:
  - `python -m pip install git+https://github.com/EthanOConnor/rinex_script.git`

Usage
- CLI entrypoint:
  - `rinex-stitch file1.obs file2.obs`
- Module entrypoint (no PATH changes required):
  - `python -m rinex_stitch file1.obs file2.obs`
- With glob:
  - `rinex-stitch data/*.rnx`
- Options:
  - `--output-dir OUT` directory for results (default: `out`)
  - `--interval SECONDS` override sampling interval (auto-detect by default)
  - `--tolerance SECONDS` time tolerance when detecting gaps/adjacency (default: 0.1)
  - `--no-merge` disable cross-file merging
  - `--dry-run` print planned actions without writing files
  - `--log-level LEVEL` set logging level (INFO, DEBUG, etc.)

Examples
- Split by gaps only (auto-detect interval):
  - `rinex-stitch day1.obs --output-dir out`
- Merge adjacent daily files at 5s sampling:
  - `rinex-stitch day1.obs day2.obs --interval 5 --output-dir out`
- Plan actions without writing:
  - `rinex-stitch data/*.rnx --dry-run --log-level DEBUG`

Cross‑Platform Notes
- Windows PowerShell and cmd are supported. If `rinex-stitch` isn’t found on PATH after install, use the module form: `python -m rinex_stitch ...` (or `py -m rinex_stitch ...`).
- Globbing is performed by your shell. If a glob does not expand, quote it or use unquoted in shells that expand (e.g., Bash, Zsh). The tool accepts multiple explicit files regardless.

Installation Options
- pipx (isolated, recommended for CLI tools):
  - Install: `pipx install git+https://github.com/EthanOConnor/rinex_script.git`
  - Upgrade: `pipx upgrade rinex-stitch`
  - Run without install: `pipx run git+https://github.com/EthanOConnor/rinex_script.git -- --help`
- pip (current environment):
  - From GitHub: `python -m pip install git+https://github.com/EthanOConnor/rinex_script.git`
  - From local clone: `python -m pip install .`
- Module execution (no install):
  - Clone this repo, then: `python -m rinex_stitch --help`

Developer Guide
- Clone and set up an isolated environment:
  - Unix/macOS:
    - `python3 -m venv .venv && source .venv/bin/activate`
  - Windows (PowerShell):
    - `py -m venv .venv; .venv\Scripts\Activate.ps1`
- Editable install with extras:
  - `python -m pip install -U pip setuptools wheel`
  - `python -m pip install -e .[rich,georinex]`
- Run the CLI during development:
  - `rinex-stitch --help` or `python -m rinex_stitch --help`
- Linting/formatting (optional suggestions):
  - Ruff: `python -m pip install ruff && ruff check src`
  - Black: `python -m pip install black && black src`
- Build a wheel/sdist (optional):
  - `python -m pip install build`
  - `python -m build`
- Running from source tree (no install):
  - `PYTHONPATH=src python -m rinex_stitch --help` (Unix)

Behavior & Notes
- Version detection: RINEX version is read from the header; epoch lines are parsed as v3 when starting with `>` and as v2 otherwise.
- Interval detection: Auto-detected as the most common difference between consecutive epoch timestamps across all inputs; override with `--interval` for certainty.
- Splitting: A new output segment starts when the epoch gap exceeds `interval + tolerance`.
- Merging: Adjacent segments with compatible headers merge when the next starts within `interval + tolerance` of the previous end. Duplicate epoch at exact boundary is dropped.
- Output names: `<stem>_<YYYYmmddTHHMMSS>_<YYYYmmddTHHMMSS>.rnx`, where `stem` is the common prefix of source filenames.

Optional Libraries
- `rich`: Colorized logs (detected automatically if installed).
- `georinex`: For potential additional RINEX validation or future extensions. The current parser streams epoch blocks without heavy dependencies and works without `georinex`.

Limitations
- This tool focuses on RINEX observation files (O-files) and does not parse or interpret observation values—only epoch boundaries and raw block content.
- Extremely irregular files may benefit from explicitly setting `--interval`.

