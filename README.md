**RINEX Split & Merge CLI**

This tool splits RINEX observation files at gaps in observation epochs and merges adjacent files across boundaries when they align at the sampling interval. It preserves the original header for each split segment and removes duplicate headers when merging.

Features
- Split within files at observation time gaps (v2 and v3).
- Merge across files when adjacent at the sampling interval.
- Preserve header for each split; remove duplicate header on merge.
- Concise structured logging of operations.
- Works on large files via streaming line processing.

Usage
- Basic:
  - `rinex-stitch file1.obs file2.obs` 
- With glob:
  - `rinex-stitch data/*.rnx`
- Options:
  - `--output-dir OUT` directory for results (default: `out`)
  - `--interval SECONDS` override sampling interval (auto-detect by default)
  - `--tolerance SECONDS` time tolerance when detecting gaps/adjacency (default: 0.1)
  - `--no-merge` disable cross-file merging
  - `--dry-run` print planned actions without writing files
  - `--log-level LEVEL` set logging level (INFO, DEBUG, etc.)

Install
- Standard Python 3.9+.
- Optional libraries:
  - `georinex` for additional RINEX validation (detected if installed).
  - `rich` for colorized logs (detected if installed).
- From this repo root:
  - `python -m pip install .`

Notes
- The tool detects RINEX version from the header and parses epoch blocks without decoding observation values to ensure speed and fidelity.
- Sampling interval is auto-detected as the most frequent epoch delta across input data; you can override with `--interval`.
- Files are written as `<stem>_<YYYYmmddTHHMMSS>_<YYYYmmddTHHMMSS>.rnx` by default.

