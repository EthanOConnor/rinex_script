# Repository Guidelines

## Project Structure & Module Organization
- `src/rinex_stitch/`: Python package and CLI (`cli.py`, `__main__.py`).
- `pyproject.toml`: Metadata, dependencies, ruff/black config, entry points.
- `.github/workflows/`: CI (lint, build, smoke-test) and Release (tag-driven wheels).
- `README.md`: User and developer guide.
- `AGENTS.md`: Contributor guidelines (this file).
- Suggested tests folder: `tests/` (not present yet).

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv && source .venv/bin/activate` (Windows: `\.venv\Scripts\Activate.ps1`).
- Dev install: `python -m pip install -e .[dev]`.
- Run CLI: `rinex-stitch --help` or `python -m rinex_stitch --help`.
- Lint: `ruff check .` (CI uses this).
- Format: `black .`.
- Build artifacts: `python -m build` (creates `dist/*.whl` and `*.tar.gz`).
- Smoke run locally: `python -m rinex_stitch data/*.rnx --log-level DEBUG`.

## Coding Style & Naming Conventions
- Language: Python 3.9+ (CI: 3.9–3.12).
- Line length: 100 (ruff/black configured in `pyproject.toml`).
- Indentation: 4 spaces; descriptive variable/function names.
- Filenames: snake_case for modules; hyphenated CLI name `rinex-stitch`.

## Testing Guidelines
- Framework: Not set up yet; contributions welcome.
- Place tests under `tests/` mirroring `src/` structure.
- Keep test names descriptive, e.g., `test_merge_adjacent.py`.
- Add quick sample files for e2e tests when practical.

## Commit & Pull Request Guidelines
- Conventional commits encouraged: `feat:`, `fix:`, `docs:`, `chore:`, `ci:`, `refactor:`.
- Keep commits focused; include rationale in body where helpful.
- PRs should include: summary, motivation/linked issue, screenshots/log snippets (e.g., DEBUG merge logs) if relevant, and validation steps.
- Ensure `ruff`/`black` pass; CI must be green.

## Release & Versioning
- Version in `pyproject.toml` and `src/rinex_stitch/__init__.py`.
- Tag with `vX.Y.Z` to trigger Release workflow that uploads wheels.
- CLI logs version and short git SHA at startup when available.

## Security & Configuration Tips
- Do not commit secrets; Releases use `GITHUB_TOKEN` only.
- Avoid large binaries; include small sample RINEX files for tests if needed.
- For reproducibility, prefer `pipx install git+https://github.com/<owner>/rinex_script.git` for users.
