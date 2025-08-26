# Roadmap / TODO

This file tracks near-term tasks and owner expectations.

## Test Fixtures & CI Tests
- Owner: You (fixtures), Me (tests)
- Action: You will provide representative RINEX observation fixtures (small, anonymized if needed).
- Then I will add:
  - Unit tests for split/merge logic (gap detection, adjacency, header compatibility).
  - E2E tests using the fixtures, exercised in CI.

## Nice-to-haves
- Add sample CLI transcripts for common workflows (README “Examples”).
- Optional `--name-template` tokens sourced from header (e.g., station/marker).
- Validate headers with `georinex` when available (non-blocking).

## Done (recent)
- Cross-file merge improvements (header normalization, approx position tolerance).
- Version + git SHA logging; DEBUG merge diagnostics.
- Release/CI workflows (lint, build, smoke test; tagged releases attach wheels).
