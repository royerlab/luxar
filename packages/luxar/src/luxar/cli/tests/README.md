# luxar.cli.tests

Tests for the Luxar CLI package.

## Test Files

- `_testing.py` - Shared `normalized_cli_output()` helper; use it for assertions
  on CLI output
- `test_cli.py` - Core CLI command tests (demo, serve, viewer, info, profiles)
- `test_cli_enhanced.py` - Enhanced CLI tests with extended coverage
- `test_cli_integration.py` - Integration tests for full CLI workflows,
  including data/viewer HTTP cache-policy boundaries
- `test_cli_utils.py` - Unit tests for CLI utility functions (port finding, browser, tree formatting, zarr info)
- `test_export.py` - Tests for the standalone scene export command
- `test_network_simulation.py` - Tests for network simulation middleware and profiles
- `test_gsplat_cli_extended.py` - Tests for gsplat subcommands
- `test_restamp_lod_command.py` - What `luxar restamp-lod` adds over
  `luxar.io.lod_restamp`: the printed old→new audit trail (with each anchor
  pinned to its own group path on one line, so swapping the two labels cannot
  pass), `--dry-run`, a repeatable `--group`, `--anchor` re-derivation, and the
  exit code — 1 for an invalid anchor before any write, an unmatched `--group`,
  a missing or non-Luxar store, a re-verification residual from a genuinely stale
  consolidated index, or a ladder left alone for a reason worth acting on (which
  still writes the ladders it could convert)
- `test_gsplat_content_scoped_metrics.py` - Every `gsplat` subcommand (the
  `batch-fit` group included) is classified as content-changing or not, and a
  rewrite must drop (or keep) the fit's measured scores accordingly — through both
  writers, and identically for a flat store and a partition

## Running

```bash
hatch run pytest packages/luxar/src/luxar/cli/tests/
```
