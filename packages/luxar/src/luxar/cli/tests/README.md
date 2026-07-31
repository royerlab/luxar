# luxar.cli.tests

Tests for the Luxar CLI package.

## Test Files

- `test_cli.py` - Core CLI command tests (demo, serve, viewer, info, profiles)
- `test_cli_enhanced.py` - Enhanced CLI tests with extended coverage
- `test_cli_integration.py` - Integration tests for full CLI workflows,
  including data/viewer HTTP cache-policy boundaries
- `test_cli_utils.py` - Unit tests for CLI utility functions (port finding, browser, tree formatting, zarr info)
- `test_export.py` - Tests for the standalone scene export command
- `test_network_simulation.py` - Tests for network simulation middleware and profiles
- `test_gsplat_cli_extended.py` - Tests for gsplat subcommands

## Running

```bash
hatch run pytest packages/luxar/src/luxar/cli/tests/
```
