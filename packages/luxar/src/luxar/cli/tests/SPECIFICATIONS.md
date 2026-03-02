# luxar.cli.tests - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2026-02-28

## Purpose

Tests for the Luxar command-line interface, including all subcommands (`demo`, `serve`, `info`, `profiles`, `export`), CLI utility functions, and network simulation middleware.

---

## Test Files

| File | Description |
|------|-------------|
| `test_cli.py` | Core CLI command tests (demo, serve, info, profiles) |
| `test_cli_integration.py` | Integration tests for CLI commands with no mocking of internal server logic |
| `test_cli_enhanced.py` | Tests for enhanced CLI commands and features |
| `test_cli_utils.py` | Tests for `cli/utils.py` module (helper functions) |
| `test_export.py` | Tests for the `luxar export` command |
| `test_network_simulation.py` | Tests for network simulation middleware and utilities |

**Total**: 6 test files

---

## Key Test Patterns

- CLI tests use `click.testing.CliRunner` for invoking commands without subprocess overhead.
- Integration tests avoid mocking internal logic to verify real behavior.
- Network simulation tests verify bandwidth limiting and latency injection.

---

## Related Specifications

- `luxar.cli` package: `../SPECIFICATIONS.md`
- Network simulation: `docs/guides/developer/NETWORK_SIMULATION_SPEC.md`

---

## Changelog

- **v1.1.0** (2026-02-28): Rewritten with actual test file inventory and descriptions.
- **v1.0.0** (2026-01-02): Initial specification (boilerplate).
