# luxar.tests - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2026-02-28

## Purpose

Root test helpers package for the Luxar Python library. This package provides shared test infrastructure (conftest fixtures, helpers) used by subpackage test suites. It does not contain test files of its own.

---

## Test Files

No test files (`test_*.py`) in this directory. Contains only `__init__.py` for package initialization.

All actual tests live in subpackage `tests/` directories:
- `luxar.cli.tests` - CLI commands and network simulation
- `luxar.core.tests` - Scene graph, dimensions, transforms
- `luxar.encoding.tests` - Encoding/decoding pipeline
- `luxar.io.tests` - Zarr I/O, compiler, round-trip
- `luxar.validation.tests` - Input validation
- `luxar.typing_utils.tests` - Config and enum types
- `luxar.utils.tests` - Utilities, demos, array helpers
- `luxar.gsplats.tests` - Gaussian splatting top-level
- `luxar.gsplats.{subpackage}.tests` - Subpackage-specific tests

---

## Changelog

- **v1.1.0** (2026-02-28): Rewritten with actual content. Documented that this is a helpers-only package.
- **v1.0.0** (2026-01-02): Initial specification (boilerplate).
