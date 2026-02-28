# luxar.validation.tests - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2026-02-28

## Purpose

Tests for the Luxar validation module: input validation for positions, colors, radii, sharpness, configs, types, nD data, and helpful error messages.

---

## Test Files

| File | Description |
|------|-------------|
| `test_points_validation.py` | Validation of positions, colors, radii, and sharpness arrays |
| `test_config_validation.py` | Validation functions in `config.py` module |
| `test_base_validation.py` | Direct unit tests for `validation/base.py` functions |
| `test_validation_module.py` | Module-level validation with helpful error messages |
| `test_validation_nd.py` | nD validation (multi-dimensional data constraints) |
| `test_types_validation.py` | Validation functions in `types.py` module |

**Total**: 6 test files

---

## Key Test Patterns

- Validation tests verify both acceptance of valid inputs and rejection of invalid inputs with clear error messages.
- nD validation tests cover higher-dimensional edge cases (4D+).
- Tests use `pytest.raises` with `match` parameter to verify error message content.

---

## Related Specifications

- `luxar.validation` package: `../SPECIFICATIONS.md`

---

## Changelog

- **v1.1.0** (2026-02-28): Rewritten with actual test file inventory (6 files).
- **v1.0.0** (2026-01-02): Initial specification (boilerplate).
