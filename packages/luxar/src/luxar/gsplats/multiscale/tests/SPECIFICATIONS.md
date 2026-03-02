# luxar.gsplats.multiscale.tests - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2026-02-28

## Purpose

Tests for multiscale Gaussian splatting decomposition: basic decomposition, advanced decomposition features, and energy distribution analysis.

---

## Test Files

| File | Description |
|------|-------------|
| `test_decomposition_basic.py` | Basic multiscale decomposition tests |
| `test_decompose_advanced.py` | Advanced tests for multiscale decomposition |
| `test_energy_distribution.py` | Energy distribution analysis across scales |

**Total**: 3 test files

---

## Key Test Patterns

- Decomposition tests verify that the sum of components reconstructs the original signal.
- Energy distribution tests verify that scale hierarchy is correctly ordered.

---

## Related Specifications

- `luxar.gsplats.multiscale` package: `../SPECIFICATIONS.md`

---

## Changelog

- **v1.1.0** (2026-02-28): Rewritten with actual test file inventory (3 files).
- **v1.0.0** (2026-01-02): Initial specification (boilerplate).
