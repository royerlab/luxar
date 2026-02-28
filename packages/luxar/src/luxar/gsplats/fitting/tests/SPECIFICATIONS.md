# luxar.gsplats.fitting.tests - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2026-02-28

## Purpose

Tests for the Gaussian splatting fitting pipeline: configuration, initialization, loss functions, optimization loop, preprocessing, result structures, validation, and visualization.

---

## Test Files

| File | Description |
|------|-------------|
| `test_fitting_config.py` | Fitting configuration dataclasses |
| `test_initialization.py` | `fitting/initialization.py` module (parameter initialization) |
| `test_losses.py` | `fitting/losses.py` module (loss functions for optimization) |
| `test_optimization.py` | `fitting/optimization.py` module (optimization loop) |
| `test_fitting_preprocessing.py` | Fitting preprocessing module (volume preparation) |
| `test_results.py` | `fitting/results.py` module (result data structures) |
| `test_fitting_validation.py` | Fitting validation module (input/parameter validation) |
| `test_visualization.py` | `fitting/visualization.py` module (fitting result visualization) |

**Total**: 8 test files

---

## Key Test Patterns

- Config tests verify dataclass defaults, validation, and serialization.
- Loss function tests verify gradient computation and convergence behavior.
- Optimization tests verify the training loop with small synthetic volumes.
- Validation tests verify rejection of invalid fitting parameters.

---

## Related Specifications

- `luxar.gsplats.fitting` package: `../SPECIFICATIONS.md`
- Dynamic ops tests: `../dynamic_ops/tests/SPECIFICATIONS.md`

---

## Changelog

- **v1.1.0** (2026-02-28): Rewritten with actual test file inventory (8 files).
- **v1.0.0** (2026-01-02): Initial specification (boilerplate).
