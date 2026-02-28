# luxar.gsplats.fitting.dynamic_ops.tests - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2026-02-28

## Purpose

Tests for dynamic Gaussian splatting operations: fixed-pool relocation (split, merge, relocate) and the recently-relocated tracker cooldown mechanism.

---

## Test Files

| File | Description |
|------|-------------|
| `test_dynamic_ops.py` | Dynamic Gaussian splat operations with fixed-pool relocation |
| `test_relocation_tracker.py` | RecentlyRelocatedTracker cooldown mechanism |

**Total**: 2 test files

---

## Key Test Patterns

- Dynamic ops tests verify that split/merge/relocate operations maintain the fixed pool size invariant.
- Tracker tests verify cooldown timing and recently-relocated state management.

---

## Related Specifications

- `luxar.gsplats.fitting.dynamic_ops` package: `../SPECIFICATIONS.md`

---

## Changelog

- **v1.1.0** (2026-02-28): Rewritten with actual test file inventory (2 files).
- **v1.0.0** (2026-01-02): Initial specification (boilerplate).
