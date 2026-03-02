# luxar.gsplats.utils.tests - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2026-02-28

## Purpose

Tests for Gaussian splatting utility functions: triangular matrix packing/unpacking (`pack_tril`, `unpack_tril`, `permute_cholesky_packed`, `embed_cholesky_packed`).

---

## Test Files

| File | Description |
|------|-------------|
| `test_trils.py` | Triangular matrix packing/unpacking utilities |

**Total**: 1 test file

---

## Key Test Patterns

- Tests verify round-trip fidelity: `unpack_tril(pack_tril(L))` recovers the original lower triangular matrix.
- Covariance preservation is verified by checking that `L @ L^T` is unchanged after permutation/embedding operations.

---

## Related Specifications

- `luxar.gsplats.utils` package: `../SPECIFICATIONS.md`

---

## Changelog

- **v1.1.0** (2026-02-28): Rewritten with actual test file inventory (1 file).
- **v1.0.0** (2026-01-02): Initial specification (boilerplate).
