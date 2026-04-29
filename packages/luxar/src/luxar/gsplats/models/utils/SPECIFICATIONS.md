# luxar.gsplats.models.utils — Technical Specification

**Version**: 1.0.0
**Last Updated**: 2026-04-28

## Purpose

Low-level numerical utilities shared by the Gaussian splat models. Two operations live here, deliberately split into their own submodule so they can be reused without pulling in the full model graph:

1. **`stable_inverse_softplus`** — invert the softplus activation in a numerically stable way, used to initialize raw learnable parameters from desired post-activation values (e.g., desired Cholesky-diagonal magnitudes → raw parameter space).
2. **`solve_lower_triangular`** — wrap PyTorch's triangular solve so callers don't have to care about API drift between PyTorch versions.

Both functions are tiny but on the hot path during model initialization and the Cholesky-factorization branch of the splat parameterization, so the implementations are tuned for numerical stability, GPU-friendly control flow, and zero CPU↔GPU transfers.

---

## Core Concepts

### Inverse softplus

The standard softplus activation is

```
softplus_β(x) = (1/β) · log(1 + exp(β·x))
```

Its inverse is well-defined for `y > 0`:

```
inverse_softplus_β(y) = (1/β) · log(exp(β·y) − 1)
                      = (1/β) · log(expm1(β·y))      ← preferred form
```

The naive form `log(exp(β·y) − 1)` suffers two failure modes:
- For small `β·y`, `exp(β·y) − 1` loses precision via *catastrophic cancellation* against 1; `expm1` avoids this by computing the difference accurately.
- For large `β·y` (here, `β·y ≥ 50`), `exp(β·y)` overflows a single-precision float; the asymptotic identity `log(exp(z) − 1) ≈ z` lets us return `y` directly with no loss.

We split on the threshold `β·y ≥ 50` and pick the appropriate branch. This is exact (no further approximation) for the large-magnitude branch and numerically stable for the small-magnitude branch.

### Lower-triangular solve

`solve_lower_triangular(L, B)` returns `X` such that `L @ X = B` for a lower-triangular `L`, via forward substitution. The model uses this when materializing covariance matrices from a Cholesky factor `Σ = L · Lᵀ` and propagating gradients back through it.

The PyTorch API for this has shifted across versions (`torch.triangular_solve` → `torch.linalg.solve_triangular`). We pin to the modern API; the wrapper exists primarily as a documented entry point and to gather notes about MPS-backend gotchas.

---

## Algorithms

### `stable_inverse_softplus(y, beta=1.0)` — NumPy

**Inputs**:
- `y: np.ndarray` — values in `(0, ∞)` (the output range of softplus). Non-positive values trigger a `RuntimeWarning` and produce undefined output.
- `beta: float` — softplus scaling parameter (default `1.0`).

**Outputs**:
- `np.ndarray` of the same shape as `y`, dtype preserved (`float32` or `float64`).

**Algorithm**:
```
1. Cast input to float32 or float64 (preserve precision class).
2. Compute z = β · y elementwise.
3. mask = z >= 50.0
4. result[mask]    = y[mask]                        # asymptotic branch
5. result[~mask]   = log(expm1(z[~mask])) / β       # stable branch
6. Cast back to original dtype.
```

**Complexity**: O(n) elementwise; a single boolean mask + two vectorized operations.

**Edge cases**:
- Non-positive `y`: warning emitted; output values for those entries are unspecified (`log` of zero or negative). Callers must validate inputs.
- `y == 0`: not a valid input for softplus inverse; behaviour matches the warning above.

### `stable_inverse_softplus_torch(y, beta=1.0)` — PyTorch

Identical mathematics to the NumPy version, but expressed with `torch.where` so the entire computation runs on the input tensor's device with no CPU↔GPU traffic. Specifically:

```
result = where(β·y ≥ 50,
               y,                          # asymptotic
               log(expm1(β·y)) / β)        # stable
```

**Critical**: this is GPU-safe. The earlier numpy implementation would break the autograd graph if used inside a forward pass; the `torch.where` form preserves gradients through both branches (the `large` branch's value is exactly `y`, so gradient is 1, which is also the asymptotic gradient of the true inverse — consistent).

**Edge cases**:
- Negative or zero `y`: produces `nan` in the stable branch (`expm1(β·y) ≤ 0` → `log` of non-positive). Callers are expected to clamp inputs to a positive range before calling.

### `solve_lower_triangular(L, B)`

**Inputs**:
- `L: torch.Tensor` of shape `(d, d)` or `(N, d, d)` — lower-triangular coefficient matrix. Upper-triangular elements are ignored by the underlying kernel.
- `B: torch.Tensor` of shape `(d, P)` or `(N, d, P)` — right-hand side with `P` solution vectors stacked column-wise. Batch dimensions of `L` and `B` must broadcast.

**Outputs**:
- `torch.Tensor` of the same batch+shape as `B`, the unique `X` satisfying `L @ X = B`.

**Algorithm**: forward substitution via `torch.linalg.solve_triangular(L, B, upper=False)`. Numerically stable for well-conditioned `L`; condition number of `L` controls the condition number of the solution.

**Complexity**: O(d² · P) per system, batched over `N`.

**Edge cases**:
- Singular `L` (zero on the diagonal): `nan`/`inf` in the corresponding solution rows. The model layer guarantees positive softplus-activated diagonals to avoid this.
- Shape mismatch: a `RuntimeError` from PyTorch — not caught here; callers see the underlying error.

---

## Performance Notes

### Apple Silicon (MPS) gotcha

On PyTorch 2.5 and earlier, `torch.linalg.solve_triangular` on the MPS backend was **~10× slower than CPU** for typical splat batch sizes, and had a device-check bug fixed in PyTorch PR #142477 (December 2024). Callers running on Apple Silicon may want to dispatch this specific operation to CPU even when the rest of the model is on MPS — the cost of the device transfer is amortized by the speedup. The wrapper does not auto-dispatch; it's the model's responsibility.

### Branch threshold (50)

The `β·y ≥ 50` threshold for the asymptotic branch is conservative. `exp(50) ≈ 5.18 × 10²¹`, which fits comfortably in float64 but overflows float32 (`exp(89) ≈ 4.5 × 10³⁸` is the float32 cap). Choosing 50 means even float32 inputs use the safe asymptotic branch well before any risk of overflow, with the inverse-softplus error introduced by the asymptotic substitution bounded by `log(1 - exp(-50)) ≈ -1.9 × 10⁻²²` — entirely negligible.

---

## Validation Rules

- `y > 0` for all `stable_inverse_softplus*` calls. Warnings on the NumPy variant; silent `nan` on the PyTorch variant. Callers are expected to validate.
- `L` lower-triangular for `solve_lower_triangular`; the underlying solver ignores the upper triangle, but the assumption is part of the contract.

---

## Cross-Language Compatibility

Pure Python / PyTorch / NumPy. No serialization, no cross-language considerations. The output of `stable_inverse_softplus` is a learnable raw parameter that gets re-activated through `softplus` at forward time, so the only cross-language concern is bit-for-bit reproducibility of the activation function — handled by `torch.nn.functional.softplus` (NumPy reference: `np.log1p(np.exp(x))` with similar branching).

---

## Related Specifications

- `luxar.gsplats.models.gsplats` — consumes `stable_inverse_softplus` to initialize `raw_diag` parameters from desired sigma magnitudes, and `solve_lower_triangular` in the Cholesky-based covariance pathway. (See `../gsplats/SPECIFICATIONS.md`.)
- `luxar.gsplats.fitting` — calls into the model layer during the gradient-descent fit loop; depends transitively on these utilities. (See `../../fitting/SPECIFICATIONS.md`.)

---

## Changelog

- **v1.0.0** (2026-04-28): Initial specification
  - Documented the dual NumPy/PyTorch `stable_inverse_softplus` API, the `β·y ≥ 50` asymptotic branch threshold, and the precision/overflow rationale.
  - Documented `solve_lower_triangular` as a thin wrapper around `torch.linalg.solve_triangular`, including the MPS performance gotcha.
