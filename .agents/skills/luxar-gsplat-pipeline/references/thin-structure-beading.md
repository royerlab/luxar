# Thin filaments render as chains of beads: the six-knob schedule

Loaded from the `luxar-gsplat-pipeline` skill ("The Python API defaults to 1000
iters"). Measured on a 1-voxel-wide neuron dataset.

**Symptom: thin filaments (axons, vessels, fibres) render as chains of beads.**
Under-converged splats never leave their seed shape. Edge seeding initialises them
*isotropic at σ = 1.0 voxel*, so a 1-voxel-wide filament is rebuilt from 1-voxel
spheres spaced ~2.5σ apart, which beads by construction. Diagnose by measuring the
fitted shapes — a median axis ratio near 1.0-1.3 with σ ≈ 1 voxel means the
optimizer never moved them, not that the basis cannot represent the structure.

Six defaults must move TOGETHER; any one alone underperforms:

```python
result = fit_gaussian_splats(
    volume,
    n_iters=10_000,               # 1000 leaves splats at their seed shape
    patience=200,                 # 15 decays the shape LR away early
    early_stop_patience=2000,     # 300 stops before shapes settle
    enable_dynamic_ops=False,     # relocation RESETS splats to isotropic mid-fit
    max_eccentricity=None,        # 10.0 caps the axis ratio at sqrt(10)
    l1_diag=0.0,                  # the default penalty pulls toward isotropy
)
```

Measured on a 1-voxel-wide neuron dataset (skeleton points dipping below 25% of
their local ridge / foreground PSNR): 1000 iters 22.2% / 22.94 dB → 10000 iters
with the above 16.5% / **25.85 dB**, against 11.9% for the raw data itself. Raising
`n_iters` while leaving relocation on LOSES ~1 dB, because it resets the shapes the
extra iterations just bought. 20000 iterations adds only +0.2 dB — beading converges
by ~5000, fidelity by ~10000.

**What does NOT fix beading** (all measured, so don't retry them): more seeds makes
it *worse* (finer subdivision shrinks σ faster than the gaps, so spacing went 2.45σ →
3.22σ); fewer seeds gives tighter spacing and still beads more; and widening the
render truncation radius barely moves it — the splats genuinely do not reach each
other.

Cost is fit time only (~2x), and it is one-time if you cache. **If you cache fits,
key the cache on the schedule too** — it changes splat SHAPES while leaving the
count identical, so a cache keyed only on seeds/floor/retention silently returns the
old fit and makes a retune look like a no-op.
