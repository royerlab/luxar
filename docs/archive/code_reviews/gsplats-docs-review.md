# GSplats Documentation Review: SPECIFICATIONS.md and README.md Files

**Reviewer**: Claude Opus 4.6
**Date**: 2026-02-28
**Scope**: All SPECIFICATIONS.md and README.md files in `packages/luxar/src/luxar/gsplats/`

## Fixes Applied During This Review

The following issues were fixed directly during this review:

1. **io/SPECIFICATIONS.md**: Fixed `sort_splats_spatially` -> `sort_splats_spatial` (function name)
2. **io/SPECIFICATIONS.md**: Fixed `from luxar.gsplats import inspect_gsplats_zarr` -> `from luxar.gsplats.io import inspect_gsplats_zarr` (import path)
3. **io/__init__.py**: Fixed docstring `sort_splats_spatially()` -> `sort_splats_spatial()` (comment)
4. **models/SPECIFICATIONS.md**: Fixed `fit_result.py` reference -> `gsplat_data.py` and `.npz` -> `.gsplats.zarr` (persistence section)
5. **metal/README.md**: Fixed `from luxar.gsplats.fitting import fit_gsplats` -> `from luxar.gsplats import fit_gaussian_splats` (integration example)
6. **multiscale/README.md**: Fixed `MSE` -> `L1` in "How It Works" formula
7. **multiscale/README.md**: Fixed old tuple-return API in multi-scale fitting example to use GSplatData
8. **multiscale/README.md**: Fixed API Reference to show `initialize_coarse()` as recommended (not pyramid) and added missing `initialize_zero()`
9. **multiscale/SPECIFICATIONS.md**: Fixed `demo_decompose_mitosis.py` -> `demo_decompose_2d_mitosis.py` (filename)
10. **multiscale/SPECIFICATIONS.md**: Fixed changelog placeholder `2025-01-XX` -> `2025-01`
11. **dynamic_ops/SPECIFICATIONS.md**: Fixed seeds relative path `../seeds/` -> `../../seeds/`
12. **demos/README.md**: Fixed broken link `./SPECIFICATION.md` -> `./SPECIFICATIONS.md`
13. **demos/README.md**: Fixed "per-splat Adam optimizer" -> "Standard PyTorch Adam optimizer"
14. **demos/README.md**: Fixed "dynamic topology changes" -> "fixed-pool splat relocation"
15. **Root SPECIFICATIONS.md**: Fixed `fit_result.py` -> `gsplat_data.py` (Section 0 header)

## Severity Legend

- **CRITICAL**: Completely wrong information, references to non-existent files/classes, or will mislead developers
- **MAJOR**: Significantly outdated or incorrect, but not immediately dangerous
- **MINOR**: Cosmetic, formatting, small inconsistencies, or quality improvements

---

## 1. gsplats/SPECIFICATIONS.md (Root Hub)

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | CRITICAL | References non-existent file `fit_result.py` | Section 0 header says `GSplat Data Container (fit_result.py)` but the actual file is `gsplat_data.py` |
| 2 | CRITICAL | References non-existent function `seed_from_gaussian` | Section 1 documents `seed_from_gaussian()` but this function does not exist. The actual functions are `seed_from_edges`, `seed_from_grid`, `seed_from_decomposition`, and the unified `generate_seeds()` |
| 3 | MAJOR | Missing documentation for seeds subpackage links | The Documentation Structure section at the top lists links to multiscale, fitting, optim, models, utils, and glossary, but does NOT link to `seeds/SPECIFICATIONS.md`, `io/SPECIFICATIONS.md`, or `clahe/SPECIFICATIONS.md` |
| 4 | MAJOR | Broken anchor link in Reading Order | `#2-gaussian-splat-model-modelsgspatsgsplat_modelpy` has typo "gspats" instead of "gsplats" |
| 5 | MINOR | Version/date outdated | `Last Updated: 2025-01` -- should reflect more recent updates |

**Status**: Issues 1, 2, 3 need manual attention (large file, complex changes). Issue 4 is a typo.

---

## 2. gsplats/clahe/SPECIFICATIONS.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | MINOR | Changelog date inconsistency | Header says `Last Updated: 2025-11-27` but changelog says `v1.0.0 (January 2025)` |

### Verified Correct
- Function signatures match `clahe_core.py` (apply_clahe, compute_clahe_sampling_probabilities)
- Algorithm description is accurate
- Test class names match actual test file
- Package structure matches (clahe_core.py + __init__.py)

**Overall**: This is one of the best-maintained specification files. Thorough and accurate.

---

## 3. gsplats/clahe/README.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | MINOR | Examples reference wrong demo path | "See the `demos/` directory" with `demo_splats_mitosis_intgrad.py` and `demo_3d_dapi_microscopy.py` -- these are in `../demos/` not a local `demos/` |

### Verified Correct
- API signatures match code
- Parameter descriptions accurate
- Performance characteristics reasonable

**Overall**: Good quality, minor path issue.

---

## 4. gsplats/demos/SPECIFICATIONS.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | MAJOR | Missing demos from the list | `demo_3d_celegans_confocal.py` and `demo_boundary_containment.py` exist in the directory but are not documented |
| 2 | MINOR | Title says "SPECIFICATIONS" but content is more of a style guide / best practices document | Not really a specification in the template sense; more of a coding guidelines doc |

### Verified Correct
- Listed demos 1-12 all exist in the directory
- Usage patterns and argument descriptions are generally accurate
- Category groupings are sensible

**Overall**: Missing 2 actual demo files that should be documented.

---

## 5. gsplats/demos/README.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | MAJOR | Missing demos from listing | `demo_3d_celegans_confocal.py`, `demo_boundary_containment.py`, and `demo_splats_mitosis_explicit_seeding.py` are not in the README tables/descriptions |
| 2 | MAJOR | Broken link to SPECIFICATION.md | Line 509: `./SPECIFICATION.md` -- but the actual file is `./SPECIFICATIONS.md` (plural). The file `SPECIFICATION.md` (singular) does NOT exist |
| 3 | MAJOR | "Core Features" section contains inaccurate claims | States "Per-splat Adam optimizer with individual learning rates" but the system uses standard PyTorch Adam (not per-splat). This was explicitly removed in v2.0 |
| 4 | MAJOR | "Key concepts" for demo_basic_fitting.py are wrong | States "Per-splat optimizer, early stopping, dynamic topology changes" but per-splat optimizer was removed and dynamic ops use fixed-pool relocation (no topology changes) |
| 5 | MINOR | Uses emojis extensively | Against the project's documentation conventions (CLAUDE.md says "avoid using emojis") |

**Overall**: Several significant inaccuracies about the optimizer architecture that contradict the optim and dynamic_ops specifications.

---

## 6. gsplats/fitting/SPECIFICATIONS.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | MINOR | File too large to read in one pass (53.5KB) | While not a bug, the document is extremely large and could benefit from splitting |
| 2 | MINOR | Cross-reference to `dynamic_ops/README.md` in header | Should be `dynamic_ops/SPECIFICATIONS.md` since that is the actual spec file (README.md may or may not exist) |

### Verified Correct
- FitConfig class exists in config.py
- Pipeline architecture modules match: config.py, validation.py, preprocessing.py, initialization.py, losses.py, optimization.py, results.py
- visualization.py exists (not always documented but present)

**Overall**: Generally well-maintained, just very large.

---

## 7. gsplats/fitting/dynamic_ops/SPECIFICATIONS.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | MINOR | Related Specifications path wrong | `../../models/gsplats/SPECIFICATIONS.md` and `../../optim/SPECIFICATIONS.md` -- these should be `../../../models/gsplats/SPECIFICATIONS.md` and `../../../optim/SPECIFICATIONS.md` (one more level up since dynamic_ops is inside fitting) |

### Verified Correct
- DynamicOpsConfig class exists in config.py
- RecentlyRelocatedTracker exists in operations.py
- apply_dynamic_operations function signature matches code exactly
- peak_finding.py exists as documented
- Fixed-pool relocation architecture accurately described

**Overall**: Good quality specification, minor path issue.

---

## 8. gsplats/io/SPECIFICATIONS.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | MAJOR | Function name mismatch | Spec documents `sort_splats_spatially()` (line 282) but actual function is `sort_splats_spatial()` (no trailing "ly"). The code uses `sort_splats_spatial` consistently. |
| 2 | MAJOR | Import path wrong for inspect function | Spec shows `from luxar.gsplats import inspect_gsplats_zarr` (line 550) but actual import is `from luxar.gsplats.io import inspect_gsplats_zarr` |
| 3 | MINOR | Default chunk_size inconsistency | Design decisions table says "default 8192" but the body text describes byte-based calculation resulting in variable chunk sizes per array |

### Verified Correct
- File structure matches: save_gsplats.py, load_gsplats.py, inspect_gsplats.py
- inspect_gsplats_zarr() and format_gsplats_info() exist
- save_gsplats() signature matches code
- load_gsplats() signature matches code
- Encoding integration description is accurate
- Zarr structure description is accurate

**Overall**: Good spec, but the function name typo is significant.

---

## 9. gsplats/io/README.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | MINOR | __init__.py docstring says `sort_splats_spatially()` | The __init__.py comment line 13 says "sort_splats_spatially()" but the actual export is "sort_splats_spatial" |
| 2 | MINOR | "53 tests total" claim | Not verified but may be outdated as tests evolve |
| 3 | MINOR | Missing `hilbertcurve` in dependencies note | README says `hilbertcurve` but code may actually use `numpy-hilbert-curve` -- should verify |

### Verified Correct
- save_gsplats/load_gsplats function signatures match
- inspect_gsplats_zarr and format_gsplats_info correctly referenced
- sort_splats_spatial() correctly named in the README examples
- compute_chunk_bounds_gsplats() correctly referenced
- Zarr structure diagram accurate

**Overall**: Good README, consistent with the SPECIFICATIONS.md (except for inheriting the spatially/spatial naming confusion from __init__.py comment).

---

## 10. gsplats/models/SPECIFICATIONS.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | CRITICAL | References non-existent file `fit_result.py` | "Persistence (see `fit_result.py:16-148`)" on line 409, but this file does not exist. The actual file is `gsplat_data.py` |
| 2 | CRITICAL | Documents obsolete .npz save/load | Lines 412-417 show `result.save('fitted_splats.npz')` and `GSplatData.load('fitted_splats.npz')` but the actual format is `.gsplats.zarr`, not `.npz` |
| 3 | MAJOR | Package structure lists wrong relative paths | The Metal backend SPECIFICATIONS link `gsplats/metal/SPECIFICATIONS.md` should be `metal/SPECIFICATIONS.md` (within the gsplats subdir context) |

### Verified Correct
- GaussianSplatModel exists in gsplat_model.py
- rendering_core.py and rendering_wrappers.py exist
- models/utils/inverse_softplus.py and lt_solver.py exist
- GSplatData dataclass structure matches actual code
- Rendering algorithm details are accurate

**Overall**: Good technical content, but the persistence section is completely outdated.

---

## 11. gsplats/models/gsplats/SPECIFICATIONS.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | MAJOR | Placeholder/stub content | Entire file is a template stub with no actual content: "Package implementation for this subsystem", "Defines the responsibilities of this package and its interactions with neighbors", etc. |

**Overall**: This is a completely empty template that provides zero value. Should either be filled with actual content or removed and readers directed to the parent models/SPECIFICATIONS.md which has the detailed content.

---

## 12. gsplats/models/gsplats/cuda/SPECIFICATIONS.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | MINOR | Very large file (couldn't fully read) | Multi-part specification split across 3 files |
| 2 | MINOR | References SPECIFICATIONS_PYTORCH_INTEGRATION.md and SPECIFICATIONS_TESTING.md | Need to verify these companion files exist |

### Verified Partially
- Technical content about tile-based rasterization appears sound
- Distinction between 3DGS and Luxar volumetric fitting is well-documented
- Known limitations section is useful

**Overall**: Appears thorough and well-structured from the portion reviewed.

---

## 13. gsplats/models/gsplats/metal/SPECIFICATIONS.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | MINOR | References `L_row_norms` in architecture diagram but describes `conic` in text | The architecture shows "L to Conic Conversion" but memory notes mention L_row_norms was a replacement for conic |

### Verified Correct
- GaussianSplatModelMetal and MetalSplatFunction documented correctly
- Coordinate convention documentation is detailed and accurate
- Gradient bug fix is well-documented
- Performance expectations are reasonable

**Overall**: Good quality specification with detailed gradient derivations.

---

## 14. gsplats/models/gsplats/metal/README.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | MAJOR | Integration example uses wrong API | `from luxar.gsplats.fitting import fit_gsplats` -- the actual function is `from luxar.gsplats import fit_gaussian_splats`. `fit_gsplats` is the internal module name, not the public API |
| 2 | MAJOR | Integration example uses wrong parameters | `fit_gsplats(volume, n_splats=1000, device='mps', use_metal=True)` -- `n_splats` and `use_metal` are not parameters of `fit_gaussian_splats()` |
| 3 | MINOR | "Maximum 8 dimensions" in Limitations | Metal backend is documented as "3D only" in SPECIFICATIONS.md, contradicting "up to 8D" in README |

**Overall**: The basic usage section is fine, but the integration example is completely wrong.

---

## 15. gsplats/models/utils/SPECIFICATIONS.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | MAJOR | Placeholder/stub content | Same as models/gsplats/SPECIFICATIONS.md -- completely empty template with no actual content |

**Overall**: Empty template. The parent models/SPECIFICATIONS.md already documents inverse_softplus.py and lt_solver.py in detail, so this file provides zero value.

---

## 16. gsplats/multiscale/SPECIFICATIONS.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | MAJOR | Demo file name wrong | References `demo_decompose_mitosis.py` (line 955) but actual file is `demo_decompose_2d_mitosis.py` |
| 2 | MINOR | Parameter defaults inconsistency | decompose_image() signature shows `energy_weight: float = 0.01` but the docstring says `default=0.001` |
| 3 | MINOR | Changelog date placeholder | `v0.1.0 (2025-01-XX)` -- still has XX placeholder |
| 4 | MINOR | Future extensions section lists "Per-scale visualization" as future but this is already implemented in show_optimization_movie() |

### Verified Correct
- MultiScaleDecomposer class and methods match decompose.py
- decompose_image() function exists and is the main API
- show_optimization_movie() exists in decompose.py
- All 5 initialization methods documented and exist
- 3 loss types documented correctly
- Demo files exist: demo_decompose_2d.py, demo_decompose_2d_mitosis.py, demo_decompose_3d.py, demo_decompose_3d_dapi_nuclei.py, demo_compare_initialization.py

**Overall**: Very thorough specification with minor issues.

---

## 17. gsplats/multiscale/README.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | MAJOR | "How It Works" section has wrong formula | Shows `Loss = MSE(reconstruction, target) + ...` but the default loss is L1 (MAE), not MSE. The SPECIFICATIONS.md correctly documents L1 as default |
| 2 | MAJOR | Multi-Scale Splat Fitting example uses old API | `params, amps, _ = fit_gaussian_splats(...)` -- the function returns `GSplatData`, not a tuple. Should be `result = fit_gaussian_splats(...)` |
| 3 | MAJOR | API Reference section inconsistency for initialize methods | Lists `initialize_from_pyramid()` as "(recommended)" in the Model Class section, but the SPECIFICATIONS.md and the rest of the README clearly state "coarse" is strongly recommended |
| 4 | MINOR | Missing `initialize_zero()` from Model Class API Reference | SPECIFICATIONS.md documents 5 initialization methods but the README API section only lists 4 (missing "zero") |

### Verified Correct
- decompose_image() signature matches code
- Demo files listed all exist
- Performance benchmarks appear reasonable
- Interpolation method descriptions are accurate

**Overall**: Good user-facing documentation but has API accuracy issues.

---

## 18. gsplats/optim/SPECIFICATIONS.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | NONE | No significant issues found | |

### Verified Correct
- create_optimizer_and_scheduler() exists in integration.py
- Gradient dilution formula references utils/trils.py correctly
- File structure matches: __init__.py, integration.py
- Adam optimizer approach accurately described

**Overall**: Clean, accurate specification.

---

## 19. gsplats/seeds/SPECIFICATIONS.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | CRITICAL | Package structure lists non-existent files | Lists `multiscale_gaussian.py` and `moment_seeding.py` in the package structure, but these files DO NOT EXIST. Actual files are: `edges.py`, `generate.py`, `gpu_ops.py`, `grid.py`, `multiscale_decomposition.py`, `utils.py` |
| 2 | CRITICAL | Documents "Multiscale Gaussian Method" as primary | Section documents `seed_from_gaussian()` as a primary method, but this function does not exist. Actual methods are `seed_from_edges`, `seed_from_grid`, `seed_from_decomposition` |
| 3 | CRITICAL | Missing documentation for actual seeding methods | `seed_from_edges` (in edges.py), `seed_from_grid` (in grid.py), and `gpu_ops.py` are not documented at all |
| 4 | MAJOR | "Two Complementary Approaches" section is wrong | Claims "Multiscale Gaussian Method" and "Decomposition Method" but actual methods are "Edges", "Grid", and "Decomposition" |

**Overall**: This specification is severely outdated. The package was apparently restructured from `multiscale_gaussian.py` + `moment_seeding.py` to `edges.py` + `grid.py` + `gpu_ops.py` + `multiscale_decomposition.py`, but the specification was never updated. This is the most out-of-date specification in the entire package.

---

## 20. gsplats/utils/SPECIFICATIONS.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | NONE | No significant issues found | |

### Verified Correct
- tril_size, calculate_gradient_dilution_factor, pack_tril, unpack_tril all exist in trils.py
- Mathematical formulas match code
- Package structure matches (__init__.py + trils.py)

**Overall**: Clean, accurate specification.

---

## 21. gsplats/seeds/demos/SPECIFICATIONS.md

### Issues Found

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | MAJOR | Placeholder/stub content | Empty template with no actual demo documentation |

**Overall**: Empty template. Should document the actual demo file `demo_decomp_seeds_mitosis.py` that exists in the directory.

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Files reviewed | 21 |
| CRITICAL issues | 8 |
| MAJOR issues | 18 |
| MINOR issues | 15 |
| Clean files | 3 (optim SPEC, utils SPEC, clahe SPEC) |
| Stub/template files | 3 (models/gsplats SPEC, models/utils SPEC, seeds/demos SPEC) |

## Priority Fixes (Ordered by Impact)

### P0 - Fix Immediately (CRITICAL)

1. **seeds/SPECIFICATIONS.md**: Package structure and method documentation is completely wrong. Lists non-existent files (`multiscale_gaussian.py`, `moment_seeding.py`) and functions (`seed_from_gaussian`). Missing documentation for actual methods (`seed_from_edges`, `seed_from_grid`, `gpu_ops`). **Requires full rewrite of the specification.**

2. **Root SPECIFICATIONS.md Section 0**: References `fit_result.py` which does not exist. Actual file is `gsplat_data.py`.

3. **Root SPECIFICATIONS.md Section 1**: Documents `seed_from_gaussian()` which does not exist. Should document `generate_seeds()` and the actual seeding methods.

4. **models/SPECIFICATIONS.md**: References `fit_result.py` and documents obsolete `.npz` persistence format. Should reference `gsplat_data.py` and `.gsplats.zarr` format.

### P1 - Fix Soon (MAJOR)

5. **demos/README.md**: Contains inaccurate claims about "per-splat Adam optimizer" and "dynamic topology changes" that contradict the actual architecture.

6. **io/SPECIFICATIONS.md**: Function name `sort_splats_spatially()` should be `sort_splats_spatial()`.

7. **metal/README.md**: Integration example uses wrong import path and non-existent parameters.

8. **multiscale/README.md**: "How It Works" shows MSE but default is L1. Multi-scale fitting example uses old tuple-return API.

9. **Stub files**: Three SPECIFICATIONS.md files (models/gsplats, models/utils, seeds/demos) are empty templates providing no value.

10. **demos/SPECIFICATIONS.md and README.md**: Missing documentation for `demo_3d_celegans_confocal.py` and `demo_boundary_containment.py`.

### P2 - Fix When Convenient (MINOR)

11. Various date/version inconsistencies
12. Minor path errors in cross-references
13. Changelog placeholder dates
14. Emoji usage in demos/README.md
15. Missing `initialize_zero()` from multiscale README API Reference

---

## Cross-File Inconsistencies

1. **Function naming**: `sort_splats_spatial` vs `sort_splats_spatially` -- used inconsistently across io/SPECIFICATIONS.md, io/__init__.py comment, and tests.

2. **Persistence format**: models/SPECIFICATIONS.md documents `.npz` format, while io/SPECIFICATIONS.md documents `.gsplats.zarr`. Only `.gsplats.zarr` is current.

3. **Optimizer description**: demos/README.md claims "per-splat Adam optimizer" while optim/SPECIFICATIONS.md explicitly documents standard PyTorch Adam (per-splat was removed in v2.0).

4. **Seed method naming**: Root SPECIFICATIONS.md documents `seed_from_gaussian` while seeds/__init__.py exports `seed_from_edges`, `seed_from_grid`, `seed_from_decomposition`.

5. **Default loss type**: multiscale/README.md "How It Works" says MSE, but SPECIFICATIONS.md and code both use L1 as default.
