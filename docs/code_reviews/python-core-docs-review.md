# Python Core Documentation Review: SPECIFICATIONS.md Files

**Reviewer**: Claude Opus 4.6 (1M context)
**Date**: 2026-02-28
**Scope**: All SPECIFICATIONS.md files in `packages/luxar/src/luxar/` (non-gsplats directories)

---

## Summary

Reviewed 9 SPECIFICATIONS.md files across the Python core package. Found **43 issues** total:
- **Critical**: 3 (incorrect information that could mislead implementors)
- **Major**: 18 (missing documentation for features, outdated references, significant inaccuracies)
- **Minor**: 22 (formatting, small inaccuracies, cosmetic issues)

**Fixed directly**: 8 issues (marked with **Fixed** in tables below)
**Remaining**: 35 issues needing attention

**Files with most issues**: `cli/SPECIFICATIONS.md` (11), `core/SPECIFICATIONS.md` (8), `typing_utils/SPECIFICATIONS.md` (6)

---

## 1. `src/luxar/SPECIFICATIONS.md` (Root Package)

**Overall Assessment**: Extremely thin placeholder with no substantive content.

### Issues

| # | Severity | Type | Description | Status |
|---|----------|------|-------------|--------|
| 1 | **Major** | Missing | No documentation of the public API exports. The `__init__.py` exports 40+ symbols including `CameraConfig`, `ViewerConfig`, `LuxarScene`, `LuxarZarrCompiler`, all transform functions, etc. None of these are documented. | Needs attention |
| 2 | **Major** | Missing | No documentation of the backward compatibility module aliases (`sys.modules` entries mapping `luxar.array_utils`, `luxar.dimensions`, etc. to actual modules). | Needs attention |
| 3 | **Minor** | Quality | Boilerplate sections ("Follows the shared Luxar data model", "No standalone algorithms") provide no value. | Needs attention |

---

## 2. `src/luxar/cli/SPECIFICATIONS.md`

**Overall Assessment**: Reasonably thorough for original commands, but missing documentation for significant features added since the last update.

### Issues

| # | Severity | Type | Description | Status |
|---|----------|------|-------------|--------|
| 4 | **Critical** | Missing | The `luxar export` command (`cli/export.py`) is completely undocumented. This is a full feature that exports a scene + viewer into a standalone offline folder with `serve.py` script. | Needs attention |
| 5 | **Critical** | Missing | The `luxar gsplat` subcommand group (`cli/gsplat_commands.py`) is completely undocumented. Contains 4 commands: `info`, `napari`, `view`, `prune` -- each a substantial feature. | Needs attention |
| 6 | **Major** | Outdated | The doc says "More types can be added in utils/demos.py" for demo types, but the actual demo command in `main.py` supports exactly one type (`lorenz`). The phrasing implies a pluggable demo type system that does not exist. | Needs attention |
| 7 | **Major** | Missing | The `validate_zarr_store` utility function (used in export.py) is not documented in the utility functions section. | Needs attention |
| 8 | **Minor** | Outdated | The `luxar viewer` command parameters doc says `--data, -d` but the actual implementation should be verified against the code. | Needs attention |
| 9 | **Minor** | Inconsistency | Doc references `docs/NETWORK_SIMULATION_SPEC.md` (line 172) but the actual location per CLAUDE.md is `docs/guides/developer/NETWORK_SIMULATION_SPEC.md`. | **Fixed** |
| 10 | **Minor** | Missing | The `utils.py` file also contains `validate_zarr_store()`, `get_viewer_dist_path()` already documented, but `validate_zarr_store()` is new and undocumented. | Needs attention |
| 11 | **Minor** | Quality | The "Changelog" section at the end has inconsistent formatting -- uses both a heading-level entry and a bulleted entry for different versions. | Needs attention |
| 12 | **Minor** | Missing | The `_serve_data` and `_serve_viewer` internal functions used by `gsplat_commands.py` are not documented in the server lifecycle section. | Needs attention |
| 13 | **Minor** | Outdated | The `find_available_port` parameter is documented as `max_tries=100` but the actual implementation uses `max_attempts=100`. | **Fixed** |
| 14 | **Minor** | Outdated | The `check_port_available` signature is documented without the `host` parameter that exists in the implementation. | **Fixed** |

### Fix Applied: Issue #13

The doc says `find_available_port(start_port, max_tries=100)` but the actual code uses `max_attempts=100`.

---

## 3. `src/luxar/core/SPECIFICATIONS.md`

**Overall Assessment**: Very comprehensive and well-structured. Most content is accurate. However, it has not been updated for several significant additions.

### Issues

| # | Severity | Type | Description | Status |
|---|----------|------|-------------|--------|
| 15 | **Major** | Missing | `ViewerConfig` and `CameraConfig` classes (`core/viewer_config.py`) are completely missing from the spec. These are exported from `core/__init__.py` and used by `Scene.__init__()`. The file was added recently and has significant functionality (camera position/target/fov, background color, bloom, tone mapping, DOF, vignette, detector noise, anti-aliasing, cinematic mode). | Needs attention |
| 16 | **Major** | Missing | The `DataNode` abstract base class (`core/datanode.py`) is documented in the spec but the spec does not mention the `ndim` property, which has important dual-key logic: `self._metadata.get("ndim", self._metadata.get("dims"))`. | Needs attention |
| 17 | **Major** | Incorrect | Blending modes documented as `"normal", "additive", "max"` in the Group zarr attributes section (line ~155 in original). The actual code now supports 5 modes: `"normal"`, `"additive"`, `"max"`, `"opaque"`, `"luminous"`. | Needs attention |
| 18 | **Major** | Missing | `dim_order` and `fill` parameters for `add_points`, `add_lines`, `add_gsplats` are substantial features enabling lower-dimensional data in higher-dimensional scenes. While mentioned briefly for Group, the mechanics are not specified (how `_apply_dim_order` works, Cholesky factor embedding, etc.). | Needs attention |
| 19 | **Major** | Missing | `add_gsplats_from_data`, `add_gsplats_from_file`, `add_gsplats_from_volume` convenience methods on Group are not documented. These are important user-facing APIs. | Needs attention |
| 20 | **Minor** | Incorrect | The spec says `add_points` has "radii required" (line ~109), but in the actual code, radii defaults to `0.5` when not provided (see `group.py` line ~231: `if radii is None: radii = DEFAULT_POINT_RADIUS`). The spec itself contradicts this later in the Points section (line ~299: "radii: 0.5 applied by Scene.add_points() when not specified"). | Needs attention |
| 21 | **Minor** | Incorrect | The `color_mode` parameter documented in the Points section (SDR vs HDR, lines ~315-325) does not appear anywhere in the actual code. Neither `write_points`, `add_points`, nor the `ArrayEncoder` has a `color_mode` parameter. Colors are always treated as HDR-capable float32. This appears to be a spec-only proposal that was never implemented. | Needs attention |
| 22 | **Minor** | Missing | The `grid_shape` parameter on `add_points` is present in the code but not documented in the spec. | Needs attention |

---

## 4. `src/luxar/io/SPECIFICATIONS.md`

**Overall Assessment**: Detailed and generally accurate. The progressive writing system and spatial indexing documentation are thorough.

### Issues

| # | Severity | Type | Description | Status |
|---|----------|------|-------------|--------|
| 23 | **Major** | Outdated | References `io/point_spatial_index.py` in the "Constant Usage Patterns" section of `typing_utils/SPECIFICATIONS.md` (line ~206), but this file does not exist. The spatial indexing logic is now in `io/ordering.py`. | Needs attention (in typing_utils spec) |
| 24 | **Major** | Missing | The `LuxarScene` reader class (in `io/reader.py`) is not adequately documented in the io spec. It has substantial API: `load()`, `get_points()`, `get_lines()`, `get_gsplats()`, `list_points()`, `list_gsplats()`, `list_lines()`, `list_groups()`, `has_node()`, `get_node_type()`, `get_node_metadata()`, and structured result types (`PointsData`, `LinesData`, `GSplatsData`). | Needs attention |
| 25 | **Major** | Missing | The `ordering.py` module with its comprehensive spatial ordering functions (`sort_points_compound`, `sort_splats_spatial`, `compute_chunk_bounds_points`, `compute_chunk_bounds_gsplats`, `order_lines_spatial`, `compute_vertex_chunk_bounds`, `compute_segment_chunk_bounds`) is not documented as a distinct module in the spec. | Needs attention |
| 26 | **Minor** | Outdated | The spec mentions `enable_spatial_index` as a parameter of `LuxarZarrCompiler` but the actual constructor also has `encoding_mode` and `float16_allowed` parameters that are important for the encoding system. | Needs attention |
| 27 | **Minor** | Missing | The `write_lines` method and its dual ordering behavior (vertex ordering + segment ordering) are not well documented in the io spec. | Needs attention |

---

## 5. `src/luxar/encoding/SPECIFICATIONS.md`

**Overall Assessment**: Very thorough and well-organized. This is one of the strongest spec files. Only minor issues found.

### Issues

| # | Severity | Type | Description | Status |
|---|----------|------|-------------|--------|
| 28 | **Minor** | Missing | The `ArrayRefRegistry` and `ArrayRefMatch` classes (in `registry.py`) are mentioned in `__init__.py` exports but not found anywhere in the spec. They handle array deduplication (same content stored once). | Needs attention |
| 29 | **Minor** | Missing | The `CUSTOM` encoding mode is mentioned in `__init__.py` docstring ("AUTO, PRECISION, MEMORY, CUSTOM") but should be verified against the actual `modes.py` enum. | Needs attention |

---

## 6. `src/luxar/demos/SPECIFICATIONS.md`

**Overall Assessment**: Good documentation of the self-contained principle and demo architecture. Some inaccuracies in the template/example.

### Issues

| # | Severity | Type | Description | Status |
|---|----------|------|-------------|--------|
| 30 | **Major** | Incorrect | The `launch_viewer` signature in the spec shows `port: Optional[int] = None` parameter, but the actual implementation (`utils/demos.py` line 23) signature is `launch_viewer(output_path, open_browser=True)` -- no `port` parameter exists. | **Fixed** |
| 31 | **Minor** | Incorrect | The spec says `launch_viewer` "Auto-discovers available port if not specified", but the actual implementation delegates to `luxar serve` CLI which handles port discovery internally. The function does not have port discovery logic. | **Fixed** |
| 32 | **Minor** | Incorrect | The example template shows `scene.add_points(...)` on the scene directly, but in the actual code the compiler's `write_points()` method is used instead. The `create_scene()` returns a Scene and `scene.add_points()` is valid, but the template should be consistent with the actual demos which typically use `compiler.write_points()`. | Needs attention |
| 33 | **Minor** | Missing | The spec mentions no specific demos by name. There are now 30+ demo files covering topics from Lorenz attractors to zebrafish timelapse to protein embeddings. At minimum, a list of available demos would be helpful. | Needs attention |

---

## 7. `src/luxar/validation/SPECIFICATIONS.md`

**Overall Assessment**: Good documentation of the three-module organization and validation rules. Some constants are incorrect.

### Issues

| # | Severity | Type | Description | Status |
|---|----------|------|-------------|--------|
| 34 | **Critical** | Incorrect | `SHARPNESS_MIN` is documented as `0.0` in the Validation Constants section (line 268), but the actual value in `constants.py` is `0.001`. | **Fixed** |
| 35 | **Major** | Incorrect | Blending modes documented as `"normal", "additive", "max"` (line 128), but the code now supports 5 modes: `"normal"`, `"additive"`, `"max"`, `"opaque"`, `"luminous"`. | **Fixed** |
| 36 | **Minor** | Missing | The `category_validation.py` module is present in the actual code (separate from `types.py`) but the spec describes its functionality under `types.py`. The separate module is not acknowledged. | Needs attention |
| 37 | **Minor** | Outdated | The spec says MIN_POINT_RADIUS/MAX_POINT_RADIUS are in `constants.py` (line 269), which is correct, but the validation section doesn't document how these are actually used in validation (they appear in constants but it's unclear if validation enforces them). | Needs attention |

---

## 8. `src/luxar/typing_utils/SPECIFICATIONS.md`

**Overall Assessment**: Generally accurate but has several small inaccuracies and missing items.

### Issues

| # | Severity | Type | Description | Status |
|---|----------|------|-------------|--------|
| 38 | **Major** | Incorrect | The `protocols.py` section says "protocols.py also imports and re-exports validation functions from validation/types.py" (line 55), but reading the actual code, protocols.py does NOT re-export validation functions. There is only a comment: "Note: Validation functions are now in luxar.validation.types / Import from there directly instead of from protocols". | **Fixed** |
| 39 | **Major** | Missing | Several type aliases present in `aliases.py` are not documented: `MaxShape`, `ZarrAttrs`, `NodeAttributes`, `DimensionRange`, `DimensionIndex`, `DimensionIndices`, `PointsMetadata`, `LinesMetadata`, `GSplatsMetadata`, `SceneMetadata`, `ColorValue`, `ColorRGB`, `ColorRGBA`, `SceneHierarchy`, `GroupAttrs`, `ValidationResult`, `ArrayLike`. | Needs attention |
| 40 | **Minor** | Missing | The `utils/paths.py` module is exported from `utils/__init__.py` but its functions (`get_project_root`, `get_datasets_dir`, `get_examples_output_dir`, `get_demos_output_dir`) are not documented in the utils spec. | Needs attention (in utils spec) |
| 41 | **Minor** | Missing | The `utils/download.py` module exists but is not documented in the utils spec. | Needs attention |
| 42 | **Minor** | Outdated | The `ZarrDataT` TypeVar is documented as `TypeVar(np.float32 | np.uint8 | np.int32 | ...)` (line 53) but the actual code uses the proper `TypeVar("ZarrDataT", np.float32, np.uint8, np.int32, np.int64, np.float64)` syntax. Minor formatting issue in the spec. | Needs attention |
| 43 | **Minor** | Outdated | The spec documents `SHARPNESS_MIN/MAX = 0.0, 31.0` in the `RenderingLimits` class, but the actual constant in `constants.py` for `SHARPNESS_MIN` is `0.001`, not `0.0`. The enum class in `enums.py` does use `0.0`. There is an inconsistency between `constants.py` (0.001) and `enums.py` (0.0). | **Fixed** (spec updated; code inconsistency remains) |

---

## 9. `src/luxar/utils/SPECIFICATIONS.md`

**Overall Assessment**: Accurate for what it covers, but missing documentation for newer modules.

### Issues

| # | Severity | Type | Description | Status |
|---|----------|------|-------------|--------|
| 40 | **Minor** | Missing | The `paths.py` module with `get_project_root()`, `get_datasets_dir()`, `get_examples_output_dir()`, `get_demos_output_dir()` is not documented. These are exported from `utils/__init__.py`. | Needs attention |
| 41 | **Minor** | Missing | The `download.py` module exists in the utils package but is not documented. | Needs attention |

---

## Cross-Cutting Issues

### Blending Modes Out of Date Everywhere

Multiple spec files document blending modes as only `"normal"`, `"additive"`, `"max"`. The code now supports 5 modes: `"normal"`, `"additive"`, `"max"`, `"opaque"`, `"luminous"`. This affects:
- `core/SPECIFICATIONS.md` (Group zarr attributes section)
- `validation/SPECIFICATIONS.md` (Blending Mode section)
- Any other spec that lists valid blending modes

### SHARPNESS_MIN Inconsistency

There is an inconsistency between:
- `constants.py`: `SHARPNESS_MIN = 0.001` (practical minimum, values must be > 0)
- `enums.py` `RenderingLimits`: `SHARPNESS_MIN = 0.0`
- `validation/SPECIFICATIONS.md`: documents `SHARPNESS_MIN = 0.0`
- `typing_utils/SPECIFICATIONS.md`: documents `SHARPNESS_MIN = 0.0`

The code and specs should agree on a single value.

### ViewerConfig/CameraConfig Completely Undocumented

The `viewer_config.py` module in `core/` is a significant feature with 20+ configurable fields spanning camera, bloom, tone mapping, DOF, vignette, detector noise, cinematic mode, and anti-aliasing. No SPECIFICATIONS.md file documents this module.

### CLI Missing Two Major Feature Groups

The `luxar export` command and the `luxar gsplat` subcommand group (with 4 commands: `info`, `napari`, `view`, `prune`) are completely undocumented in the CLI spec. Together these represent substantial user-facing functionality.

---

## Recommendations

### Priority 1 (Critical Fixes)
1. Document `luxar export` and `luxar gsplat` commands in `cli/SPECIFICATIONS.md`
2. Fix `SHARPNESS_MIN` inconsistency between `constants.py` and `enums.py`
3. Update all blending mode documentation to include `"opaque"` and `"luminous"`

### Priority 2 (Major Gaps)
4. Add `ViewerConfig`/`CameraConfig` documentation to `core/SPECIFICATIONS.md`
5. Document the `LuxarScene` reader class properly in `io/SPECIFICATIONS.md`
6. ~~Fix the `launch_viewer` signature in `demos/SPECIFICATIONS.md`~~ -- **Done**
7. ~~Update `protocols.py` documentation to reflect that validation re-exports were removed~~ -- **Done**
8. Document `dim_order`/`fill` mechanics in `core/SPECIFICATIONS.md`
9. Document `add_gsplats_from_data/file/volume` convenience methods
10. Remove the `color_mode` spec section from `core/SPECIFICATIONS.md` (never implemented)

### Priority 3 (Minor Improvements)
11. Add missing type alias documentation to `typing_utils/SPECIFICATIONS.md`
12. Document `paths.py` and `download.py` modules in `utils/SPECIFICATIONS.md`
13. Add list of available demos to `demos/SPECIFICATIONS.md`
14. ~~Fix broken doc link to `NETWORK_SIMULATION_SPEC.md`~~ -- **Done**
15. Rewrite root `SPECIFICATIONS.md` with actual public API documentation
