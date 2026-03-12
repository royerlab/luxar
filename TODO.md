# Luxar TODO List

This file tracks known issues, planned features, and improvements for the Luxar project.

## Paper-Blocking (HIGH Priority)

11 - ~~**Screenshot export**~~: **DONE.** Press `G` for quick screenshot or `T` to open the Recording panel. Supports PNG/WebP/JPEG with quality slider, transparent background (alpha channel), and automatic max-DPR for highest resolution capture. Uses `canvas.toBlob()` with `preserveDrawingBuffer: false` safety (synchronous pixel read). Implemented in `packages/luxar-viewer/src/ui/recording-panel.ts`.

12 - ~~**Video export / animation recording**~~: **DONE.** Recording panel (`T` key) supports three modes: **Image** (screenshot), **Video** (canvas recording via `MediaRecorder` + `captureStream`), and **Turntable** (auto-rotate camera 360° then stop). Video mode includes: confirmation dialog, pulsing REC indicator with elapsed time, duration limit slider, FPS/codec selection, and **Sync to Slider** (record synced to a dimension animation — auto-stops at end). All browser-native, zero npm dependencies. Implemented in `packages/luxar-viewer/src/ui/recording-panel.ts` with 32 unit tests and 8 E2E tests.

13 - ~~**Scale bar overlay**~~: **DONE.** Press `B` to toggle a physical scale bar overlay. Computes bar width from camera distance and FOV at the orbit target depth, snaps to nice numbers (1/2/5 × 10^n), and displays the unit from dimension metadata (e.g., "10 μm"). Implemented in `packages/luxar-viewer/src/ui/components/scale-bar.ts`. Will become pixel-exact when orthographic mode (TODO #6) is added.

14 - **Colorbar / channel legend**: Display a legend overlay showing each node's name and color swatch. Required for multi-channel figures and general usability.

16 - ~~**PSNR/SSIM quality metrics in CLI**~~: **DONE.** New `luxar gsplat compare fitted.gsplats.zarr original.tiff` command computes PSNR, SSIM, MSE, relative L2, and max absolute error. All metrics computed on GPU via PyTorch (SSIM uses `F.conv2d`/`F.conv3d`). Supports `--output-json` for paper tables and `--quiet` for scripting. Post-fit metrics (PSNR, SSIM, MSE) are now computed automatically after fitting and stored in the `.gsplats.zarr` metadata, visible via `luxar gsplat info`. Implemented in `gsplats/metrics.py` (PyTorch SSIM/PSNR), `cli/gsplat_commands.py` (compare command), and `fitting/results.py` (post-fit metrics).

18 - ~~**Tiled fitting for large volumes**~~: **DONE.** Fit arbitrarily large volumes by splitting into overlapping tiles with Hann cosine apodization (partition-of-unity windowing), fitting gsplats independently per tile, and concatenating. No post-merge pruning needed — the Hann window guarantees seamless blending. CLI: `luxar gsplat fit volume.tiff -o splats.gsplats.zarr --tiled --tile-size 256 --overlap 32` (all-in-one) or `luxar gsplat fit volume.tiff -o tile_3.gsplats.zarr --tile 3/16 --tile-size 256 --overlap 32` (single-tile, Slurm-ready). Python API: `fit_tiled(volume, tile_size=256, overlap=32)` and `fit_tile(volume, spec)`. Supports zarr lazy loading for out-of-core processing. Implemented in `gsplats/tiling.py` (tile geometry + cosine windows), `gsplats/fit_tiled_gsplats.py` (fitting orchestration), and CLI options on `fit_volume()`.

## Feature Requests (MEDIUM Priority)

5 - **Layers panel (per-node controllability)**: Expose all scene graph nodes (Groups, Points, Lines, GSplats — except the Scene root) as "layers" in a dedicated UI panel. Each layer provides controls for visibility, brightness, offset, and gamma, applied to the node and its entire sub-tree. Most of the viewer machinery is already in place; the remaining work is the Layers panel itself. Inspired by napari: multi-select layers, show only the intersection of available controls, and propagate setting changes to all selected layers.

6 - **Orthographic projection mode**: Add an orthographic camera option. In this mode the user can pan and zoom and only rotate around the view direction. Standard expectation for microscopy viewers of 2D data.

15 - **Viewer-side colormaps**: Support a `colormap` attribute on nodes (e.g., `"green"`, `"magenta"`, `"fire"`, `"viridis"`) with LUT application on the viewer side. Currently all colors must be pre-baked in Python. The microscopy convention is to apply lookup tables at display time, enabling users to change coloring interactively. Right now colors are 'baked' in the scene nodes. but we can add explicit support for LUTs, definitely a gap in the current implementation. There is machinery in the encoding step to handle LUTs but it is only internal to the encoding. Making LUTs explicit makes it possible to implement the corresponding legends. This will probably require to first implement the scene 'domains' and in particular the 'overlay domain' to render the legends in normalized screen space.

20 - ~~**nD Transforms on non-displayed dimensions**~~: **DONE.** Per-dimension affine (scale/offset) and permutation transforms on non-displayed dimensions, separate from the 4x4 spatial transform. Enables time alignment, unit conversion, and channel remapping between datasets in the same scene. Uses **inverse-query** approach in the viewer: the query (slicePosition + tolerance) is inverse-transformed from world to local space once (O(1)), leaving all loader internals untouched. Hierarchical composition works on Python side (`world_nd_transform`); viewer reads composed transforms from scene graph (`computeWorldNdTransform`). Full stack: Python (validation, Node property, compiler, reader — 41 tests), TypeScript (types, inverse-query utility, scene-loader integration — 18 unit tests), E2E Playwright test (3 tests verifying time-shifted visibility). Spec: `docs/guides/specs/ND_TRANSFORMS_SPEC.md`. Demo: `demo_nd_transforms.py`.
    - **Remaining work**: Bounds expansion (store world-space bounds in zarr for auto-ranging sliders), documentation propagation to all README/SPECIFICATIONS files.


17 - ~~**OME-Zarr (NGFF) input support**~~: **DONE.** `luxar gsplat fit` supports OME-Zarr with full 5D TCZYX handling via `--channel` and `--timepoint` flags. Auto-detects OME-Zarr layout (key `"0"` for highest resolution). Implemented in `gsplat_config.py:_load_zarr_volume()`.

19 - **Slurm batch fitting for 3D+t OME-ZARR**: Add `luxar gsplat batch` CLI command to convert entire 3D+t OME-ZARR datasets to splats via Slurm array jobs. Thin wrapper approach (not a full orchestrator): auto-discover T/C dimensions from OME-ZARR metadata, generate and submit `sbatch --array=0-{T-1}` where each task runs `luxar gsplat fit --timepoint $SLURM_ARRAY_TASK_ID`, then merge results with `combine_as_new_dimension()`. Components: (1) OME-ZARR shape discovery from `.zattrs` (~50 lines in `gsplat_config.py`), (2) `batch` command with Slurm script generation + submission (~200-300 lines), (3) `batch status` subcommand to check completion (~80 lines), (4) optional `--auto-merge` that submits a dependent job (`--dependency=afterok`) to run the merge automatically. All per-timepoint fitting and 4D merging primitives already exist. CLI: `luxar gsplat batch input.ome.zarr -o output_dir/ --preset standard --partition gpu --time 2:00:00 --gpus-per-node 1`. Total estimated scope: ~600-700 lines including tests.

21 - ~~**Transform model review & fixes**~~: **DONE.** Systematic review of the 4x4 transform pipeline (Python → zarr → TypeScript). Fixed: flat array ambiguity in `prepare_transform_for_zarr` (lists now row-major), `transform=None` persistence via `delete_group_attr` protocol method, bottom-row `[0,0,0,1]` affine validation, reader group transform support (`get_group()`), `world_transform` property. Added `nd_transform` support (see #20). All 1197 Python tests + 18 TS unit tests + 3 E2E tests passing.

## Infrastructure & Polish

4 - **Cache eviction policy**: Clarify and verify the cache eviction behavior — eviction does not appear to trigger when expected.

7 - **Theme layout consistency**: All Luxar UI themes should differ only in colors, transparency, and visual effects — never in the size or layout of panels and their components. This ensures a consistent user experience across themes.

8 - **Panel visibility configuration**: Allow configuring which panels are visible (Logs, Rendering Controls, Data Monitor, Dimensions, etc.) from the Python side. Optionally lock panel visibility to enforce a particular look and prevent user modifications.

9 - **UI ergonomics**: The current UI relies heavily on hidden keyboard shortcuts to reveal panels, which is poor discoverability. Improve with visible affordances (buttons, menus, or indicators).

10 - **Retire sharpness from the viewer**: Remove or hide the sharpness parameter from the viewer UI. Keep it in the data model and rendering pipeline, but do not expose it as a user-facing control.

## Future / Exploratory (LOW Priority)

1 - **Ray casting with object labels**: Associate descriptive strings with scene objects. When the user picks an object via ray casting, display the associated label at a fixed screen position. Useful for providing context during exploration.

2 - **Scene domains**: Introduce the concept of rendering "domains" beyond the main nD-to-3D slice:
    - **Overlay domain**: For a given set of non-visible dimensions, render an associated scene as a transparent overlay in normalized canvas coordinates ([0,1] x [0,1]), unaffected by camera controls.
    - **Sound domain**: Associate audio with a scene, played back on load to provide auditory context.

3 - **VR/AR mode**: Add the ability to activate VR/AR rendering for immersive exploration of 3D scenes.

## Notes

- Review and update this list regularly.
- Items marked HIGH priority should be addressed before the first preprint.
- Consider creating GitHub issues for tracking progress on individual items.
- Update CLAUDE.md when implementing significant changes.