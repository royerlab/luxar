# Luxar TODO List

This file tracks known issues, planned features, and improvements for the Luxar project.

## Paper-Blocking (HIGH Priority)

11 - ~~**Screenshot export**~~: **DONE.** Press `G` for quick screenshot or `T` to open the Recording panel. Supports PNG/WebP/JPEG with quality slider, transparent background (alpha channel), and automatic max-DPR for highest resolution capture. Uses `canvas.toBlob()` with `preserveDrawingBuffer: false` safety (synchronous pixel read). Implemented in `packages/luxar-viewer/src/ui/recording-panel.ts`.

12 - ~~**Video export / animation recording**~~: **DONE.** Recording panel (`T` key) supports three modes: **Image** (screenshot), **Video** (canvas recording via `MediaRecorder` + `captureStream`), and **Turntable** (auto-rotate camera 360° then stop). Video mode includes: confirmation dialog, pulsing REC indicator with elapsed time, duration limit slider, FPS/codec selection, and **Sync to Slider** (record synced to a dimension animation — auto-stops at end). All browser-native, zero npm dependencies. Implemented in `packages/luxar-viewer/src/ui/recording-panel.ts` with 32 unit tests and 8 E2E tests.

13 - ~~**Scale bar overlay**~~: **DONE.** Press `B` to toggle a physical scale bar overlay. Computes bar width from camera distance and FOV at the orbit target depth, snaps to nice numbers (1/2/5 × 10^n), and displays the unit from dimension metadata (e.g., "10 μm"). Implemented in `packages/luxar-viewer/src/ui/components/scale-bar.ts`. Now pixel-exact in orthographic mode (TODO #6 completed).

14 - ~~**Colorbar / channel legend**~~: **DONE.** Press `J` to toggle the colormap legend overlay. Displays per-layer entries with layer name, colormap gradient bar (120×12px canvas from `BUILTIN_COLORMAPS`), and formatted min/max data range labels. Reactively updates via `LayerStateManager` subscription with hash-based rebuild optimization to avoid redundant DOM updates. Implemented in `packages/luxar-viewer/src/ui/components/colormap-legend.ts` with styles in `packages/luxar-viewer/src/styles/components/colormap-legend.css`.

16 - ~~**PSNR/SSIM quality metrics in CLI**~~: **DONE.** New `luxar gsplat compare fitted.gsplats.zarr original.tiff` command computes PSNR, SSIM, MSE, relative L2, and max absolute error. All metrics computed on GPU via PyTorch (SSIM uses `F.conv2d`/`F.conv3d`). Supports `--output-json` for paper tables and `--quiet` for scripting. Post-fit metrics (PSNR, SSIM, MSE) are now computed automatically after fitting and stored in the `.gsplats.zarr` metadata, visible via `luxar gsplat info`. Implemented in `gsplats/metrics.py` (PyTorch SSIM/PSNR), `cli/gsplat_commands.py` (compare command), and `fitting/results.py` (post-fit metrics).

18 - ~~**Tiled fitting for large volumes**~~: **DONE.** Fit arbitrarily large volumes by splitting into overlapping tiles with Hann cosine apodization (partition-of-unity windowing), fitting gsplats independently per tile, and concatenating. No post-merge pruning needed — the Hann window guarantees seamless blending. CLI: `luxar gsplat fit volume.tiff -o splats.gsplats.zarr --tiled --tile-size 256 --overlap 32` (all-in-one) or `luxar gsplat fit volume.tiff -o tile_3.gsplats.zarr --tile 3/16 --tile-size 256 --overlap 32` (single-tile, Slurm-ready). Python API: `fit_tiled(volume, tile_size=256, overlap=32)` and `fit_tile(volume, spec)`. Supports zarr lazy loading for out-of-core processing. Implemented in `gsplats/tiling.py` (tile geometry + cosine windows), `gsplats/fit_tiled_gsplats.py` (fitting orchestration), and CLI options on `fit_volume()`.

## Feature Requests (MEDIUM Priority)

5 - ~~**Layers panel (per-node controllability)**~~: **DONE.** Napari-inspired per-layer control panel. Nodes marked with `layer=True` in the Python API are exposed as controllable layers in the viewer (press `L` to toggle). Each layer provides: **visibility toggle** (eye icon), **display range** [min, max] mapped to shader intensity/offset uniforms, **gamma** correction, and **blending mode** (additive, normal, max, opaque, luminous). Material cloning ensures independent per-layer rendering. Python side: `layer` attribute on `Node` with validation. Viewer side: `layers-panel.ts` (panel UI), `layer-state.ts` (state management), `range-slider.ts` (custom dual-handle slider). Includes CSS styling for all themes, 217+ unit tests for layer state, and demo `demo_gsplats_3d_kidney_multichannel_layers.py`. Implemented in `packages/luxar-viewer/src/ui/layers/` and `packages/luxar/src/luxar/core/node.py`.

6 - ~~**Orthographic projection mode**~~: **DONE.** Toggle via rendering controls or keyboard shortcut. Uses `THREE.OrthographicCamera` with `LuxarCamera` type union (`camera-utils.ts`). In ortho mode: pan and zoom only, rotation restricted to view-axis roll (Shift+wheel). Custom `LuxarOrbitControls` (quaternion-based, no gimbal lock) handles both perspective and ortho modes. Scene-manager tracks ortho zoom level for material frustum updates. Clean front-view alignment for 2D microscopy data. Auto-framing computes correct camera distance from bounding-box diagonal. Implemented across `camera-utils.ts`, `luxar-orbit-controls.ts`, `scene-manager.ts`, and `controls-manager.ts`.

15 - ~~**Viewer-side colormaps**~~: **DONE.** Full colormap (CLUT) support for all geometry types. Python side: `colormap` attribute on nodes (e.g., `"green"`, `"magenta"`, `"fire"`, `"viridis"`) serialized to zarr. Viewer side: interactive per-layer colormap selection dropdown in the Layers panel (`L` key), with LUT applied via 256×1 RGB `DataTexture` in shaders. Includes 60+ built-in colormaps organized by category (sequential, diverging, cyclic, microscopy/BOP). Custom LUT data also supported. Colormap legend overlay (`J` key) shows active colormaps with gradient bars (see #14). Implemented in `packages/luxar/src/luxar/colormaps/` (Python), `packages/luxar-viewer/src/rendering/colormap-data.ts` and `colormap-textures.ts` (viewer LUT textures), and `packages/luxar-viewer/src/ui/layers/layers-panel.ts` (interactive selection).

20 - ~~**nD Transforms on non-displayed dimensions**~~: **DONE.** Per-dimension affine (scale/offset) and permutation transforms on non-displayed dimensions, separate from the 4x4 spatial transform. Enables time alignment, unit conversion, and channel remapping between datasets in the same scene. Uses **inverse-query** approach in the viewer: the query (slicePosition + tolerance) is inverse-transformed from world to local space once (O(1)), leaving all loader internals untouched. Hierarchical composition works on Python side (`world_nd_transform`); viewer reads composed transforms from scene graph (`computeWorldNdTransform`). Full stack: Python (validation, Node property, compiler, reader — 41 tests), TypeScript (types, inverse-query utility, scene-loader integration — 18 unit tests), E2E Playwright test (3 tests verifying time-shifted visibility). Spec: `docs/guides/specs/ND_TRANSFORMS_SPEC.md`. Demo: `demo_nd_transforms.py`. **Bounds expansion** wired into compiler `finalize()` — scene-level `position_bounds` now reflects world-space ranges for non-displayed dimensions. Viewer auto-ranges dimension sliders from these bounds when `Dimension.range` is not set. Documentation propagated to core, io, scene, and data READMEs.


17 - ~~**OME-Zarr (NGFF) input support**~~: **DONE.** `luxar gsplat fit` supports OME-Zarr with full 5D TCZYX handling via `--channel` and `--timepoint` flags. Auto-detects OME-Zarr layout (key `"0"` for highest resolution). Implemented in `gsplat_config.py:_load_zarr_volume()`.

19 - ~~**Slurm batch fitting for 3D+t OME-ZARR**~~: **DONE.** `luxar gsplat batch` CLI command converts entire 3D+t OME-ZARR datasets to splats via Slurm array jobs. Full implementation: `batch plan` (generates Slurm scripts with GPU profiling and time estimation), `batch status` (checks job completion), `batch merge` (combines fitted tiles/timepoints with channel color support). Includes environment capture (`env_capture.py`), manifest tracking (`manifest.py`), merge orchestration (`merge_orchestrator.py`), Slurm script generation (`slurm_gen.py`), and time estimation (`time_estimate.py`). Supports tiled fitting with `--tile-size`/`--overlap` flags. CLI: `luxar gsplat batch data.ome.zarr output/ --partition gpu --submit`. Implemented in `gsplats/batch/` module and `cli/gsplat_commands.py` with tests in `gsplats/tests/test_batch.py`.

21 - ~~**Transform model review & fixes**~~: **DONE.** Systematic review of the 4x4 transform pipeline (Python → zarr → TypeScript). Fixed: flat array ambiguity in `prepare_transform_for_zarr` (lists now row-major), `transform=None` persistence via `delete_group_attr` protocol method, bottom-row `[0,0,0,1]` affine validation, reader group transform support (`get_group()`), `world_transform` property. Added `nd_transform` support (see #20). All 1197 Python tests + 18 TS unit tests + 3 E2E tests passing.

## Infrastructure & Polish

4 - **Cache eviction policy**: Clarify and verify the cache eviction behavior — eviction does not appear to trigger when expected.

7 - **Theme layout consistency**: All Luxar UI themes should differ only in colors, transparency, and visual effects — never in the size or layout of panels and their components. This ensures a consistent user experience across themes.

8 - **Panel visibility configuration**: Allow configuring which panels are visible (Logs, Rendering Controls, Data Monitor, Dimensions, etc.) from the Python side. Optionally lock panel visibility to enforce a particular look and prevent user modifications.

9 - **UI ergonomics**: The current UI relies heavily on hidden keyboard shortcuts to reveal panels, which is poor discoverability. Improve with visible affordances (buttons, menus, or indicators).

0 - ~~**Retire gsplat sharpness from the viewer**~~: **DONE.** Sharpness attribute fully removed from GSplats across Python, TypeScript, and WASM. Commits `91be25c` (refactor: remove sharpness attribute from GSplats and update docs) and `f437c1f` (fix: remove sharpness refs from sorting and add edge case tests).

## Rendering & Performance (MEDIUM Priority)

22 - **Level-of-Detail (LOD) with SplitNode**: Two new composable scene graph node types for scalable rendering of large datasets:
    - **LODNode**: Contains children at different detail levels (e.g., full-res and merged/coarse gsplats). Selects active LOD based on **screen-space projected size** of the node's bounding box. Uses **per-splat opacity crossfade** during transitions — each splat's opacity is modulated by the LOD transition factor, avoiding brightness doubling from overlapping semi-transparent layers.
    - **SplitNode**: Spatially partitions a single logical node into sub-nodes (appears as one node to the user). Enables **frustum culling** per region — off-screen regions are not rendered at all. The writer decides the spatial split strategy (octree, axis-aligned tiles, irregular regions).
    - **Composition**: `SplitNode > LODNode > GSplats` gives per-region LOD selection through composition. Nearby regions render at full resolution, distant regions at coarse resolution, off-screen regions are culled entirely.
    - **Decimation**: Merge nearby splats — cluster and re-fit fewer, larger Gaussians. Done offline via CLI tooling (e.g., `luxar gsplat decimate`).
    - **Open questions**: nD LOD metric for non-displayed dimensions, split seam handling at LOD boundaries, optimal split granularity (8-64 regions), zarr format for LODNode/SplitNode metadata, LOD for lines (connectivity-preserving simplification), hysteresis thresholds.
    - **Advanced — Recursive composition**: SplitNode and LODNode can be alternated recursively (`SplitNode > LODNode > SplitNode > LODNode > ... > GSplats`) to form a hierarchical LOD tree (similar to 3D Tiles / Nanite). The viewer traverses the tree top-down: at each LODNode, if the coarse child is sufficient (screen-space error below threshold), it renders the coarse summary and **stops traversal** — never loading or rendering the finer splits below. This gives: (1) view-adaptive memory usage — only fine-grained data for nearby regions is loaded, (2) adaptive draw call count — zoomed-out views render few coarse nodes, close-ups render many fine nodes, (3) natural progressive streaming — coarse LODs load first, fine LODs on demand. Key requirement: each LODNode's coarse child must be a faithful summary of the entire subtree below it, not just one level down.
    - **Advanced — Lessons from Nanite (UE5)**:
        - **Monotonic error guarantee**: Each LODNode must store its simplification error (e.g., max amplitude difference, spatial displacement vs. fine level). The hierarchy must guarantee `parent_error >= child_error` at every level so traversal always converges. Without this, a coarse LOD might look "sufficient" while hiding a region where it's actually terrible.
        - **Pixel-error metric, not just size**: LOD selection should be based on "how many pixels of screen-space error would this simplification introduce" — not just projected bounding box size. A flat region with 1M splats may have near-zero simplification error (coarse is fine), while a detailed region at the same screen size may need fine LOD. Store the error, project it to pixels at runtime.
        - **Density-adaptive splitting**: SplitNode should partition based on splat density / detail, not a uniform grid. Dense regions get more splits, empty regions fewer (k-d tree at median, or cluster by density). Avoids wasting splits on empty space.
        - **Virtual residency via zarr chunks**: Zarr's chunked storage maps naturally to Nanite's virtual memory model — coarse LOD chunks (small) stay resident, fine LOD chunks are fetched on demand and evicted when the camera moves away.
        - **Blending may be minimal**: Nanite avoids blending entirely via seamless DAG cuts (mesh-specific, not transferable). However, overlapping Gaussians at region boundaries provide natural continuity — test whether per-splat crossfade can use a very narrow transition or be skipped entirely for well-constructed LODs.

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