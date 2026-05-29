# Luxar TODO List

This file tracks known issues, planned features, and improvements for the Luxar project.

## Infrastructure & Polish

4 - **Cache eviction policy**: Clarify and verify the cache eviction behavior — eviction does not appear to trigger when expected.

7 - **Theme layout consistency**: All Luxar UI themes should differ only in colors, transparency, and visual effects — never in the size or layout of panels and their components. This ensures a consistent user experience across themes.

8 - **Panel visibility configuration**: Allow configuring which panels are visible (Logs, Rendering Controls, Data Monitor, Dimensions, etc.) from the Python side. Optionally lock panel visibility to enforce a particular look and prevent user modifications.

9 - **UI ergonomics**: The current UI relies heavily on hidden keyboard shortcuts to reveal panels, which is poor discoverability. Improve with visible affordances (buttons, menus, or indicators).

## Bugs

23 - **Fix bugs surfaced by examples**: Several example datasets surface viewer/loader bugs that need fixing:
    - `scalars_and_colormap_example.zarr`
    - `scene_dimensions_example.zarr`
    - `transform_example.zarr`

## Rendering & Performance (MEDIUM Priority)

22 - **Level-of-Detail (LOD) with SplitNode** — core landed, advanced refinements remain.

    **Implemented (see Completed archive, item 22-core):** LODNode (`kind:'lod'`, pixel-size selector + hysteresis), SplitNode (`kind:'split'`, midpoint + SAH BSP, auto-split heuristic), recursive scene-graph composition, `luxar gsplat lod` (additive / substitutive / pyramid), Poisson-disk + spatial-uniform LOD ordering, LOD for lines (connectivity-preserving), progressive multi-additive-LOD loading for Points & Lines, zarr v2.0 (substitutive × additive matrix) format.

    **Still TODO:**
    - **Per-splat opacity crossfade** during LOD transitions — each splat's opacity modulated by the transition factor to avoid brightness doubling from overlapping semi-transparent layers. (Currently transitions are hard switches with hysteresis; no crossfade.)
    - **Pixel-error metric, not just projected size**: LOD selection should be based on "how many pixels of screen-space error would this simplification introduce" — not just projected bounding box size. A flat region with 1M splats may have near-zero simplification error (coarse is fine), while a detailed region at the same screen size may need fine LOD. Store per-level error, project it to pixels at runtime. (Currently selector is pure `pixel_size`.)
    - **Monotonic error guarantee**: Each LODNode should store its simplification error (max amplitude difference, spatial displacement vs. fine level) with `parent_error >= child_error` guaranteed at every level so top-down traversal always converges. (Greedy additive ordering has a `(1-1/e)` submodular guarantee but no stored monotonic error bound.)
    - **Density-adaptive splitting**: SplitNode should partition based on splat density / detail, not a uniform/midpoint grid. Dense regions get more splits, empty regions fewer (k-d tree at median, or cluster by density). SAH BSP is a step toward this but is cost-driven, not density-driven.
    - **Nanite-style stop-traversal**: top-down traversal that renders a coarse summary and **stops** (never loading finer splits below) when screen-space error is below threshold, for view-adaptive memory + draw-call counts and progressive streaming. (Currently progressive loaders stream additive LODs but there is no error-driven subtree pruning.)
    - **nD LOD metric for non-displayed dimensions**, split seam handling at LOD boundaries, optimal split granularity tuning.
    - **Virtual residency via zarr chunks**: coarse LOD chunks stay resident, fine LOD chunks fetched on demand and evicted when the camera moves away (Nanite-style virtual memory model).

## Future / Exploratory (LOW Priority)

1 - **Ray casting with object labels**: Associate descriptive strings with scene objects. When the user picks an object via ray casting, display the associated label at a fixed screen position. Useful for providing context during exploration.

2 - **Scene domains**: Introduce the concept of rendering "domains" beyond the main nD-to-3D slice:
    - **Overlay domain**: For a given set of non-visible dimensions, render an associated scene as a transparent overlay in normalized canvas coordinates ([0,1] x [0,1]), unaffected by camera controls.
    - **Sound domain**: Associate audio with a scene, played back on load to provide auditory context.

3 - **VR/AR mode**: Add the ability to activate VR/AR rendering for immersive exploration of 3D scenes.

---

## Completed (Archive)

<details>
<summary>Click to expand completed items</summary>

### Paper-Blocking (completed)

11 - ~~**Screenshot export**~~: **DONE.** Press `G` for quick screenshot or `T` to open the Recording panel. Supports PNG/WebP/JPEG with quality slider, transparent background (alpha channel), and automatic max-DPR for highest resolution capture.

12 - ~~**Video export / animation recording**~~: **DONE.** Recording panel (`T` key) supports three modes: Image, Video, and Turntable.

13 - ~~**Scale bar overlay**~~: **DONE.** Press `B` to toggle a physical scale bar overlay.

14 - ~~**Colorbar / channel legend**~~: **DONE.** Press `J` to toggle the colormap legend overlay.

16 - ~~**PSNR/SSIM quality metrics in CLI**~~: **DONE.** `luxar gsplat compare` command.

18 - ~~**Tiled fitting for large volumes**~~: **DONE.** `luxar gsplat fit --tiled`.

### Rendering & Performance (completed)

22-core - ~~**LOD + SplitNode core**~~: **DONE.** Composable scene graph LOD landed across Python and viewer:
    - **LODNode** (`kind:'lod'`): screen-space pixel-size selector with asymmetric hysteresis — `lod-group-registry.ts`, `types/lod-group.ts`.
    - **SplitNode** (`kind:'split'`): spatial BSP partitioning (recursive midpoint + opt-in SAH BSP) with per-mesh frustum culling and an auto-split heuristic — `core/group/split.py`, `core/group/auto_split.py`, `data/scene-loader/nodes/load-split-group-node.ts`.
    - **Recursive composition**: arbitrary nesting of `lod`/`split` groups via standard scene-graph loading.
    - **Decimation CLI**: `luxar gsplat lod additive | substitutive | pyramid` (greedy / self_energy / mass / kmeans_lloyd, energy/count breakpoints) — `cli/gsplat_commands.py`.
    - **LOD ordering**: Poisson-disk (Bridson) + spatial-uniform — `core/group/lod/poisson_disk.py`.
    - **LOD for lines**: polyline-aware, connectivity-preserving simplification — `core/group/lod/lines.py`.
    - **Progressive multi-additive-LOD loading** for Points & Lines — `data/points/points-progressive-loader.ts`, `data/lines/lines-progressive-loader.ts`.
    - **Zarr v2.0 format**: substitutive × additive LOD matrix — see `docs/specs/GSPLATS_ZARR_FORMAT.md`.

### Feature Requests (completed)

5 - ~~**Layers panel**~~: **DONE.** Press `L` to toggle per-layer control panel.

6 - ~~**Orthographic projection mode**~~: **DONE.** Toggle via rendering controls or keyboard shortcut.

15 - ~~**Viewer-side colormaps**~~: **DONE.** Full colormap (CLUT) support for all geometry types.

17 - ~~**OME-Zarr (NGFF) input support**~~: **DONE.** `luxar gsplat fit` supports OME-Zarr.

19 - ~~**Slurm batch fitting**~~: **DONE.** `luxar gsplat batch` CLI command.

20 - ~~**nD Transforms**~~: **DONE.** Per-dimension affine and permutation transforms.

21 - ~~**Transform model review & fixes**~~: **DONE.**

0 - ~~**Retire gsplat sharpness from the viewer**~~: **DONE.**

</details>

## Notes

- Review and update this list regularly.
- Consider creating GitHub issues for tracking progress on individual items.
- Update CLAUDE.md when implementing significant changes.
