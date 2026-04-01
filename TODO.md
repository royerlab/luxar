# Luxar TODO List

This file tracks known issues, planned features, and improvements for the Luxar project.

## Infrastructure & Polish

4 - **Cache eviction policy**: Clarify and verify the cache eviction behavior — eviction does not appear to trigger when expected.

7 - **Theme layout consistency**: All Luxar UI themes should differ only in colors, transparency, and visual effects — never in the size or layout of panels and their components. This ensures a consistent user experience across themes.

8 - **Panel visibility configuration**: Allow configuring which panels are visible (Logs, Rendering Controls, Data Monitor, Dimensions, etc.) from the Python side. Optionally lock panel visibility to enforce a particular look and prevent user modifications.

9 - **UI ergonomics**: The current UI relies heavily on hidden keyboard shortcuts to reveal panels, which is poor discoverability. Improve with visible affordances (buttons, menus, or indicators).

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
