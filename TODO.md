# Luxar TODO List

This file tracks known issues, planned features, and improvements for the Luxar project.

## Paper-Blocking (HIGH Priority)

11 - **Screenshot export**: Add a keyboard shortcut (e.g., `S`) to capture the current viewer canvas as a PNG and trigger a download. Essential for generating paper figures.

12 - **Video export / animation recording**: Record turntable or flythrough animations as frame sequences or video. Auto-rotate already works; the missing piece is exporting frames (e.g., PNG sequence for ffmpeg). Critical for supplementary materials. Here is a good reference: https://webrtc.github.io/samples/src/content/capture/canvas-record/

13 - **Scale bar overlay**: Render a physical scale bar in the viewer using the `unit` metadata from Dimensions. Every microscopy figure requires one. Implementation: a simple bar with a label rendered as a screen-space overlay.

14 - **Colorbar / channel legend**: Display a legend overlay showing each node's name and color swatch. Required for multi-channel figures and general usability.

16 - **PSNR/SSIM quality metrics in CLI**: Add `luxar gsplat info --compare <original.tiff/npy>` to compute PSNR, SSIM, and other quality metrics against the original volume. Currently only `final_rel_l2` and `final_max_abs_error` are tracked during fitting. This would make quality comparison tables for papers trivial. Compute round-trip PSNR/SSIM after fitting, controlled by parameter, on by default. 

18 - **Tiled fitting for large volumes**: Fit arbitrarily large volumes by splitting into overlapping 3D tiles with cosine apodization (raised cosine / Hann window), fitting gsplats independently per tile, and concatenating all splats. The cosine window ensures smooth amplitude tapering at tile boundaries, eliminating visible seams. Optional post-merge pruning removes redundant splats in overlap zones. This removes the GPU memory ceiling and enables parallelism (multi-GPU, cluster). Critical for real-world microscopy data (e.g., 2048x2048x500 light-sheet stacks). CLI: `luxar gsplat fit volume.tiff -o splats.gsplats.zarr --tiled --tile-size 256 --overlap 32`. best to implement this after slurm integration so that tiles can be processed across GPUs.

## Feature Requests (MEDIUM Priority)

5 - **Layers panel (per-node controllability)**: Expose all scene graph nodes (Groups, Points, Lines, GSplats — except the Scene root) as "layers" in a dedicated UI panel. Each layer provides controls for visibility, brightness, offset, and gamma, applied to the node and its entire sub-tree. Most of the viewer machinery is already in place; the remaining work is the Layers panel itself. Inspired by napari: multi-select layers, show only the intersection of available controls, and propagate setting changes to all selected layers.

6 - **Orthographic projection mode**: Add an orthographic camera option. In this mode the user can pan and zoom and only rotate around the view direction. Standard expectation for microscopy viewers of 2D data.

15 - **Viewer-side colormaps**: Support a `colormap` attribute on nodes (e.g., `"green"`, `"magenta"`, `"fire"`, `"viridis"`) with LUT application on the viewer side. Currently all colors must be pre-baked in Python. The microscopy convention is to apply lookup tables at display time, enabling users to change coloring interactively. Right now colors are 'baked' in the scene nodes. but we can add explicit support for LUTs, definitely a gap in the current implementation. There is machinery in the encoding step to handle LUTs but it is only internal to the encoding. Making LUTs explicit makes it possible to implement the corresponding legends. This will probably require to first implement the scene 'domains' and in particular the 'overlay domain' to render the legends in normalized screen space.


17 - ~~**OME-Zarr (NGFF) input support**~~: **DONE.** `luxar gsplat fit` supports OME-Zarr with full 5D TCZYX handling via `--channel` and `--timepoint` flags. Auto-detects OME-Zarr layout (key `"0"` for highest resolution). Implemented in `gsplat_config.py:_load_zarr_volume()`.

19 - **Slurm batch fitting for 3D+t OME-ZARR**: Add `luxar gsplat batch` CLI command to convert entire 3D+t OME-ZARR datasets to splats via Slurm array jobs. Thin wrapper approach (not a full orchestrator): auto-discover T/C dimensions from OME-ZARR metadata, generate and submit `sbatch --array=0-{T-1}` where each task runs `luxar gsplat fit --timepoint $SLURM_ARRAY_TASK_ID`, then merge results with `combine_as_new_dimension()`. Components: (1) OME-ZARR shape discovery from `.zattrs` (~50 lines in `gsplat_config.py`), (2) `batch` command with Slurm script generation + submission (~200-300 lines), (3) `batch status` subcommand to check completion (~80 lines), (4) optional `--auto-merge` that submits a dependent job (`--dependency=afterok`) to run the merge automatically. All per-timepoint fitting and 4D merging primitives already exist. CLI: `luxar gsplat batch input.ome.zarr -o output_dir/ --preset standard --partition gpu --time 2:00:00 --gpus-per-node 1`. Total estimated scope: ~600-700 lines including tests.

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