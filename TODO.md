# Luxar TODO List

This file tracks known issues, planned features, and improvements for the Luxar project.

---

## Release planning

The 813-line "Path to Release & Announcement" section that used to live here
has been retired. It had drifted badly — its Zenodo status was several weeks
stale, its gallery plan no longer matched the residual curation work, and it
predated the demo-hosting stack entirely, so it never mentioned the two live
sites that are now the project's most visible artifacts.

Release planning is tracked outside this repository. It names unpublished
record identifiers, machine-specific paths and an in-progress collaborator
conversation, none of which belong in a repository that is about to be public.
Ask before assuming a release item is unowned.

Two things that lived in the retired section have durable homes now:

- The 2026-07-12 `--floor` decision — why the manuscript benchmarks pin
  `--floor none` while the shipped CLI default stays `--floor auto` — moved to
  [`docs/guides/developer/BENCHMARK_FLOOR_DECISION.md`](docs/guides/developer/BENCHMARK_FLOOR_DECISION.md),
  which is where its three in-tree citations now point.
- The demo-hosting architecture and its hazards are documented in
  [`docs/guides/developer/DEMO_SITE_RUNBOOK.md`](docs/guides/developer/DEMO_SITE_RUNBOOK.md).

Everything below this line is ordinary engineering backlog and is still live.

Legacy tags retained by code and documentation mean:

- **R10:** depth sorting; **R10a:** normal-mode gsplat demos.
- **R17:** retire git-LFS-heavy demo data to Zenodo — (1) fetch helper and
  manifest, (2) upload and publish records, (3) repoint demos, (4) `git rm` the
  payload.
- **R19:** README and gallery refresh.

---

## Infrastructure & Polish

4 - ~~**Cache eviction policy**~~: **DONE.** Root cause found & fixed: L2 (OPFS) eviction was gated only on the configured `maxSize` (default `l2MaxSizeMB: 2048` → 2 GB), but the browser-granted OPFS quota is often far smaller. The quota gate in `OPFSStore.doSet` rejected writes (counted as `quotaWriteSkipped`) long before `totalSize` reached 2 GB, so the maxSize-based eviction loop never ran — the LRU froze holding old entries and silently dropped new ones (worst on Firefox/private-mode/small disks; invisible on roomy Chrome). Fix: `doSet` now evicts LRU entries on quota pressure (not just maxSize pressure) and re-checks, since deleting files genuinely frees quota. Bounded by index size with a no-progress guard. Regression test: `tests/unit/cache/opfs-eviction-quota.todo4.test.ts`. (L0/L1 size-gated eviction was already correct.)

7 - **Theme layout consistency**: All Luxar UI themes should differ only in colors, transparency, and visual effects — never in the size or layout of panels and their components. This ensures a consistent user experience across themes. *Update 2026-08-11:* the authoritative **UI Design Guide** now exists (`docs/guides/developer/UI_DESIGN_GUIDE.md`, #1474) and codifies exactly this (plus highlight-as-interactive-accent, green-as-semantic); enforcement work should cite it. Known theme defects to fold in: glass secondary/muted text below WCAG AA (#1513), white-on-white context menu in liquid-glass (#1510), two §5.1 glass-surface violations (#1483).

32 - **UI modernization arc — interaction depth + follow-up drain** (2026-08). A large restyle wave landed on the "quiet instrument" language: UI Design Guide (#1474), accent migration + panel polish (#1480), Select Dataset dialog modernization (#1472), emoji → stroke icons + neutral scene-graph names (#1479), a11y hardening (embed-safe focus rings, reduced motion, #1476), token hygiene (#1478). **Open:** PR #1508 (panel entry motion, modal focus trap, type-to-filter, Layers context menus) plus its filed follow-ups — submenu unmount on hover (#1509), Layers rows all hidden after dataset switch (#1512), listbox keyboard tab stop (#1511) — and the theme defects listed under item 7. Also health: the control-rail light-theme snapshot is red on `main` (#1493).

8 - **Panel visibility configuration**: Allow configuring which panels are visible (Logs, Rendering Controls, Data Monitor, Dimensions, etc.) from the Python side. Optionally lock panel visibility to enforce a particular look and prevent user modifications.

9 - ~~**UI ergonomics**~~: **DONE** (#432). Always-visible left **control rail** — one icon per panel (Help/Dimensions/Rendering/Layers/Data monitor/Datasets/Recording/Screenshot/Logs/View options/Performance), each firing the same command as its shortcut, with tooltips, event-driven active-state, collapse, idle-dim, first-run hint, and full theme integration. Next slice: panels dock into a tray beside the rail (Concept A step 2).

## Bugs

27 - ~~**Demos can't be stopped with Ctrl-C; a new demo shows the old one**~~: **DONE** (2026-07-24, #652). `luxar demo run` spawned a 3-level tree (`demo run` → demo script → `luxar serve` uvicorn) with no process-group isolation or owned teardown, so Ctrl-C orphaned the server on ports 8000/5173; `pick_port` then auto-incremented and the stale browser tab kept showing the old scene. Fix: the stdlib-only `luxar/_process.py::run_child_process` runs the child in its own session (`start_new_session`) and, on any exit, tears the whole subtree down with escalating SIGINT → SIGTERM → SIGKILL in a `finally` (SIGTERM/SIGHUP routed in too; a second Ctrl-C jumps straight to SIGKILL). Wired into `demo_run`/`demo_run_all` (isolate the group) and `launch_viewer` (stay in the group so the group-kill cascades). Verified with a real foreground Ctrl-C via a PTY: exits 130, zero survivors, ports freed. Also folded in demo-CLI robustness (installed-wheel guard, clean `DEMO_META` errors, run-all GPU/large-download skips + `--include-gpu`/`--max-download-mb`, corrupt-download classification, honest cache-clear totals, no `datasets/` dir creation on `demo list`). **Completed by the stale-demo-tab campaign (2026-08, #1462–#1467):** `luxar demo stop` (#1462), per-demo derived port pairs instead of shared 8000/5173 (#1463), a scene-identity watchdog so a tab that no longer shows what its address serves says so (#1466), and browser tabs named after the scene they show (#1467). The "new demo shows the old one" failure class is now defended at every layer.

23 - **Fix bugs surfaced by examples** (reproduced & triaged 2026-06-30):
    - ✅ `scene_dimensions_example` — **FIXED.** `[`/`]` navigation emptied the
      view because `step` (time 0.5 s, z 0.1 µm) was finer than the data
      sampling (time every 2.5 s, z every 20 µm). Aligned the steps to the
      data so every keypress lands on a populated slice.
    - ✅ `transform_example` — **no bug.** All cubes/axes + parent-child
      hierarchy load and place correctly; could not reproduce a defect
      (already fixed upstream).
    - ✅ `scalars_and_colormap_example` — **FIXED** (#430). The three spirals
      washed out to near-identical white because points default to
      `blending_mode="additive"`, which *sums* the self-overlapping turns
      toward white. This is order-independent (additive sum is commutative),
      so it was NOT a #24 depth-sorting bug. Switched the demo to
      `blending_mode="max"` (brightest-wins, order-independent) so each
      colormap reads with its true hues.

## Rendering & Performance (MEDIUM Priority)

24 - **Depth sorting for proper alpha blending** (**RE-PROMOTED to pre-release [LAUNCH]** 2026-07-15): Sort transparent geometry (Points, Lines, GSplats) back-to-front per frame so semi-transparent elements composite correctly. Without depth sorting, overlapping translucent primitives blend in submission order rather than depth order, producing incorrect colors and visible artifacts depending on view angle. Full phased plan (Option 3a — viewer-only, no format change): `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md`. Status: spec Phases 0–3 MERGED for gsplats (#511/#523/#553: premultiplied alpha, texture-backed storage, SortWorker, camera-triggered re-sort); partial appends (spec Phase 4 Stage 2) landed for all three geometry types. GSplats-first; Points sorting symmetry LANDED (arc PR-B, spec §8); Lines storage + sorting symmetry LANDED (arc PR-C, spec §8) — all three geometry types now share the texture storage + SortWorker machinery. Volumetric Phase 3 (points: isotropic chord-integral emission–absorption + points RGBA alpha + mandelbulb showcase) LANDED (arc PR-D); volumetric Phase 4 (lines: transverse chord integral + lines RGBA per-vertex alpha, `effectiveGeometryMode` deleted — all three geometry types now render + depth-sort real volumetric) LANDED (arc PR-E) — **ARC COMPLETE**. **Successor arc (#1352, 2026-08, in progress → see item 30):** lines are being moved off the screen-space quad onto a cylindrically-symmetric **capsule primitive** (ρ = G₂D(r) × box(s)) — ray-integral math (#1419), primitive behind `?linePrimitive=capsule` (#1426/#1481), picking (#1451), sharpness via an Abel-transform radial LUT (#1458), deficit-rule joint composition (#1487), the **default flip (#1492)**, and the docs fix (#1516) are all merged/closed; the capsule-joint defect cluster (#1488/#1490/#1494/#1495/#1497/#1501/#1502) remains open as a post-flip drain (see item 30).

30 - **Capsule line primitive — drain the joint follow-ups**
     (#1352, the successor arc to item 24's volumetric work; added 2026-08-11,
     updated same day). ✅ **The default flip is MERGED (#1492)** — capsule is
     now the default line primitive — and the docs fix landed (#1516, closed).
     Remaining to close the arc: the open capsule-joint defects, now live on
     the default path — one-sided deficit gate (#1495), lost far cap (#1490),
     stencil-reach bound (#1488), interpolated-radius partner reconstruction
     (#1494), hairpin re-chop (#1501), cut-normal snap banding (#1502),
     untested sharp-turn clause (#1497).

31 - **Mesh LOD/structure program** (in flight 2026-08; the "mesh follow-up
     program" — mesh reached UI feature-completeness earlier, this is its LOD
     story). Landed: **substitutive LOD** via `luxar.mesh.decimate` +
     `add_mesh(substitutive_lod=…)` (#1351), **kind=partition** path (#1382),
     uniform-colour support under substitutive_lod (#1485), lazy-level pinning
     out of the per-slice sweep (#1464), near-camera fade parity (#1438).
     The prerequisite **radial (concentric-shell) reveal ordering shipped for
     GSplats/Points/Lines** (#1448, `-m radial`, deliberately unstamped so no
     1/e brightening) — its own follow-ups are open: stale `reference_energy`
     on the substitutive path (#1455), `annotate-quality` re-stamping a reveal
     (#1454), Points/Lines accepting `method='radial'` with no ordering behind
     it (#1453), untested spatial-dims default (#1452).
     **In flight:** the mesh **additive ladder — a reveal, and only a reveal**
     (radial prefix ordering; a generic prefix of an index buffer is a holed
     surface, so only the reveal semantic is offered): authoring #1503 + viewer
     half #1515, integrating through **draft holding PR #1499**, with design
     follow-ups filed (#1506 cumulative-vs-per-level counts, #1507 prefix
     contiguity on closed surfaces, #1514 stacked-nD sequencing, #1517
     per-level vs per-node budgets). **Decimation-quality backlog:** colour
     loss/corruption for non-uint8 inputs (#1355), `luxar mesh lod` dropping
     transform/scalars/labels/siblings
     (#1357). Related: the nD-clipping deferral measurement is item 29.

22 - ~~**Level-of-Detail (LOD) with PartitionNode**~~: **DONE for release** (code-verified 2026-07-11). Beyond the core (archive item 22-core), the 2026-07 wave shipped: intent-first `--recipe` topologies (flat/stream/levels/tiles/overview/adaptive) with stream ladders on by default, viewport-relative coverage-fraction switching (`sqrt(N_i/N_finest)`, self-calibrating — no threshold knob), Q·e quality stamps + energy-gated upgrade release (`e(k) ≥ 0.6`), the never-downgrade display gate with subtree aggregation and refinement kick, sibling-aware ladders, per-part LOD at fit/merge time (`--recipe` on tiled fits and batch-fit merges), coverage inflation + mass conservation + `--refine l2|volume`, `annotate-quality` retrofitting, and byte-budget VRAM residency (coarse eager levels stay resident; fine lazy levels load on demand and evict off-screen-first under pressure). The advanced refinements formerly listed here were re-verified against the code (2026-07-11: 3 missing, 4 partial) and **demoted to Future/Exploratory item 25** — none is release-gating.

## Future / Exploratory (LOW Priority)

1 - ~~**Ray casting with object labels**~~: **DONE** (code-verified 2026-07-11). Implemented end-to-end: per-element `labels=`/`image_labels=` on all three geometry adders (`core/group/adders/{points,lines,gsplats}.py`) → CSR zarr arrays (`label_offsets`/`label_bytes`, `io/_compiler/labels/`) → GPU pick-buffer ray casting (`rendering/picking/`, per-geometry pick shaders) → hover pick resolves `elementId` → lazy CSR label decode (`data/loaders/picking/label-loader.ts`) → label shown in a fixed-screen-position overlay (default top-right `(0.98, 0.02)`, auto-injected by `core/scene/overlays/hover_inject.py`; `{hover_label}`/`{hover_node}`/`{hover_index}` templating in `ui/overlay-manager.ts`). Unit + E2E coverage (`hover-tooltip.spec.ts`, `label-loader.test.ts`). Note: the trigger is hover (mousemove settle) rather than click; an embedder `selection` event fires on the same pick.

2 - **Scene domains**: Introduce the concept of rendering "domains" beyond the main nD-to-3D slice:
    - **Overlay domain**: For a given set of non-visible dimensions, render an associated scene as a transparent overlay in normalized canvas coordinates ([0,1] x [0,1]), unaffected by camera controls.
    - **Sound domain**: Associate audio with a scene, played back on load to provide auditory context.

3 - **VR/AR mode**: Add the ability to activate VR/AR rendering for immersive exploration of 3D scenes.

25 - **Advanced LOD refinements** (demoted from item 22; code-verified still open 2026-07-11 — none release-gating):
    - **Per-splat opacity crossfade** during LOD transitions — MISSING. Transitions are hard visibility toggles with hysteresis (`scene/lod-group-registry.ts`); the only shader fade is the near-plane/coverage single-splat guard, not a transition crossfade.
    - **Pixel-error selection metric** — PARTIAL. Per-level fidelity is now *stored* (mixture-L² `Q`, `energy_fraction_cum`, `reference_energy` quality stamps) but selection is still projected-size `coverage_fraction`; the stored error feeds only the display/hold gate and is never projected to pixels.
    - **Monotonic error guarantee** — PARTIAL. Per-level `Q`/energy is stored, but no `parent_error >= child_error` bound is enforced across substitutive levels (additive ladders are monotone in cumulative energy by construction).
    - **Density-adaptive partitioning** — PARTIAL. `PartitionNode` rules remain median/midpoint/SAH (count/cost-balanced); density-driven partitioning exists only at fit time via `fit --tiling content` box plans, not as a PartitionNode splitter.
    - **Nanite-style stop-traversal** — MISSING. Partition children all load and stay visible (frustum culling only); no error-driven subtree pruning.
    - **Virtual residency via zarr chunks** — PARTIAL (close). Coarse eager levels stay resident, fine lazy levels fetch on demand and evict off-screen-first/furthest-first — but driven by byte-budget pressure over LOD geometry + decoded-chunk caches, not per-chunk camera-keyed paging.
    - **nD LOD metric for non-displayed dimensions, split-seam handling, split-granularity tuning** — MISSING (`extend_to_all` governs visibility only, not LOD; granularity is `max_elements`-count-driven).

26 - **Lines compiler auto-partition heuristic** (three-geometry symmetry gap, staged): `add_points` auto-partitions large clouds at the compiler level; `add_lines` does not (documented at the seam in `core/group/adders/lines.py` — the `partition=False` sentinel is already normalized for the day it's wired). Wire the same heuristic for Lines (and evaluate GSplats parity) or decide it's permanently Points-only and update the adder docs.

28 - **Probed-and-parked perf backlog** (measure-first campaign 2026-07-25/26 — verdicts + calibrated revisit triggers; raw archives `~/luxar-perf-campaign/`, method + numbers in the campaign memory. Campaign shipped L1 #693 + L8 #696; rejected-by-measurement archives #690/#694/#695; do NOT rebuild any of these without re-running the probe):
    - **OPFS segment packing — DROPPED for the probed regime.** OPFS-warm reload steps cost exactly network-cold-localhost steps (44 ms = 44 ms; 100% L2-served, decode dominates); writes are off the critical path since #574. No revisit trigger at current file-count profiles and decode costs (probe was localhost, OPFS-warm, decode-dominated) — the verdict flips only on a storage backend/browser where per-file metadata latency is a material fraction of a step, or a much higher file count per timepoint.
    - **Decoded-f32 chunk cache — DROPPED.** Warm scrub steps already skip decode via the S-cache (23 ms); the cold-arm L0-hit share (~35%) bounds the savable dequant at ≲15 ms/step. A higher hit share saves more, but the cold−warm gap caps the whole skipped pipeline (fetch + decompress + dequant) at ≈44 − 23 ≈ 21 ms/step (44 = the cold-arm step above — same scrub loop, first pass vs revisit), so even a perfect-hit decoded-f32 cache only brings a miss step to warm-step parity (~23 ms), never below it. **Trigger: L0 (chunk) re-read hit rate ≳ 50% CONCURRENT with S-cache miss rate ≳ 50%** — order-of-magnitude re-probe gates derived from the numbers above, not measurements; no workload meeting them is known (chunks are time-local).
    - **Post-projection cache for gsplats/lines (fold `uTruncate` into the S-cache key) — DROPPED at current scales.** Warm step ≈ 23 ms, mostly re-projection (10–19 ms; gsplat S-cache is pre-projection); saving ~15 ms/step is imperceptible. **Trigger: timepoints ≥ ~2 M splats** (cost is linear in N/tp — h2afva-class ~2.5 M/tp ⇒ ~50–60 ms/step, then this is a small, mechanical win).
    - **Manual-scrub prefetch (t±1 during keyboard/slider nav) — CONDITIONAL, remote-only.** Local ceiling 21 ms/step (imperceptible). At `luxar serve --profile 4g`: 208 → 105 ms/step (~50%) IF dwell allows the prefetch and it wins the throttled link from background ladder deepening (shared bandwidth — a reallocation policy, not a free win). **Trigger: remote-dataset scrub UX becomes a goal** (hosted demos / shared exports over real networks).
    - Instrument notes for whoever re-probes: the timelapse bench's perTp includes a 250–350 ms settle floor (not step latency); CDP `emulateNetworkConditions` does NOT throttle data-worker fetches — use `luxar serve --profile`; under throttle measure time-to-FIRST-commit (background deepening pollutes quiet-based metrics).

29 - **Exact nD triangle clipping for mesh** — DEFERRED BY DECISION, with a measurement now installed. `docs/specs/MESH_NODE_SPEC.md` §5 culls whole triangles by per-vertex slab membership. That is a *true cut* when the hidden dims are discrete (time, channel — the case mesh was scoped for) and only a **thick slab** when a hidden dim is continuous and spatial (§5.2.1). Exact clipping would remove the approximation at roughly **1500 LOC across two backends** (Rust + the TS reference, which must stay in 1:1 parity and is also the production >16D path).
    - **Why deferred, explicitly:** not for lack of time — the cost/benefit is bad *today*. It is dual-backend code with parity tests, permanently maintained, for a configuration with no known users. The §5 kernel was chosen precisely because it covers the dominant real case for ~10% of the cost.
    - **The promotion trigger, now falsifiable.** The spec's condition was "if continuous hidden spatial dims turn out to be a real use case", which nothing measured. `processMeshData` now emits a `log.info` when a mesh's hidden dims include a continuous one, naming each such dimension and its unit (`data/scene-loader/process/data-processor-mesh.ts::noticeContinuousHiddenDim`), deduplicated per node and dimension (an `extend_to_all` dim is skipped — its slab is infinite, so the approximation cannot bite there). Hidden-and-continuous tracks the spec's "continuous hidden **spatial**" condition closely rather than being a loose superset: `core/dimensions.py` forces a dimension *authored* non-displayed and non-spatial to be discrete, so an axis authored hidden can only reach the line with `spatial=true`. The one gap is runtime rather than authoring — a dimension authored `display=true, spatial=false` keeps the flag false and becomes hidden the moment the display axes are swapped — which argues for reporting it, not filtering it out: testing the flag would narrow nothing on the authored case while dropping evidence there and from any scene whose metadata omits it. The name and unit are printed for the one judgement no flag can make: a dimension may be *declared* spatial and still be a time axis, where a slab is a reasonable thing to want.
    - **Promote when** that line starts appearing against real datasets with a spatial axis. Until then the deferral stands on evidence rather than on assertion.

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

18 - ~~**Tiled fitting for large volumes**~~: **DONE.** `luxar gsplat fit --tiling uniform`.

### Rendering & Performance (completed)

22-core - ~~**LOD + PartitionNode core**~~: **DONE.** Composable scene graph LOD landed across Python and viewer:
    - **LODNode** (`kind:'lod'`): screen-space pixel-size selector with asymmetric, spacing-aware hysteresis — `lod-group-registry.ts`, `types/lod-group.ts`.
    - **PartitionNode** (`kind:'partition'`): spatial BSP partitioning (balanced median [default] + midpoint + opt-in SAH BSP) with per-mesh frustum culling and an auto-partition heuristic — `core/group/partition.py`, `core/group/auto_partition.py`, `data/scene-loader/nodes/load-partition-group-node.ts`.
    - **Recursive composition**: arbitrary nesting of `lod`/`partition` groups via standard scene-graph loading.
    - **Decimation CLI**: `luxar gsplat lod --recipe {flat,additive,partitioned,multiscale,mosaic,substitutive,pyramid}` (greedy / self_energy / mass / kmeans_lloyd, energy/count breakpoints; per-part additive ladders, the unbalanced multiscale tree, and per-part substitutive mosaic) — `cli/lod.py` + `gsplats/lod/recipes.py`.
    - **LOD ordering**: Poisson-disk (Bridson) + spatial-uniform — `core/group/lod/poisson_disk.py`.
    - **LOD for lines**: polyline-aware, connectivity-preserving simplification — `core/group/lod/lines.py`.
    - **Progressive multi-additive-LOD loading** for Points & Lines — `data/points/points-progressive-loader.ts`, `data/lines/lines-progressive-loader.ts`.
    - **Zarr v2.0 format**: substitutive × additive LOD matrix — see `docs/specs/GSPLATS_ZARR_FORMAT.md`.

### Feature Requests (completed)

5 - ~~**Layers panel**~~: **DONE.** Press `L` to toggle per-layer control panel.

6 - ~~**Orthographic projection mode**~~: **DONE.** Toggle via rendering controls or keyboard shortcut.

15 - ~~**Viewer-side colormaps**~~: **DONE.** Full colormap (CLUT) support for all geometry types.

17 - ~~**OME-Zarr (NGFF) input support**~~: **DONE.** `luxar gsplat fit` supports OME-Zarr.

19 - ~~**Slurm batch fitting**~~: **DONE.** `luxar gsplat slurm-fit` CLI command.

20 - ~~**nD Transforms**~~: **DONE.** Per-dimension affine and permutation transforms.

21 - ~~**Transform model review & fixes**~~: **DONE.**

0 - ~~**Retire gsplat sharpness from the viewer**~~: **DONE.**

</details>

## Notes

- Review and update this list regularly.
- Consider creating GitHub issues for tracking progress on individual items.
- Update CLAUDE.md when implementing significant changes.
