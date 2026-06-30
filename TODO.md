# Luxar TODO List

This file tracks known issues, planned features, and improvements for the Luxar project.

---

## 🚀 Path to Release & Announcement

The goal: a clean public launch (GitHub announce + `pip install luxar` + bioRxiv
preprint). The engineering is deep and the science has converged — the remaining
gap is mostly **release mechanics + first-impression polish**, not missing
capability. Items below are grouped by track and tagged **[BLOCKER]** (must land
before announce) / **[LAUNCH]** (strongly wanted for day one) / **[POST]** (fine
to ship after). Sequencing is at the bottom.

### A. Release mechanics — the literal blockers to "install luxar"

- **R1 [LAUNCH] — Version bump (CalVer `YYYY.MM.DD`).** ✅ Mechanism already
  exists: `pyproject.toml` has `dynamic = ["version"]` sourced from
  `__version__` in `packages/luxar/src/luxar/__init__.py`, and CalVer is already
  in use (currently `2026.06.05`). On release day, **bump that one line** to the
  date (and the viewer `package.json` to match). Verified: `hatch build`
  produces `luxar-<version>-py3-none-any.whl`.
  - **⚠️ PEP 440 / semver caveat:** both ecosystems **strip leading zeros** from
    numeric segments, so `2026.06.29` normalizes to **`2026.6.29`** on PyPI
    (PEP 440) and is what semver/npm requires too. The git **tag** can stay
    zero-padded (`v2026.06.29`) for human readability, but the installed package
    version will display unpadded. Pick one display form and use it consistently
    in the tag, release notes, and citation so they don't visibly diverge.
  - Because the version equals the release date, **set it last** (a release-prep
    step), not now — otherwise it goes stale. A `make set-version DATE=...`
    helper that stamps both `pyproject.toml` and `package.json` keeps them in
    sync.
- **R2 [BLOCKER] — First git tag + release.** ✅ **Release tooling built** —
  `scripts/release.sh` + Make targets, tested (bash-3.2 safe, guards verified):
  - `make set-version [DATE=YYYY.MM.DD]` — stamps `__version__` (zero-padded) +
    viewer `package.json` (semver-normalized); you commit it via a PR (main is
    branch-protected, `enforce_admins` on — no direct pushes).
  - `make release-check` — dry-run: runs ALL preflight (on main, clean tree, in
    sync with origin, publish.yml committed, CalVer valid, tag free, **required
    CI checks green on the main SHA**), mutates nothing.
  - `make release` — same preflight + interactive confirm, then creates and
    pushes the `v<version>` tag → fires `publish.yml` → OIDC publish to PyPI.
    Never pushes to main, never uploads locally.
  - The old `make publish` / `publish-test` (`hatch publish`, a local-upload
    foot-gun that bypassed OIDC and shipped a viewer-less wheel) are now
    **disabled** with a pointer to `make release`.
  - **Runbook for the first release:** `make set-version` → PR + merge (CI green)
    → `make release-check` → `make release`. Then draft the GitHub release notes
    from CHANGELOG `[Unreleased]`.
- **R3 [BLOCKER] — PyPI publication.** Luxar is not on PyPI.
  - ✅ **Trusted publisher registered** (owner `royerlab`, repo `luxar`, workflow
    `publish.yml`, env `pypi`) and **`.github/workflows/publish.yml` written** —
    tag-triggered (`v*`), builds viewer (WASM+dist) → `hatch build` → publishes
    via OIDC; includes a tag↔`__version__` guard, a "viewer bundled" assertion,
    and a `workflow_dispatch` dry-run path to TestPyPI.
  - ✅ **GitHub environments created**: `pypi` (hardened — deploys only from
    `v*` tags) and `testpypi`. The PyPI trusted publisher is registered.
  - ⏭️ **TestPyPI dry-run deliberately skipped** (decision 2026-06-29): TestPyPI
    is a *public* index and the GitHub repo is still private, so an upload would
    be Luxar's first public footprint — not worth it pre-launch. Local validation
    (twine check + clean install) covers the packaging; the OIDC upload handshake
    is the only thing left unverified and can be confirmed at launch. If a
    dry-run is ever wanted: register a *TestPyPI* trusted publisher (test.pypi.org
    account) and use a throwaway `.devN` version (TestPyPI version strings are
    immutable, same as PyPI).
  - ✅ **Packaging dry-run validated locally** (`hatch build` + `twine check` +
    clean-venv install + CLI smoke test): wheel installs, `luxar --version`
    works, and the bundled viewer resolves to `site-packages/luxar/_viewer_dist`.
  - ✅ **Size blocker found & fixed:** the wheel/sdist were **258 MB / 257 MB**
    because ~237 MB of Git-LFS demo data under `demos/data/` was swept in (and in
    CI it would have shipped as broken LFS *pointer* files). Added
    `**/demos/data/**` to the wheel + sdist `exclude` in `pyproject.toml` →
    **wheel 10 MB, sdist 8.5 MB, max file 11.9 MB** (PyPI per-file limit 100 MB).
    `twine check` PASSED.
  - ⚠️ **Launcher caveat:** a locally-built wheel bundles the host launcher
    (`cli/_launchers/darwin-universal`, 11.9 MB) into a `py3-none-any` wheel. The
    CI publish runs on `ubuntu-latest` with no launchers built, so the **PyPI
    artifact won't carry it** — but never publish a wheel built on a dev machine
    that has run `make build-launchers`. (Proper per-platform launcher wheels are
    a larger, post-launch design item.)
  - TODO before first publish: confirm extras (`luxar[gsplats]`) resolve on a
    clean machine; do a TestPyPI upload via the dry-run; decide whether to also
    trim `tests/` from the wheel (minor — largest test file is ~0.12 MB).
- **R3-npm [LAUNCH] — npm publication of the viewer library.** The TS/WebGL
  viewer ships to npm as **`@royerlab/luxar-viewer`** (scoped; bare `luxar` is
  taken on npm, `luxar-viewer` is free but the scope protects the brand and
  matches the GitHub org).
  - ✅ **Package scoped + publish-ready:** `package.json` renamed to
    `@royerlab/luxar-viewer` with `publishConfig.access=public` +
    `provenance=true`. The lib build path is mature (`pnpm run ci:release` =
    `build:lib` + export sanity check); verified locally — builds clean, `three`
    externalized, barrel side-effect-free.
  - ✅ **Sourcemaps trimmed:** `vite.lib.config.ts` now builds the lib with
    `sourcemap: false` (the lib build exists only to be packed for npm). Cut the
    tarball from **5.1 MB / 21 MB unpacked → 2.2 MB packed / 8.6 MB unpacked**
    (no `.js.map` files; 551 files).
  - ✅ **Workflow written:** `.github/workflows/publish-npm.yml` — tag-triggered
    (`v*`, same as PyPI), tag↔`package.json` version guard, builds the lib,
    `npm publish --provenance --access public`, plus a `workflow_dispatch`
    dry-run (build+pack, no secrets). `scripts/release.sh` now also checks for it
    and names it in the plan.
  - ✅ **Version sync:** `make set-version DATE=...` (`scripts/set_version.py`)
    already stamps the viewer `package.json` (semver-normalized) alongside the
    Python `__version__`.
  - ⛔ **One-time npm-side setup still TODO (do at launch).** Target end-state is
    token-less **OIDC trusted publishing with provenance**, mirroring the PyPI
    setup. **Key gotcha:** npm has no "pending publisher" (unlike PyPI), so a
    trusted publisher attaches only to a package that *already exists* — the
    first publish must bootstrap with a token; OIDC takes over afterward. Full
    steps live in the `publish-npm.yml` header; in brief:
    1. **Account + org:** create an npm account (enable 2FA) and the free
       `royerlab` org (the scope must exist before any `@royerlab/*` publish).
    2. **Bootstrap first publish** (token, once): locally `npm login` →
       `cd packages/luxar-viewer` → `pnpm run ci:release` →
       `npm publish --access public` (creates the package). *Or* in CI via an
       `NPM_TOKEN` secret + `ENABLE_NPM_PUBLISH=true` + a tag push.
    3. **Configure trusted publisher** on
       npmjs.com/package/@royerlab/luxar-viewer → Settings → Trusted Publisher →
       GitHub Actions: org `royerlab`, repo `luxar`, workflow `publish-npm.yml`,
       environment `npm`.
    4. **Switch to token-less + turn on:** create the GitHub `npm` environment
       (restrict to `v*`), **delete** the `NPM_TOKEN` secret (OIDC takes over via
       `id-token: write`), and set repo variable `ENABLE_NPM_PUBLISH=true` (the
       master switch — until `true`, tag pushes build+pack but never publish).
    From then on `make release` (tag push) publishes both PyPI and npm, the
    latter token-less with a verified provenance attestation.
- **R4 [BLOCKER] — Citation + DOI.** README cites a placeholder
  `@software{luxar2024}` with year 2024 and no preprint. Wire in the bioRxiv DOI
  and a proper `CITATION.cff` once the preprint is posted.
- **R5 [BLOCKER] — Branch cleanup to a clean `main`.** Currently on
  `feat/gsplat-cholesky-split-storage` with ~7 unmerged feature branches +
  dependabot PRs. Merge/close everything intended for 1.0, drain dependabot,
  confirm CI green on `main`.
- **R6 [LAUNCH] — Repo hygiene.** Remove/relocate the 141-entry `delme/`, the
  untracked `lightsheet_overview/`, `test-results/`, and stray caches before the
  repo is public-facing. Confirm `.gitignore` covers generated `.zarr`, build
  artifacts, and coverage.
- **R7 [LAUNCH] — License/authorship sanity.** BSD-3 is in place; confirm
  third-party acknowledgments (Three.js, Zarr/Zarrita, datasets) are complete
  and the announce copy doesn't overclaim.

### B. First-impression polish — what a visitor sees on day one

- **R8 [LAUNCH] — UI discoverability** (see detailed item **#9**). The UI leans
  on hidden keyboard shortcuts; add visible affordances (buttons/menus/hints).
  Highest-leverage polish item for a public launch.
- **R9 [LAUNCH] — Example-dataset bugs** (see detailed item **#23**).
  `scalars_and_colormap`, `scene_dimensions`, `transform` examples surface
  loader/viewer bugs — these are exactly what a first-time visitor runs.
- **R10 [LAUNCH] — Depth sorting for alpha blending** (see detailed item
  **#24**). Translucent geometry composites in submission order → view-dependent
  artifacts, visible in any splat/point demo.
- **R11 [LAUNCH] — README/landing pass.** Refresh the gallery, ensure the
  quick-start path works end-to-end on a fresh machine, and that `luxar demo`
  is flawless (it's the first thing everyone runs).
- **R12 [POST] — Theme layout consistency (#7)**, **Python-side panel visibility
  config (#8)** — nice-to-have, not launch-gating.

### C. Manuscript / reproducibility — bioRxiv defensibility

- **R13 [BLOCKER for preprint] — Methods specificity.** Add per-preset
  hyperparameter tables (draft/standard/hifi/ultra), exact seed counts, the
  H.265 encoder version + exact command line, and blind-spot mask seeding.
  Classic reviewer bait; currently ~100 lines for a paper claiming 12 datasets +
  CV + tiled fitting + web architecture + an H.265 baseline.
- **R14 [LAUNCH] — Consumer-GPU timing benchmark.** Wall-time is softened to
  "minutes per dataset" pending a consumer-GPU sweep (only RTX PRO 6000 numbers
  documented). Run the benchmark on a commodity card (e.g. RTX 3070) to firm up
  the claim.
- **R15 [POST] — Tiled-fitting figure.** Described in Methods + Results but has
  no figure; a reviewer may ask for a seamless-stitching demonstration. The
  `tiled_fitting/` SD is method-only (empirical eval deferred).
- **R16 [LAUNCH] — Manuscript repo hygiene.** Commit a clean "regenerate all
  artifacts" pass; prune stale/duplicate figure PDFs (`cross_validation.pdf`,
  the three `compression_*.pdf` variants); confirm SD6 viewer-perf PDFs use the
  committed `sweep_v6` numbers, not placeholders.

### D. Sequencing (suggested order, parallelizable across tracks)

1. **Stabilize `main`** — R5 (merge/close branches) → R6 (hygiene) → CI green.
2. **Decide versioning & package** — R1 → R3 (PyPI dry-run / TestPyPI) in parallel
   with the day-one polish (R8/R9/R10/R11).
3. **Post the preprint** — R13/R14/R16 land → bioRxiv → obtain DOI → R4 (wire
   citation back into the repo).
4. **Cut the release** — R2 (tag + GitHub release) → flip PyPI to live (R3) →
   announce.
5. **Post-launch backlog** — R12, R15, and the existing Rendering/LOD and
   Future/Exploratory items below.

> Single biggest unblocking action: **R5 → R1** (clean `main`, then a real
> version). Everything else can proceed in parallel once those two land.

---

## Infrastructure & Polish

4 - ~~**Cache eviction policy**~~: **DONE.** Root cause found & fixed: L2 (OPFS) eviction was gated only on the configured `maxSize` (default `l2MaxSizeMB: 2048` → 2 GB), but the browser-granted OPFS quota is often far smaller. The quota gate in `OPFSStore.doSet` rejected writes (counted as `quotaWriteSkipped`) long before `totalSize` reached 2 GB, so the maxSize-based eviction loop never ran — the LRU froze holding old entries and silently dropped new ones (worst on Firefox/private-mode/small disks; invisible on roomy Chrome). Fix: `doSet` now evicts LRU entries on quota pressure (not just maxSize pressure) and re-checks, since deleting files genuinely frees quota. Bounded by index size with a no-progress guard. Regression test: `tests/unit/cache/opfs-eviction-quota.todo4.test.ts`. (L0/L1 size-gated eviction was already correct.)

7 - **Theme layout consistency**: All Luxar UI themes should differ only in colors, transparency, and visual effects — never in the size or layout of panels and their components. This ensures a consistent user experience across themes.

8 - **Panel visibility configuration**: Allow configuring which panels are visible (Logs, Rendering Controls, Data Monitor, Dimensions, etc.) from the Python side. Optionally lock panel visibility to enforce a particular look and prevent user modifications.

9 - **UI ergonomics**: The current UI relies heavily on hidden keyboard shortcuts to reveal panels, which is poor discoverability. Improve with visible affordances (buttons, menus, or indicators).

## Bugs

23 - **Fix bugs surfaced by examples**: Several example datasets surface viewer/loader bugs that need fixing:
    - `scalars_and_colormap_example.zarr`
    - `scene_dimensions_example.zarr`
    - `transform_example.zarr`

## Rendering & Performance (MEDIUM Priority)

24 - **Depth sorting for proper alpha blending**: Sort transparent geometry (Points, Lines, GSplats) back-to-front per frame so semi-transparent elements composite correctly. Without depth sorting, overlapping translucent primitives blend in submission order rather than depth order, producing incorrect colors and visible artifacts depending on view angle.

22 - **Level-of-Detail (LOD) with PartitionNode** — core landed, advanced refinements remain.

    **Implemented (see Completed archive, item 22-core):** LODNode (`kind:'lod'`, pixel-size selector + hysteresis), PartitionNode (`kind:'partition'`, median [default] + midpoint + SAH BSP, auto-partition heuristic), recursive scene-graph composition, `luxar gsplat lod --recipe` (flat / additive / partitioned / multiscale / mosaic / substitutive / pyramid), Poisson-disk + spatial-uniform LOD ordering, LOD for lines (connectivity-preserving), progressive multi-additive-LOD loading for Points & Lines, zarr v2.0 (substitutive × additive matrix) format.

    **Still TODO:**
    - **Per-splat opacity crossfade** during LOD transitions — each splat's opacity modulated by the transition factor to avoid brightness doubling from overlapping semi-transparent layers. (Currently transitions are hard switches with hysteresis; no crossfade.)
    - **Pixel-error metric, not just projected size**: LOD selection should be based on "how many pixels of screen-space error would this simplification introduce" — not just projected bounding box size. A flat region with 1M splats may have near-zero simplification error (coarse is fine), while a detailed region at the same screen size may need fine LOD. Store per-level error, project it to pixels at runtime. (Currently selector is pure `pixel_size`.)
    - **Monotonic error guarantee**: Each LODNode should store its simplification error (max amplitude difference, spatial displacement vs. fine level) with `parent_error >= child_error` guaranteed at every level so top-down traversal always converges. (Greedy additive ordering has a `(1-1/e)` submodular guarantee but no stored monotonic error bound.)
    - **Density-adaptive partitioning**: PartitionNode should partition based on splat density / detail, not a balanced/uniform grid. Dense regions get more parts, empty regions fewer (cluster by density). The default median BSP gives balanced *counts* and SAH BSP is cost-driven, but neither is density-driven.
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
