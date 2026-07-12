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
- **R5 [BLOCKER] — Branch cleanup to a clean `main`.** ✅ **DONE** (2026-07-01),
  with the caveat that dependabot re-accumulates continuously — **re-drained
  2026-07-11**: 4 green dep bumps merged (knip, fast-check, mediabunny,
  playwright); the TypeScript 5.9→6.0 major bump was **deferred post-launch**
  (`@dependabot ignore this major version`) because TS 6 deprecates `baseUrl`
  (TS5101) and needs a real tsconfig migration. Treat "drain dependabot +
  confirm CI green on `main`" as a recurring final release-prep step, not a
  one-time task.
- **R6 [LAUNCH] — Repo hygiene.** Remove/relocate the 141-entry `delme/`, the
  untracked `lightsheet_overview/`, `test-results/`, and stray caches before the
  repo is public-facing. Confirm `.gitignore` covers generated `.zarr`, build
  artifacts, and coverage.
- **R7 [LAUNCH] — License/authorship sanity.** BSD-3 is in place; confirm
  third-party acknowledgments (Three.js, Zarr/Zarrita, datasets) are complete
  and the announce copy doesn't overclaim.
- **R17 [LAUNCH] — Retire git-LFS for heavy processed datasets → Zenodo
  (versioned, fetch-on-demand).** The compute-expensive precomputed demo data
  (`.gsplats.zarr.zip`, catalogs) lives in **git-LFS** under
  `packages/luxar/src/luxar/demos/data/**` — ~237 MB tracked today, dominated by
  `gsplats_cmu1_pathology` (~330 MB / 3 ch), `gsplats_celegans` (206 MB), and
  `milky_way_gaia_8m` (143 MB). This does not scale: newer/larger datasets can't
  be committed at all, a public repo pays LFS storage+bandwidth limits, the wheel
  already has to `exclude` `demos/data/**` (see R3), and CI risks shipping broken
  LFS *pointer* files. Migrate heavy datasets out of the repo and fetch them on
  demand.
  - **Access confirmed:** `ZENODO_TOKEN` is on the dev machine (`~/.zshrc`); the
    Zenodo API is reachable (HTTP 200) and the account already owns 14
    depositions (zebrahub / ultrack / daxi / …), so there is both access and
    royerlab precedent. Zenodo records carry a **concept DOI with versioning** —
    exactly the "multiple steps / versions" the migration wants.
  - **Reusable mechanism already in-tree:** `demos/demo_cosmicflows_laniakea.py`
    has the pattern (`~/.cache/luxar/<name>` + atomic `download_file` +
    `ensure_input_data`). Generalize it into a shared
    `luxar.demos.data_fetch` helper that pulls a named dataset from a Zenodo
    record (verifying a checksum) into the cache on first run.
  - **Plan (incremental, largest-first, keeps LFS as fallback until proven):**
    (1) shared fetch helper + a committed `demos/data/manifest.json` pinning each
    dataset's Zenodo record/URL + checksum + expected version; (2) create the
    "Luxar demo datasets" Zenodo deposition, upload current files; (3) repoint
    demos at the helper one at a time; (4) `git rm` the migrated files and drop
    their `.gitattributes` LFS globs. Future large datasets land as **new Zenodo
    versions**, with the manifest pinning what each Luxar release expects.
  - **Scope note:** only *heavy processed datasets* move. Small README/doc images
    (`docs/images/**`) must stay in-repo so GitHub renders them (see R19).

### B. First-impression polish — what a visitor sees on day one

- **R8 [LAUNCH] — UI discoverability** ✅ **DONE** (2026-07-01, #432; see detailed
  item **#9**). Shipped an always-visible left **control rail** (Concept A slice 1):
  one icon per panel wired to the *same* commands as the keyboard shortcuts (via
  `InputHandler.getUiActions()` — can't drift), a vendored FPS/ms/graph perf
  readout (drops `stats.js`), event-driven active-state, collapse/idle-dim/first-run
  hint, and full theming via a single `.luxar-glass-surface` marker. Hardened over
  4 adversarial deep-double-check rounds (~40 fixes w/ regression tests: webkit
  fullscreen across all consumers, a11y, embedder ref-count, dataset toggle);
  CI green. Follow-up: full panel-docking trays (Concept A slice 2).
- **R9 [LAUNCH] — Example-dataset bugs** ✅ **DONE** (see detailed item **#23**).
  Reproduced & resolved 2026-07-01: `transform` renders correctly (no bug —
  already fixed); `scene_dimensions` nav fixed by aligning `step` to the data
  sampling (#428, merged); `scalars_and_colormap` washout fixed with
  `blending_mode="max"` (#430) — the washout was additive *summation* of the
  self-overlapping spiral (order-independent, so NOT a #24 depth-sorting
  issue), and `max` (brightest-wins) shows each colormap's true hues.
- **R10 [POST] — Depth sorting for alpha blending** (see detailed item
  **#24**). Translucent geometry composites in submission order → view-dependent
  artifacts, visible in any splat/point demo. **DEFERRED post-publication/release**
  (re-confirmed 2026-07-11; retagged [LAUNCH] → [POST] — not launch-gating).
- **R11 [LAUNCH] — README/landing pass.** ✅ **Mostly done** (2026-07-01):
  audited the quick-start end-to-end — `luxar demo` generates + renders
  flawlessly (10k-pt Lorenz), every documented Python snippet runs, and all
  gallery images + doc links resolve. Fixed 3 stale/confusing README spots
  (#429): sharpness range (`0.5-10` → normalized `0-1`), `gsplat lod`
  signature (→ `--recipe {…}`), and the redundant two-terminal "View it"
  block (→ single `luxar serve … --viewer --open`). **Remaining:** verify a
  truly-fresh-machine `make setup-dev`, and optionally regenerate the gallery
  media (`make generate-readme-images/videos`).
- **R18 [LAUNCH] — Documentation: content pass + confirm it's publicly
  viewable.** The **hosting is already wired**: `.github/workflows/docs.yml`
  builds Sphinx (Python API) + typedoc (viewer) and deploys to **GitHub Pages**
  (`https://royerlab.github.io/luxar/`) on every push to `main` touching
  `docs/**` or the sources. Two gaps remain for day one:
  - **Publicly viewable:** the repo is still private, so the Pages site isn't
    reachable by outsiders yet. Confirm it goes live when the repo is made public
    (or enable/verify Pages visibility), and that the built site actually renders
    — nav, API autosummary, viewer typedoc, and images all resolve.
  - **Content cleanup:** `docs/` carries internal/stale trees that should not
    ship in public docs — `archive/`, `handoffs/`, `reports/`, `bugs/`,
    `templates/`, `benchmarks/`. Prune or exclude them from the Sphinx build,
    then update/improve the user-facing guides + API reference to match the
    current surface (gsplat cal→fit→lod pipeline, LOD recipes, export/native,
    batch-fit, filtering). Cross-check against the CLI so examples don't drift.
- **R19 [LAUNCH] — README refresh + showcase the newer/better demos (images +
  video).** Extends R11 (whose one open remainder was "regenerate the gallery
  media"). The pipeline exists: `make generate-readme-demos →
  generate-readme-images / generate-readme-videos` (Playwright captures of the
  live viewer → `docs/images/readme/*.{png,gif,webp}`).
  - **Curate a stronger gallery** from the newer/better datasets (H&E pathology
    gsplats, 4D *C. elegans* tracking, organoid multichannel, Gaia) — decide
    which few best convey the range (volumetric splats, nD navigation, scale).
  - **Regenerate** stills + short loops via the Playwright pipeline; refresh the
    README gallery section + captions; ensure everything renders on GitHub.
  - **LFS interaction (coordinate with R17):** README/doc images stay in-repo
    (GitHub must render them inline) — only the heavy *datasets* move to Zenodo.
    But the media the gallery captures come from demos whose *inputs* may have
    moved to Zenodo, so `generate-readme-demos` must run after R17's fetch path
    exists (or before the migration). Keep gallery media small/optimized.
- **R20 [LAUNCH] — Demos quality overhaul** (in progress, parallel agent —
  **PR #488**, branch `worktree-demos-quality-overhaul`, ~+1785/−1286 across the
  demo suite). Crash fixes, stale-doc fixes, shared caching, alias removal, and
  colorbar/channel legends across the demos. First-impression-critical: the
  demos are what `luxar demo` / the gallery / new users hit first. **Land this
  before R19** (the gallery is captured from these demos) and coordinate with
  R17 (shared caching should route through the same fetch/cache layer).
- **R21 [LAUNCH] — New "turnkey three" science demos** (in progress, parallel
  agent — branch `demos-turnkey-three`, **no PR yet**). Adds three geometry-
  showcasing demos with tests: **asteroids / solar system** (points),
  **Milky Way dust** (full-resolution gsplat fit), and **Dip-C 3D genome**
  (with a Layers-panel haplotype toggle). Strong candidates for the R19 gallery.
  **⚠ Feeds R17 directly:** this branch commits *new* heavy LFS data
  (`milkyway_dust.gsplats.zarr.zip`, `dipc_gm12878.npz`) — exactly the kind of
  compute-expensive processed dataset R17 moves to Zenodo. Open a PR, then either
  migrate its data as part of R17 or land it to Zenodo from the start rather than
  adding more git-LFS weight.
- **R12 [POST] — Theme layout consistency (#7)**, **Python-side panel visibility
  config (#8)** — nice-to-have, not launch-gating.

### C. Manuscript / reproducibility — bioRxiv defensibility

- **R13 [BLOCKER for preprint] — Methods specificity.** ✅ **DONE** (2026-07-08).
  A three-way audit (manuscript claims ↔ `methods.tex` ↔ code) found the Methods
  were stronger than assumed — blind-spot mask seeding (seed 42, 5%, 3×3×3 donut),
  Adam/LR/dilution, loss+regularizers, dynamic ops, tiled window, encoding, and
  viewer were already specified. The overnight `manuscript-update/code-sync-jul2026`
  session incorporated the remaining gaps: the literal H.265 `ffmpeg`/`libx265`
  command + a version-capture note, seed-generation numerics (edge threshold 0.1,
  min-separation 2 vox), the reference fit config, and τ=2.75. The one confirmed
  factual error — Methods+SD2 claimed a nonexistent "fourth, pywt Haar-wavelet"
  MAD estimator — was corrected to the three estimators actually in
  `calibration.py` and both PDFs rebuilt (luxar-paper `fc0780c`, pushed). NOTE:
  the named draft/standard/hifi/ultra presets only vary 4 knobs and the paper fits
  by explicit config, so a large per-preset matrix was correctly not added.
- **R14 [LAUNCH] — Consumer-GPU timing benchmark.** ✅ **DONE** (2026-07-12).
  Ran the fit-timing sweep on a commodity NVIDIA RTX 3070 (8 GB) across the 9
  core datasets × 5 splat counts (4K–256K), same config as the main
  rate-distortion analysis (early stopping). Operating point (32K): all 9 fit in
  38–355 s (**median ~4.2 min**) — substantiates "minutes per dataset" on
  commodity hardware. 0/45 cells OOM (peak ≤4.8 GB even for the ~100-Mvox
  light-sheet volumes, since the render kernel bounds VRAM independently of
  volume size). Shipped as luxar-paper **SD13** (new supplement + harness), with
  the claim wired into the main-text wall-time sentence and the Methods hardware
  paragraph (luxar-paper #5 merged; lint/type sweep #6 merged alongside).
- **R15 [POST] — Tiled-fitting figure.** Described in Methods + Results but has
  no figure; a reviewer may ask for a seamless-stitching demonstration. The
  `tiled_fitting/` SD is method-only (empirical eval deferred).
- **R16 [LAUNCH] — Manuscript repo hygiene.** Commit a clean "regenerate all
  artifacts" pass; prune stale/duplicate figure PDFs (`cross_validation.pdf`,
  the three `compression_*.pdf` variants); confirm SD6 viewer-perf PDFs use the
  committed `sweep_v6` numbers, not placeholders.

### D. Sequencing (suggested order, parallelizable across tracks)

1. ✅ **Stabilize `main`** — R5 (branches merged/drained) + R6 (hygiene:
   `delme/`/`test-results/` are gitignored, won't ship) **done**; `main` clean,
   no open feature branches, dependabot drained, release pipeline landed (#417).
2. **Decide versioning & package** — R1 → R3 (PyPI dry-run / TestPyPI) in parallel
   with the day-one polish (R8/R9/R10/R11).
3. **Post the preprint** — R13 ✅ / R14 ✅ / R16 land → bioRxiv → obtain DOI → R4
   (wire citation back into the repo).
4. **Public-repo readiness** (parallel with the preprint track, before the repo
   goes public) — first land the in-flight demo work: **R20** (demos quality
   overhaul, PR #488) and **R21** (new "turnkey three" demos, branch
   `demos-turnkey-three`); then **R17** (retire git-LFS heavy datasets → Zenodo —
   fold in R21's new `milkyway_dust`/`dipc` data so the public clone is lean and
   future large datasets are addable), **R18** (docs content pass + confirm the
   GitHub Pages site is publicly viewable), **R19** (README refresh + regenerated
   gallery). Order within the step: **R20/R21 → R17 → R19** (the gallery is
   captured from the finalized demos, whose inputs live in Zenodo by then); R18
   is independent and can run any time.
5. **Cut the release** — R2 (tag + GitHub release) → flip PyPI to live (R3) →
   announce.
6. **Post-launch backlog** — R10 (#24 depth sorting), R12, R15, and the
   remaining Future/Exploratory items below (incl. #25 advanced LOD
   refinements).

> Update 2026-07-01: R5/R6 (clean `main`) and the day-one polish R8/R9/R11 have
> landed (R8 control rail merged #432); R10 (#24 depth sorting) is **deferred
> post-release**. The critical path is now the **preprint** (R13/R14/R16 →
> bioRxiv → DOI → R4) and the **release cut** (R1 version bump → R2 tag →
> R3/R3-npm go-live).

> Update 2026-07-11: R13 (Methods specificity) is **done** (#466) and
> dependabot is re-drained (R5 note above). All release *tooling* is built and
> validated — what remains on the critical path is **execution**, in order:
> **(1) preprint track** — R14 (consumer-GPU timing sweep) ✅ **done**
> 2026-07-12 (luxar-paper SD13); remaining: R16 (manuscript repo hygiene), and
> decide whether the 13-dataset analyses get re-run with floor suppression
> (`--floor`, #463) before or after bioRxiv — then post → DOI → R4
> (CITATION.cff). **(2) day-one leftovers** — R11 remainder
> (fresh-machine `make setup-dev` verify, optional gallery media regen) + R7
> (license/acknowledgments audit). **(3) the mechanical cut** — final
> dependabot/branch drain → `make set-version` → PR → `make release-check` →
> `make release` (fires PyPI OIDC publish) → one-time npm bootstrap (R3-npm)
> → GitHub release notes from CHANGELOG `[Unreleased]` → announce.

> Update 2026-07-12: R14 done (see above). Added three **public-repo-readiness**
> items (new sequencing step 4, all [LAUNCH], parallelizable with the preprint):
> **R17** — retire git-LFS heavy processed datasets (~237 MB today; pathology
> /celegans/Gaia dominate) to **Zenodo** with fetch-on-demand + a pinned
> manifest (access verified: `ZENODO_TOKEN` present, API live, 14 royerlab
> depositions; reuse the `demo_cosmicflows_laniakea.py` cache pattern; migrate
> largest-first, versioned). **R18** — docs content pass + confirm the already-
> wired GitHub Pages site (`royerlab.github.io/luxar`, built by
> `docs.yml`) is publicly viewable and free of internal `docs/` trees. **R19** —
> README refresh + regenerated gallery from the newer demos (Playwright pipeline
> exists; depends on R17's fetch path). These gate a *clean public repo*, not the
> preprint; they should land before the repo is flipped public in the cut.
>
> Also tracking two in-flight demo efforts by parallel agents: **R20** (demos
> quality overhaul — PR #488) and **R21** (new "turnkey three" demos:
> asteroids / Milky-Way-dust gsplats / Dip-C genome — branch
> `demos-turnkey-three`, no PR yet). Sequenced ahead of R17/R19 in step 4:
> finalize the demos, then migrate their (incl. R21's new) heavy data to Zenodo,
> then capture the gallery. R21 adds new git-LFS data, so it should be folded
> into R17 rather than growing LFS further.

---

## Infrastructure & Polish

4 - ~~**Cache eviction policy**~~: **DONE.** Root cause found & fixed: L2 (OPFS) eviction was gated only on the configured `maxSize` (default `l2MaxSizeMB: 2048` → 2 GB), but the browser-granted OPFS quota is often far smaller. The quota gate in `OPFSStore.doSet` rejected writes (counted as `quotaWriteSkipped`) long before `totalSize` reached 2 GB, so the maxSize-based eviction loop never ran — the LRU froze holding old entries and silently dropped new ones (worst on Firefox/private-mode/small disks; invisible on roomy Chrome). Fix: `doSet` now evicts LRU entries on quota pressure (not just maxSize pressure) and re-checks, since deleting files genuinely frees quota. Bounded by index size with a no-progress guard. Regression test: `tests/unit/cache/opfs-eviction-quota.todo4.test.ts`. (L0/L1 size-gated eviction was already correct.)

7 - **Theme layout consistency**: All Luxar UI themes should differ only in colors, transparency, and visual effects — never in the size or layout of panels and their components. This ensures a consistent user experience across themes.

8 - **Panel visibility configuration**: Allow configuring which panels are visible (Logs, Rendering Controls, Data Monitor, Dimensions, etc.) from the Python side. Optionally lock panel visibility to enforce a particular look and prevent user modifications.

9 - ~~**UI ergonomics**~~: **DONE** (#432, R8). Always-visible left **control rail** — one icon per panel (Help/Dimensions/Rendering/Layers/Data monitor/Datasets/Recording/Screenshot/Logs/View options/Performance), each firing the same command as its shortcut, with tooltips, event-driven active-state, collapse, idle-dim, first-run hint, and full theme integration. Next slice: panels dock into a tray beside the rail (Concept A step 2).

## Bugs

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

24 - **Depth sorting for proper alpha blending** (**deferred post-publication/release**, re-confirmed 2026-07-11 — see R10): Sort transparent geometry (Points, Lines, GSplats) back-to-front per frame so semi-transparent elements composite correctly. Without depth sorting, overlapping translucent primitives blend in submission order rather than depth order, producing incorrect colors and visible artifacts depending on view angle.

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
