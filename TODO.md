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
  `packages/luxar/src/luxar/demos/data/**` — **~468 MB across 32 files today**
  (re-measured 2026-07-15; the old "~237 MB" was stale — the data nearly doubled),
  dominated by `gsplats_celegans` (75 MB), `desi_galaxies` (72 MB),
  `gsplats_cmu1_pathology` (122 MB / 3 ch), `milky_way_gaia_3m` (42 MB), and the
  `3d_umap_coords_human` parquet (36 MB). (`docs/images/**` adds 42 MB but stays
  in-repo — see Scope note.) This does not scale: newer/larger datasets can't
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
    `luxar.utils.data_fetch` helper that pulls a named dataset from a Zenodo
    record (verifying a checksum) into the cache on first run.
  - **Plan (incremental, largest-first, keeps LFS as fallback until proven):**
    (1) ✅ **DONE** — shared fetch helper + a committed manifest pinning each
    dataset's Zenodo record/URL + checksum + expected version; (2) create the
    "Luxar demo datasets" Zenodo deposition, upload current files; (3) repoint
    demos at the helper one at a time; (4) `git rm` the migrated files and drop
    their `.gitattributes` LFS globs. Future large datasets land as **new Zenodo
    versions**, with the manifest pinning what each Luxar release expects.
  - **Step 1 landed.** `packages/luxar/src/luxar/demos/data_manifest.json`
    (note: *not* under `demos/data/`, which packaging excludes wholesale) +
    `luxar.utils.data_fetch` (`ensure_dataset`, `load_dataset_gsplats`) +
    `scripts/gen_data_manifest.py`, gated by `hatch run check-data-manifest`.
    All 23 datasets are classified and licensed; every record ID is still null,
    so the fetch path is dormant and demos run off the in-repo LFS copy. No demo
    is migrated yet.
  - **⚠ Step 4 has a licensing trigger, not just a size one.** `gsplats_tribolium`,
    `gsplats_acto3d_heart`, `gsplats_tng_cosmic_web`, and `milky_way_gaia_3m` are
    `local-compute` ("cannot redistribute even the derived product") yet their
    derived files are committed in LFS **today** (see `demos/data/README.md`). All
    four must be `git rm`-ed before the repo goes public, independently of the
    Zenodo upload — and those demos must *not* be migrated to the fetch helper (it
    returns `None` for a `local-compute` dataset, which would silently start a
    from-scratch rebuild instead of loading the file that is right there).
  - **License audit — DONE (web-verified 2026-07-15).** A gsplat fit / point
    catalog is a *derived* product (lossy transform, not the raw voxels/pixels),
    which is broadly redistributable — but "derived" does **not** launder three
    hazards: ShareAlike, Non-Commercial, and access-gate/no-license. **We only
    ever redistribute the derived/processed product, never the raw source.** The
    verified per-dataset verdict splits the datasets into three buckets:
    - **Bucket 1 — regenerate client-side, NOT on Zenodo:** the ~17 procedural
      demos (no data file at all) + `dipc_genome` (CPU polyline build, no GPU;
      already has an auto-download+rebuild fallback). Nothing to migrate.
    - **Bucket 2 — redistribute the derived product on Zenodo,** but under **three
      different license stamps** (so a single blanket-licensed record is
      impossible — group records by license family and pin per-dataset license in
      the manifest):
      - *Clean CC0 / CC-BY / public-domain (attribution only):* kidney (CC0),
        cmu1 (CC0), cryoem/EMDB (CC0), ct_totalsegmentator (CC BY 4.0),
        milkyway_dust (CC BY 4.0), desi (CC BY 4.0), celegans (CC BY 4.0),
        organoid + dapi (IDR idr0062, CC BY 4.0), visible_human_head (US public
        domain — NLM license gate retired Jul 2019), cells3d (skimage), census +
        multiome UMAP (our derived coords over CC-BY sources).
      - *CC BY-**SA** (derived product MUST be relicensed CC BY-SA 4.0):*
        `gsplats_zebrafish` (Zenodo 1211599), `gsplats_opencell_map4` (CZ Biohub,
        via AWS Open Data registry — SA, not plain CC-BY as once assumed).
      - *NEW heavy timelapses computed on obsidian (not in LFS — upload straight
        to Zenodo, never commit; decision 2026-07-21):*
        - `gsplats_4d_neuromast_2ch` (~250 MB, membranes + nuclei channels) —
          author known to us → obtain CC-BY permission. The demo
          (`demo_gsplats_4d_neuromast_2ch.py`) already EXISTS and documents this
          exact pending upload ("not yet hosted… upload the two `.gsplats.zarr`
          to the demo data host and switch `load_neuromast_gsplats` to fetch").
        - `h2afva` zebrafish timelapse — **our own data** (Royer lab) → license
          CC-BY. Fits on obsidian: 51tp ≈ **2.9 GB**, 253tp ≈ **16 GB**. ⚠️ Size
          decision needed: 16 GB is heavy for fetch-on-demand — likely ship the
          51tp (or a culled/downsampled cut) as the demo and archive the full
          253tp separately. **Needs a new demo** (strong R10a / timelapse-showcase
          candidate; there is no h2afva demo yet).
    - **Bucket 3 — CANNOT redistribute even the derived product → SHIP THE DEMO
      WITH FETCH-RAW-AND-PROCESS-LOCALLY (not dropped; no data on Zenodo):**
      - `gsplats_tng_cosmic_web` — IllustrisTNG is access-gated (account + API
        key) and no license grant exists on authoritative pages (a CC-BY claim was
        an unverified search artifact); citation-request model only.
      - `gsplats_acto3d_heart` — repo MIT covers *software only*; the sample data
        has no license → default all-rights-reserved.
      - `gsplats_tribolium` — license CONFLICT: the Cell Tracking Challenge origin
        forbids cloning "or their parts" and requires permission for non-CTC use,
        while the Zenodo re-host (5270303) is CC BY 4.0 applied by an uploader who
        may lack authority. Local fetch-and-fit (or seek CTC permission) until
        resolved.
      - `milky_way_gaia_3m` — CC BY-**NC** 3.0 IGO (**non-commercial**). Decision
        2026-07-21: **do NOT host it** — the NC clause is incompatible with a
        cleanly-reusable demo-data host. **Still to build:** the Gaia demo will
        query the ESA Gaia archive and build the point cloud on the user's
        machine on first run, caching to `~/.cache/luxar/` (compute once, stays
        cached), with the mandatory ESA/Gaia/DPAC acknowledgement. No GPU needed
        (it's a point cloud, not a fit) — only a catalog query. Today
        `demo_gaia_milky_way_3m.py` still loads the committed
        `data/milky_way_gaia_3m.zarr.zip` and has no TAP query path (the archive
        query lives only in its docstring, describing how that file was
        produced), so this is pending work like the neuromast upload above.
      The gsplat cases total only ~35 MB, so the cost is a GPU-gated first run for
      those demos, not storage; Gaia needs only an archive query + CPU build.
      (Optional: email tng/acto3d/tribolium sources for written redistribution
      permission, or swap in a redistributable alternative, to promote them into
      Bucket 2 later.)
    - Full evidence + source URLs recorded in the license-audit memory.
  - **Fetch infra — BUILT (reversible piece, 2026-07-22; branch `r17-data-fetch-helper`).**
    - `demos/data_manifest.json` — the single source of truth for all 23 datasets
      (bucket + per-dataset license + real sha256/bytes from git-LFS oids),
      generated by `scripts/gen_data_manifest.py` (curated license metadata merged
      with LFS checksums; `--check` guards drift in CI).
    - `luxar.utils.data_fetch.ensure_dataset(name, variant=…)` — resolves files
      **cache → in-repo LFS → Zenodo**, dormant on the Zenodo leg until a record
      URL is set (so demos keep working on the in-repo fallback today). Raises
      `LocalComputeDataset` for `local-compute`/`regenerate` datasets.
    - **Size variants:** `h2afva` ships as two variants in one record — `51tp`
      (~2.9 GB, **default** the demo fetches) and `253tp` (~16 GB, opt-in via
      `variant="253tp"`). Avoids a 16 GB first-run download over Zenodo's
      best-effort, unguaranteed bandwidth while still hosting the full timelapse.
    - Unit tests in `utils/tests/test_data_fetch.py` (deliberately not counted
      here — the number rots every time a test lands); demo-import tests stay
      green; ruff clean.
  - **Migration runbook (irreversibility rules).** Publishing a Zenodo record is
    **permanent** (no self-delete; files immutable — edits become new versions).
    So: (1) rehearse on **`sandbox.zenodo.org`** first — a published Sandbox record
    serves files over the identical `/records/<id>/files/<name>?download=1` URL, so
    point a record's `base_url` there, fetch + checksum-verify via
    `ensure_dataset`, and only then repeat against production — a
    `ZENODO_SANDBOX_TOKEN` is needed to *create* that Sandbox record, not to
    download from it (the fetch leg is unauthenticated: `zenodo_file_url` builds a
    plain public URL and no code reads a Zenodo token today);
    (2) keep records as **drafts** (deletable) until verified; (3) **record
    grouping is a one-way door** (can't split/merge after publish) — hence h2afva
    stays its own record. Downloads go through `download_with_checksum` —
    `robust_download`'s resume plus a `verify_file_checksum` sha256 gate that
    rejects a mismatched file (Zenodo gives no bandwidth SLA).
  - **CI rule (do NOT fetch Zenodo in CI).** Zenodo is a free best-effort service;
    CI must never pull datasets from it (discourteous + undocumented throttling +
    flaky). CI stays on the in-repo git-LFS copy (or skips network / uses the
    `regenerate` path). Per-user caching means one fetch per machine; the
    `data_fetch` fallback order already makes the in-repo copy win when present.
  - **Scope note:** only *heavy processed datasets* move. Small README/doc images
    (`docs/images/**`) must stay in-repo so GitHub renders them (see R19).
  - **Also here (from R16):** the **64 MB** `luxar-paper`
    `supp_doc/splat_count_vs_quality/splat_count_vs_quality.pdf` (LFS, ~10× any
    other PDF) embeds many 2–8 MB slice montages uncompressed — rasterize/
    downsample them at build time to shrink it. A manuscript-repo LFS-weight
    item folded in alongside the software-repo dataset migration.

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
- **R10 [LAUNCH] — Depth sorting for alpha blending** (see detailed item
  **#24** and the full plan in `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md`).
  Translucent geometry composites in submission order → view-dependent artifacts.
  **RE-PROMOTED to pre-release [LAUNCH] (decision 2026-07-15):** correct
  order-dependent (`normal`-mode) gsplat transparency is now wanted *before*
  release, both to raise general rendering quality and — the load-bearing reason
  — to **incorporate "normal" (alpha-over, surface-like) gsplat datasets into the
  demos/gallery**, which cannot render correctly without it. This pulls three
  planned phases onto the critical path and adds a demo item (see R10a below):
  - ✅ **Phase 0** (normal-mode premultiplied alpha) — MERGED (#511, squash
    `e1e079e6`).
  - ✅ **Shader-symmetry B9a–c** + native-WebGPU perf validation — MERGED (#523).
  - ✅ **Phases 1–3 ALL MERGED** (#535 texture-backed storage, #540 SortWorker +
    sort-at-commit, #553 camera-triggered re-sorts), plus the cross-node
    ordering follow-ups (#563 per-part renderOrder, #565 exact BSP tile order,
    #575 one global cross-node renderOrder domain).
  - ✅ **2026-07 review + hardening campaign MERGED** (16 PRs, #568–#596): six
    lifecycle/consistency bugs (capacity-clamp vs sorted ordering, null-camera
    recovery, demotion worker-release, NaN-safe ordering, worker-less
    renderOrder), colormapped Points/Lines Intensity/Offset, front-most-wins
    picking for `normal` gsplats, permutation retention on same-count commits,
    plus structural cleanups (render-order submodule, committedData accessors,
    parity-harness split) — see `CHANGELOG.md` July 2026 and the spec's status
    block.
  - 🔄 **Phase 4** (partial texture uploads): **Stages 1–2 LANDED for all three
    geometries.** Stage 1 (slack elimination) — `writeSplatTexels` registers
    per-row `updateRanges` so only live rows upload (measured 33–59% less on
    `gsplats_4d_neuromast_2ch`, classic WebGL, pixel-identical). Stage 2 (append
    fast path) — a ladder-refinement commit that extends the committed prefix
    writes/uploads only the new suffix; prefix trust via a forward-chained
    lineage `WeakMap` (shared `types/prefix-lineage.ts`) + `viewStatesEqual`/
    generation, no projection-kernel change, with a context-restore full-dirty
    hook. Points/Lines ride the same gate through
    `writeInterleavedAttribute(…, { fromInstance })` (order-preservation
    verified; plus an optional-field presence conjunct and the Lines
    missing-sharpness 0.5-fill fix). Stage 3 (WebGPU range parity):
    **MEASURED-REJECT 2026-07-25** — built, unit-tested, and rejected at the
    ≥10% in-app bar (archived as a closed PR; numbers + revisit criteria in
    the spec §7).
  - ⏭️ **[POST] residue from the campaign reviews:** instance-based coordinator
    DI (only if multi-instance embedding lands), front-most picking in MIXED
    normal+additive scenes (single shared pick depth buffer — documented in
    `rendering/picking/README.md`), cross-TYPE (points/lines vs gsplat)
    transparent interleaving.
- **R10a [LAUNCH] — "Normal" (surface-like) gsplat demos + gallery datasets.**
  New item created by R10's promotion. Once Phase 2/3 land, build one or more
  demos that showcase `normal`-mode alpha-over gsplats (the realistic, occluding,
  surface-like look — distinct from the additive/luminous glow the current demos
  use), with tests. These are strong gallery candidates (R19). **Feeds R17
  directly:** their fitted datasets are new heavy data — land them to Zenodo from
  the start rather than growing git-LFS. Gated on R10 Phase 2 (correct at rest);
  Phase 3 for good orbit captures.
- **R11 [LAUNCH] — README/landing pass.** ✅ **Mostly done** (2026-07-01):
  audited the quick-start end-to-end — `luxar demo` generates + renders
  flawlessly (10k-pt Lorenz), every documented Python snippet runs, and all
  gallery images + doc links resolve. Fixed 3 stale/confusing README spots
  (#429): sharpness range (`0.5-10` → normalized `0-1`), `gsplat lod`
  signature (→ `--recipe {…}`), and the redundant two-terminal "View it"
  block (→ single `luxar serve … --viewer --open`). **Remaining:** verify a
  truly-fresh-machine `make setup-dev`, and optionally regenerate the gallery
  media (`make generate-gallery`).
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
  generate-readme-images` for the stills (Playwright captures of the live
  viewer → `docs/images/readme/*.png`) and `make generate-gallery` for the
  orbit videos (→ `docs/images/gallery/*.{webm,webp}`, gitignored staging;
  curated picks get copied into `docs/images/readme/gallery/`).
  - **Curate a stronger gallery** from the newer/better datasets (H&E pathology
    gsplats, 4D *C. elegans* tracking, organoid multichannel, Gaia) — decide
    which few best convey the range (volumetric splats, nD navigation, scale).
  - **Regenerate** stills + short loops via the Playwright pipeline; refresh the
    README gallery section + captions; ensure everything renders on GitHub.
  - ✅ **Gallery harness built** (2026-07-13): a manifest-driven capture tool that
    produces a still PNG **and** a seamlessly-looping orbit video (WebP+WebM) for every
    demo in one pass — auto-center + fill-to-frame, chrome hidden, robust
    screenshot-based auto-exposure (percentile target; per-demo override). SSOT
    is `scripts/gallery/manifest.json` (~20 curated demos spanning
    Points/Lines/GSplats + synthetic/astronomy/microscopy/medical/genomics);
    `scripts/gallery/generate_gallery_datasets.py` builds the datasets and
    `packages/luxar-viewer/.../generate-gallery.spec.ts` captures. Run:
    `make generate-gallery` (or `ONLY=id`). Output → `docs/images/gallery/`
    (gitignored staging); copy curated picks into `docs/images/readme/`. Per
    user calls: **DESI included** (astronomy), **cmu1_pathology dropped**. Next:
    run the full sweep (incl. heavy datasets: `galaxy`/organoid/celegans need
    generation), curate the winners, wire them into the README gallery table.
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
- **R16 [LAUNCH] — Manuscript repo hygiene.** ✅ **DONE** (2026-07-12). Scoped,
  executed, and verified:
  - **Regenerate-all / `--floor`:** resolved to **keep legacy no-floor + document**
    (no GPU re-run — a floor=auto-vs-none measurement showed floor=auto only
    *lowers* PSNR-vs-original, a reference-mismatch, so a full regen would weaken
    the numbers; the committed legacy numbers already are the `--floor none`
    numbers and stand). See the decision note at the end of section D. The
    manuscript fit harnesses were pinned to `--floor none` (luxar-paper #8) and a
    Methods paragraph documents the choice (luxar-paper #7).
  - ✅ **Orphan pruned:** deleted `preprint/figs/quantitative_analysis/cross_validation.pdf`
    (referenced by no `.tex`; CV lives in `quantitative_analysis.pdf` + suppfig
    `cv_all.pdf`) — luxar-paper #9.
  - ✅ **Compression figs — not stale, single-source verified:** all four
    `compression_*.pdf` are live `\includegraphics` (SD7 is cited), and
    `preprint/figs/suppfig/compression.pdf` is **byte-identical** to
    `supp_doc/compression_comparison/results/compression.pdf` (in sync via
    `build_figures.sh`, no divergence). The old "prune the compression variants"
    note was stale and is retired.
  - ✅ **SD6 uses `sweep_v6`, not placeholders:** `sweep_v6.json` (216 KB) is
    committed and wired as the default input (`sweep_latest.json → sweep_v6.json`);
    the committed SD6 figs carry no placeholder marker and were committed in the
    same commit as the data. (The `_emit_placeholders` path fires only when the
    JSON is absent.)
  - ➡️ **64 MB `splat_count_vs_quality.pdf` bloat moved to R17** (it's an LFS
    storage concern — the PDF embeds many 2–8 MB slice montages uncompressed;
    rasterize/downsample at build time, alongside the Zenodo/LFS work).
  - Bonus (beyond R16 proper): a full Methods fact-check corrected ~12 code-vs-text
    mismatches with inline `% source:` comments, and removed the stray 4th pywt
    noise-floor estimator so the analysis matches the "three estimators" claim
    (luxar-paper #7, merged).

### D. Sequencing (suggested order, parallelizable across tracks)

1. ✅ **Stabilize `main`** — R5 (branches merged/drained) + R6 (hygiene:
   `delme/`/`test-results/` are gitignored, won't ship) **done**; `main` clean,
   no open feature branches, dependabot drained, release pipeline landed (#417).
2. **Decide versioning & package** — R1 → R3 (PyPI dry-run / TestPyPI) in parallel
   with the day-one polish (R8/R9/R10/R11).
3. **Post the preprint** — R13 ✅ / R14 ✅ / R16 ✅ → bioRxiv → obtain DOI → R4
   (wire citation back into the repo).
4. **Engineering spine — depth sorting → new demos** (the engineering long pole,
   serial): **R10** Phase 1 → Phase 2 → Phase 3 (one PR per phase, `main`
   shippable after each), then **R10a** (build the normal-mode gsplat demos +
   fit their datasets). Land **PR #534** and refresh green baselines first. This
   gates the finalized demo set (R10a's demos + R20/R21's, all done by now) that
   R19's gallery is captured from.
5. **Public-repo readiness** (parallel with the preprint track AND the R10 spine,
   before the repo goes public). **R20** (demos quality overhaul) ✅ MERGED (#488)
   and **R21** (turnkey-three demos) ✅ MERGED (#496) are done. Remaining:
   **R17** (retire git-LFS heavy datasets → Zenodo) — **split it**: build the
   fetch helper + `manifest.json` and migrate the *existing* datasets early (this
   step, parallel with R10), then **append R10a's new normal-gsplat data as new
   Zenodo versions** once those demos exist; **R18** (docs content pass + confirm
   the GitHub Pages site is publicly viewable) — fully independent, run any time;
   **R19** (README refresh + regenerated gallery) — capture **once**, after R10a's
   demos exist and R17's fetch path is live (fold in the R11 fresh-machine
   `make setup-dev` verify). Order within the step: **R17-infra (early) → R10a
   data → R19**; R18 anytime.
6. **Cut the release** — final dependabot/branch drain → R1 (`make set-version`)
   → PR + merge → R2 (`make release-check` → `make release`: tag + GitHub release,
   fires PyPI OIDC) → flip PyPI live (R3) + extras check on a clean machine →
   R3-npm one-time bootstrap → flip repo public (confirm R18 Pages live) → release
   notes from CHANGELOG `[Unreleased]` → announce.
7. **Post-launch backlog** — R10 Phase 4 (partial appends), R12, R15, and the
   remaining Future/Exploratory items below (incl. #25 advanced LOD
   refinements). (#24 depth sorting Phases 1–3 moved *into* the pre-release
   spine, step 4.)

> Update 2026-07-01: R5/R6 (clean `main`) and the day-one polish R8/R9/R11 have
> landed (R8 control rail merged #432); R10 (#24 depth sorting) is **deferred
> post-release**. The critical path is now the **preprint** (R13/R14/R16 →
> bioRxiv → DOI → R4) and the **release cut** (R1 version bump → R2 tag →
> R3/R3-npm go-live).

> Update 2026-07-11: R13 (Methods specificity) is **done** (#466) and
> dependabot is re-drained (R5 note above). All release *tooling* is built and
> validated — what remains on the critical path is **execution**, in order:
> **(1) preprint track** — R14 (consumer-GPU timing sweep) ✅ **done** and R16
> (manuscript repo hygiene) ✅ **done** (2026-07-12; `--floor` resolved,
> orphan pruned, compression single-source + SD6 verified, Methods fact-checked;
> 64 MB PDF moved to R17). Preprint track is now clear to **post → DOI → R4
> (CITATION.cff)**, once the author does a final read. **(2) day-one leftovers** — R11 remainder
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

> **Decision 2026-07-12 — `--floor`: KEEP LEGACY NO-FLOOR FOR THE PAPER, for
> pragmatic (not scientific) reasons + DOCUMENT (no GPU re-run).** The initial
> call was "full re-run with `floor=auto`"; a quick measurement changed the
> *how*, not the science.
> - **Scientifically, floor suppression is the right thing** (author's view):
>   the constant background pedestal is *not real signal*, so removing it before
>   fitting is principled, and the *correct* way to score a floor-suppressed fit
>   is against a **floor-suppressed reference** (floor-recon vs floor-original).
> - **The measured −5.86 dB (kidney_dapi 31.77→25.91; organoid +0.01) is a
>   reference-mismatch artifact, NOT evidence floor is worse:** it scores a
>   background-free reconstruction against the *original, pedestal-bearing*
>   volume, penalising the fit for correctly dropping non-signal. Under the
>   principled floor-suppressed reference, floor would be fair (and appropriate).
> - **Why keep no-floor for the paper anyway (pragmatic):** adopting floor
>   properly means switching the evaluation to background-relative PSNR — a
>   protocol + narrative change (and re-checking the blind-spot CV story) that
>   isn't worth doing right before bioRxiv. The committed legacy numbers *are*
>   the `--floor none` numbers (committed kidney 31.81 dB ≈ floor=none 31.77) and
>   are internally consistent (original-referenced throughout), so they stand.
> - **Tool default is confirmed correct and stays:** for general CLI use the
>   floor should always be removed by default (background isn't signal), so
>   `gsplat fit`/`cal` keep `--floor auto` — **do not change the shipped
>   default.** The `--floor none` pin below is a *paper-only* deviation for
>   original-referenced comparability, not a statement about the tool.
> - **Resolution (no re-run):** (1) numbers stand as-is; (2) **pin the manuscript
>   fit harnesses to `--floor none`** (`run_analysis`/`run_convergence`/
>   `run_noise2self`/`progressive`/`loss_comparison`/`run_noise_floor`) so a
>   future re-run stays reproducible instead of silently inheriting `floor=auto`;
>   (3) **Methods paragraph** stating the benchmarks use `--floor none` with
>   original-referenced PSNR for comparability, while `floor=auto` (the shipped
>   default) is the more principled fit for real use (background isn't signal),
>   and noting background-relative evaluation as appropriate future work.
> - **Deferred (post-bioRxiv / journal / future work):** the floor-suppressed-
>   reference evaluation, and the small open check of whether floor improves the
>   blind-spot CV / K\* selection (a `cal`-sweep on a few datasets, not run_all).
> (Harness pinning + Methods paragraph land as a luxar-paper PR.)

> **Decision 2026-07-15 — DEPTH SORTING SHIPS PRE-RELEASE (R10 [POST] → [LAUNCH]).**
> Correct order-dependent (`normal`-mode) gsplat transparency is now launch-gating,
> to raise general rendering quality *and* — the load-bearing reason — to
> incorporate "normal" (alpha-over, surface-like) gsplat datasets into the
> demos/gallery, which cannot render correctly without it. Effect on the plan:
> - **New pre-release work** (see R10 / R10a / item #24): depth-sorting Phases 1
>   (texture storage, high blast radius), 2 (SortWorker), 3 (live re-sort) — all
>   [LAUNCH]; plus R10a (new normal-mode gsplat demos, gated on Phase 2/3). Phase 0
>   + shader-symmetry already merged (#511/#523). Phase 4 stays [POST].
> - **The release now has TWO parallel long poles, not one:** (A) the **preprint**
>   — author read → bioRxiv → DOI → R4 (external latency, days; start immediately,
>   finish last); and (B) the **engineering spine** — depth-sorting Phases 1–3 →
>   R10a demos → (data) R17 → R19 gallery. The cut waits on whichever finishes
>   last. R18 docs, R17-infra (migrate *existing* data), and R6/R7 hygiene are
>   cheap parallel filler that should all be green before either long pole lands,
>   so the cut (step 6) is purely mechanical.
> - **R17 is split** so it doesn't block the spine: build the fetch helper +
>   manifest and migrate current datasets early; append R10a's new data as new
>   Zenodo versions once those demos exist.
> - Full phased plan + risk register: `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md`.

---

## Infrastructure & Polish

4 - ~~**Cache eviction policy**~~: **DONE.** Root cause found & fixed: L2 (OPFS) eviction was gated only on the configured `maxSize` (default `l2MaxSizeMB: 2048` → 2 GB), but the browser-granted OPFS quota is often far smaller. The quota gate in `OPFSStore.doSet` rejected writes (counted as `quotaWriteSkipped`) long before `totalSize` reached 2 GB, so the maxSize-based eviction loop never ran — the LRU froze holding old entries and silently dropped new ones (worst on Firefox/private-mode/small disks; invisible on roomy Chrome). Fix: `doSet` now evicts LRU entries on quota pressure (not just maxSize pressure) and re-checks, since deleting files genuinely frees quota. Bounded by index size with a no-progress guard. Regression test: `tests/unit/cache/opfs-eviction-quota.todo4.test.ts`. (L0/L1 size-gated eviction was already correct.)

7 - **Theme layout consistency**: All Luxar UI themes should differ only in colors, transparency, and visual effects — never in the size or layout of panels and their components. This ensures a consistent user experience across themes.

8 - **Panel visibility configuration**: Allow configuring which panels are visible (Logs, Rendering Controls, Data Monitor, Dimensions, etc.) from the Python side. Optionally lock panel visibility to enforce a particular look and prevent user modifications.

9 - ~~**UI ergonomics**~~: **DONE** (#432, R8). Always-visible left **control rail** — one icon per panel (Help/Dimensions/Rendering/Layers/Data monitor/Datasets/Recording/Screenshot/Logs/View options/Performance), each firing the same command as its shortcut, with tooltips, event-driven active-state, collapse, idle-dim, first-run hint, and full theme integration. Next slice: panels dock into a tray beside the rail (Concept A step 2).

## Bugs

27 - ~~**Demos can't be stopped with Ctrl-C; a new demo shows the old one**~~: **DONE** (2026-07-24, #652). `luxar demo run` spawned a 3-level tree (`demo run` → demo script → `luxar serve` uvicorn) with no process-group isolation or owned teardown, so Ctrl-C orphaned the server on ports 8000/5173; `pick_port` then auto-incremented and the stale browser tab kept showing the old scene. Fix: new stdlib-only `luxar/utils/process.py::run_child_process` runs the child in its own session (`start_new_session`) and, on any exit, tears the whole subtree down with escalating SIGINT → SIGTERM → SIGKILL in a `finally` (SIGTERM/SIGHUP routed in too; a second Ctrl-C jumps straight to SIGKILL). Wired into `demo_run`/`demo_run_all` (isolate the group) and `launch_viewer` (stay in the group so the group-kill cascades). Verified with a real foreground Ctrl-C via a PTY: exits 130, zero survivors, ports freed. Also folded in demo-CLI robustness (installed-wheel guard, clean `DEMO_META` errors, run-all GPU/large-download skips + `--include-gpu`/`--max-download-mb`, corrupt-download classification, honest cache-clear totals, no `datasets/` dir creation on `demo list`).

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

24 - **Depth sorting for proper alpha blending** (**RE-PROMOTED to pre-release [LAUNCH]** 2026-07-15 — see R10): Sort transparent geometry (Points, Lines, GSplats) back-to-front per frame so semi-transparent elements composite correctly. Without depth sorting, overlapping translucent primitives blend in submission order rather than depth order, producing incorrect colors and visible artifacts depending on view angle. Full phased plan (Option 3a — viewer-only, no format change): `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md`. Status: spec Phases 0–3 MERGED for gsplats (#511/#523/#553: premultiplied alpha, texture-backed storage, SortWorker, camera-triggered re-sort); partial appends (spec Phase 4 Stage 2) landed for all three geometry types. GSplats-first; Points sorting symmetry LANDED (arc PR-B, spec §8); Lines storage + sorting symmetry LANDED (arc PR-C, spec §8) — all three geometry types now share the texture storage + SortWorker machinery. Volumetric Phase 3 (points: isotropic chord-integral emission–absorption + points RGBA alpha + mandelbulb showcase) LANDED (arc PR-D); volumetric Phase 4 (lines: transverse chord integral + lines RGBA per-vertex alpha, `effectiveGeometryMode` deleted — all three geometry types now render + depth-sort real volumetric) LANDED (arc PR-E) — **ARC COMPLETE**.

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
    - **The promotion trigger, now falsifiable.** The spec's condition was "if continuous hidden spatial dims turn out to be a real use case", which nothing measured. `processMeshData` now emits one `log.info` per node when a mesh's hidden dims include a continuous one, naming the dimension and its unit (`data/scene-loader/process/data-processor-mesh.ts::noticeContinuousHiddenDim`). Hidden-and-continuous *is* the spec's "continuous hidden **spatial**" condition rather than a superset of it: `core/dimensions.py` forces a non-displayed, non-spatial dimension to be discrete, so every dimension the line can report was authored `spatial=true` — testing the flag as well would narrow nothing and would drop evidence from any scene whose metadata omits it. The name and unit are printed for the one judgement no flag can make: a dimension may be *declared* spatial and still be a time axis, where a slab is a reasonable thing to want.
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
