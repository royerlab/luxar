# Changelog

All notable changes to Luxar are documented in this file.

## [Unreleased]

### May 2026

#### Changed — Cache final polish after S1–S7 (2026-05-10)

- Added browser screenshot artifact coverage for the Cache tab's status
  badge / Cache Health layout in `cache-hardening.spec.ts`.
- Cache disabled/fallback views now render any available cache-status
  badges (for example `no-cache`, `disabled-config`, or
  `provider-missing`) above the explanatory panel instead of only
  showing badges in the full L1/L2 view.
- SceneLoader clears the per-node predictive-prefetch baseline when a
  loader update throws, so the next successful update re-baselines
  rather than extrapolating across a stale/error gap.

#### Changed — Cache recheck polish S1–S7 (2026-05-10)

Follow-up to the cache recheck at
`delme/viewer-cache-rereview-20260510-200308/`, which confirmed R1–R7
closed most prior gaps and flagged seven concrete remaining items.
S1–S7 close those.

- **S1 — CSS for the new cache UI classes.** `data-loading-monitor.css`
  gains rules for `.luxar-cache-section__metrics--cols-4` (the L2
  ERRORS card row, with a 2-column fallback under 480px),
  `.luxar-cache-status` (badge pill row), `.luxar-badge` (pill shape,
  reuses `.luxar-color--*` text-color modifiers for tint), and
  `.luxar-cache-health` / `.luxar-cache-health__{header,row,label,value}`
  (validation-mode + last-validated panel).
- **S2 — `opfs-unavailable` badge is wired and no longer dead.**
  `OPFSStore.getStats()` exposes `available: boolean` (true when
  `opfsRoot !== null && !disposed`). `MultiLevelCachingStore.getStats().health`
  surfaces `opfsAvailable: boolean`, treating caching-disabled modes
  as `true` (no L2 expected). `aggregateCacheMetrics` emits the badge
  only when explicitly `false`. Older providers that omit the field
  stay silent (treat-as-true).
- **S3 — L0/L1/L2 hit-rate no-data color is dimmed (not red).** New
  `getCacheHitRateColorClassWithGuard(rate, totalAccesses)` returns
  dimmed when `totalAccesses === 0`; otherwise delegates to the
  existing thresholds. Used by both the initial render in
  `renderCacheContent` and (for L0/L1 parity) the incremental
  `updateCacheTab`. Fixes the first-paint red-flash on empty caches.
- **S4 — `clearOnInitCount` telemetry + a real `?clear-cache` E2E
  assertion.** `MultiLevelCachingStore` counts each `?clear-cache`
  invocation in `init()`. Surfaced via `getStats().clearOnInitCount`
  and the cache-API snapshot. `cache-persistence.spec.ts` now asserts
  `clearOnInitCount > 0` after navigating with `?clear-cache` —
  proves the clear path executed, replacing the trivially-true
  `l2.size >= 0` assertion.
- **S5 — Bandwidth compaction test now exercises the production
  path.** The R5 test stub (seeded stale entries, never asserted
  compaction) now drives a real `getResult()` fetch so the
  production push site's `start > length/2` check fires; asserts
  `bandwidthWindowStart` resets to 0 and the array shrinks.
- **S6 — Predictive prefetch uses per-loader derived view-state.**
  `SceneLoader._dispatchPredictivePrefetch` no longer fans one
  global view-state to every loader. The dispatch now lives inside
  each loader-task branch (Points / Lines / GSplats) and uses the
  per-node `derived.viewState` computed by `deriveNodeViewState`.
  `extend_to_all`-skipped nodes no longer receive prefetch hints
  they would have skipped on demand. A per-loader `Map<path,
  ViewState>` tracks prev state; cleared on `loadScene` / `dispose`
  and on skip transitions.
- **S7 — `evictionsPerMin` API break recorded.** As part of the R1–R7
  batch (commit `2f7feaaf`), the deprecated `CacheMetrics.evictionsPerMin`
  alias was removed in favor of `evictionsTotal`. External consumers
  reading `evictionsPerMin` now see `undefined`; this CHANGELOG note
  records the removal explicitly so the API break is discoverable.

**Result**: cache UI is fully styled, every documented status badge
is actually emitted, the predictor honors per-node tolerance/skip
semantics, the `?clear-cache` E2E proves real observable behavior,
and the bandwidth-compaction test exercises the real production
path. Tests: 89 cache unit + 86 monitor unit + 1 strengthened E2E.

#### Changed — Cache re-review remediation R1–R7 (2026-05-10)

Follow-up to the cache re-review at
`delme/viewer-cache-rereview-20260510-140614/`. The Phase 1–8
hardening had already landed (lifecycle/dispose, OPFS races,
in-flight coalescing, content-hash/TTL validation, Points/Lines
prefetch parity); the re-review surfaced remaining UI/observability
gaps and a handful of edge-case test holes. R1–R7 close those.

- **R1 — config validation for new cache fields.**
  `validateConfig` rejects bad `cache.opfsOperationTimeoutMs` (NaN
  / zero / negative / Infinity) and `cache.externalDatasetTtlMs`
  (NaN / negative; `null` stays valid). Prevents cryptic OPFS
  timeouts or always-expired TTLs.
- **R2 — L2 hit-rate now patches live in the cache tab.**
  `updateCacheTab` patches `l2-hitrate` + `l2-hitrate-sub` and the
  color class alongside `l2-size` / `l2-io`. The L2 hit-rate card
  no longer freezes after the initial render.
- **R3 — Status badges, Cache Health, L2 error counters.**
  The cache tab now shows a pill row of `CacheStatusBadge` chips
  (`cache-enabled`, `quota-constrained`,
  `unvalidated-external-dataset`, …), a dedicated **Cache Health**
  section with validation mode + last-validated timestamp, and an
  inline `ERRORS` card on the L2 section summing the four OPFS
  health counters. `CacheMetrics.l2` carries those counters so
  debug snapshots and E2E can assert on them without reaching into
  the provider. `cache/README.md` gains a "Cache Status Badges"
  section.
- **R4 — Direction-aware predictive prefetch wired into
  `updateView`.** `SceneLoader.updateView` extrapolates one step
  from the previous → current view-state delta and fires
  `prefetchChunks(predicted)` on every loader (Points, Lines,
  GSplats) for the predicted state, in a microtask so it never
  blocks commit. Pure helper `predictNextViewState` is unit-tested
  in isolation; `dispatchPredictivePrefetch` handles iteration +
  error swallowing. State resets on dataset switch / dispose.
- **R5 — Bandwidth window start-index pruning.**
  `MultiLevelCachingStore.bandwidthWindow` no longer uses
  `Array.shift()` (O(n)); a `bandwidthWindowStart` index advances
  forward with amortized O(n) compaction when the dead prefix
  exceeds half the array. Numeric output unchanged.
- **R6 — Test gap closure.**
  - **R6a**: Unicode OPFS key roundtrip (µ, 通, é, base64-special
    characters, slash-traversing keys).
  - **R6b**: Demand-while-prefetch-in-flight — one underlying fetch
    shared via `pendingGets`; demand counter increments exactly once;
    L1-warm path lets demand hit L1.
  - **R6c**: 4D node-type cache parity — cache stats populated, slice
    navigation produces fresh L1 activity.
  - **R6d**: OPFS quota-clears-then-write-succeeds + concurrent
    quota-skipped writes don't corrupt the index.
  - **R6e**: Negative `totalSize` in persisted metadata is clamped /
    recomputed from `entries[]` (defensive hardening in
    `OPFSStore.loadMetadata`).
- **R7 — Real-browser L2 persistence E2E.**
  New `cache-persistence.spec.ts` asserts L2 entries survive a full
  page reload in Chromium and that `?clear-cache` wipes L2 across a
  reload. Skipped on Firefox/WebKit (OPFS persistence semantics
  across Playwright contexts are unreliable there).

**Result**: 4813 unit tests pass (up from 4662 — +151 new tests
across R1–R7). 9 cache test files, 272 cache unit tests. Bandwidth
pruning, metadata corruption recovery, and config validation become
regression-locked.

Deliberately deferred (re-review §1.6 "future work"): cross-browser
OPFS comparison harness, sustained-load cache performance benchmarks,
runtime cache profiles, per-array memory breakdowns, origin-wide
cache manager UI.

#### Changed — Viewer code-review recheck hardening (2026-05-10)

Follow-up pass against the recheck reports under
`delme/viewer-code-review-recheck/`. Addresses W-tier findings still
open after the prior pass:

- **Constants**: `MAX_SUPPORTED_DIMS` consolidated into
  `src/config/constants.ts`; `wasm/typescript/gsplats-processing.ts`,
  `data/gsplats/gsplats-processor.ts`, and `workers/validation.ts`
  (`MAX_WASM_DIMS` alias) now share the canonical value.
- **Worker init guard**: `projectPointsTo3D` in `workers/data-worker.ts`
  rejects with `NOT_INITIALIZED_MSG` when called before `initialize()`,
  matching the existing `projectLinesTo3D` / `projectGSplatsTo3D`
  guards.
- **Material registration idempotency**:
  `MaterialManager.subscribeToDispose` now tracks already-subscribed
  materials in a `WeakSet` so re-registering the same instance no
  longer stacks `dispose` listeners on the THREE EventDispatcher.
- **Post-processing rebuild safety**:
  `PostProcessingManager` replaces the boolean `deferRebuild` flag with
  a depth counter; `setQualityPreset` wraps its batch in `try/finally`
  so a thrown sub-setter cannot strand the counter above zero (which
  previously made all subsequent effect changes silent no-ops).
- **Shared clamp util**: `clampDPRScale` in
  `rendering/post-processing/visual-effects-handler.ts` calls the
  general `utils/clamp` helper instead of an inline `Math.min(max,
  Math.max(min, x))`.
- **Lines TS fallback note**: `data/lines/projection.ts` documents the
  TS fallback path's allocation profile as an accepted trade-off.
- **Docs**: `src/data/SPECIFICATIONS.md`, `src/data/loaders/README.md`
  rewritten to show the `runWithTimeout(name, kind, fn)` pattern
  instead of raw `getWorker()` access. `CONVENTIONS.md` gains §§12-14
  for dependency-inversion ports, error-handling discipline, and the
  disposal pattern. `packages/luxar-viewer/README.md` documents
  `LUXAR_LAUNCHER_NO_WEBVIEW=1`.

#### Changed — Viewer code-review rerun hardening pass (2026-05-10)

Multi-phase hardening of the scalar-colormap + GPU pool + blending-state
feature work, addressing every actionable finding from a six-agent code
review rerun. Phases A–J in `delme/viewer-code-review-rerun/ACTION_PLAN.md`.

**Type-safety (Phase A):**
- `PointsAttributeTypes.scalar` is now optional (`undefined` when absent)
  instead of a `'none'` sentinel; Float16Array gets a first-class dtype tag.
- `LoadedLinesData.scalars` aligned with `ScalarArray` (Float32/Float16/
  Uint8). Lines accumulator preserves Uint8 dtype natively.
- Fail-closed scalar length validation in `projectPointsTo3D` and
  `buildInstanceBuffers` — mismatch logs a warning + suppresses colormap.
- `growLinesGeometry` preserves optional `aStartScalar`/`aEndScalar`
  attributes on resize.

**Lifecycle (Phase B):**
- Material clone-vs-pool registration bug: pooled materials are
  detached from global updates before NodeFactory clone sites; clones
  take the global slot, pooled stays in the LRU cache.
- Custom colormap LUT cache: bounded LRU (16 entries), disposed
  per scene unload via `disposeCustomColormapTextures()`. Built-ins
  survive scene switches.
- Post-processing in-place effect swaps (AO/SMAA/Bloom-levels) now
  dispose the OLD effect AFTER `rebuildEffectPass`, eliminating the
  use-after-free window.
- `rebuildEffectPass` rolls back Pass A if Pass B construction fails.
- Data accumulators expose `isDisposed()`; `fill`/`ensureCapacity`
  throw with a descriptive error post-dispose.

**Performance (Phases C–D):**
- Lines worker scalar fallback now emits a one-shot warning so users
  notice the main-thread cliff.
- Accumulator growth copies only the live prefix
  (`usedCount * stride`) instead of full capacity.
- Worker-projection fallback routes through the accumulator's
  pre-sized buffers when present.
- Scalar buffers in Points + Lines accumulators are lazily allocated
  on first scalar fill or first `getScalarBuffer()` call.
- GPU byte-budget eviction is single-pass sort + walk (replaces the
  per-iteration O(N²) re-scan). Pure helper `selectBuffersToEvict`
  exposed for tests. Loop bounded by `maxPoolSize * 3` iterations.
- `estimateGeometryBytes` cached on `geometry.userData.cachedByteSize`;
  invalidated on grow.

**Shaders (Phase E):**
- Line shader: pathological near-camera wide-quad segments now early-
  discard instead of rasterizing a half-viewport quad at reduced
  intensity.
- Shared GLSL sanitize helpers (`isInvalidFloat`, `sanitizePositive`,
  `sanitizeNonNegative`) extracted to `glsl-lib.ts` and injected into
  the Points/Lines/GSplats vertex shaders.
- Documented `LUXAR_MAX_RGB_CONTRIBUTION` and `USE_COLORMAP` defines
  in the line shader header.

**API surface (Phase F):**
- Public exports for `getCompleteBlendingState`,
  `applyBlendingStateToMaterial`, `supportsScalarColormap`,
  `applyColormapTextureToMaterial`, `applyScalarRangeToMaterial`,
  `BlendingMode`, `CompleteBlendingState`.
- Predicates (`isAdditiveMode`, `isOpaqueMode`, `isMaxMode`, etc.)
  centralize the mode discriminators across materials.
- `syncPointMaterialWithGeometry` moved to `rendering/material-sync-helpers.ts`.

**Diagnostics (Phase G):**
- Array decoder broadcast-encoding error includes zarr path + encoding shape.
- `captureHDRPixels` validates the mode at runtime, falls back with a warning.
- `updateView` log differentiates supersede vs first-queue.
- Custom LUT cache validates content fingerprint on hit (defends
  against DJB2 collisions).

**Tests + Docs (Phases H–I):**
- New `blending-state.test.ts` with predicate + canonical-state tests
  including max-mode round-trip lock-in.
- Accumulator dispose/usedCount tests, scalar buffer lazy-alloc tests,
  Float16 detection tests, Lines Uint8 roundtrip test.
- Byte-budget eviction tests for the pure selector + pool integration.
- `LUXAR_ZARR_FORMAT.md` documents `has_scalars`, `scalar_data_range`,
  `colormap`, and the `colormap_lut` sibling array.
- `rendering/SPECIFICATIONS.md` documents the byte-cache + bounded
  eviction policy.

#### Changed — Viewer shader/material follow-through (2026-05-10)

Completes the remaining shader/material work in this branch, excluding
conditional Gaussian slicing for correlated nD splats.

- **Custom Python-authored colormaps now display.** When a node's
  metadata declares `colormap='custom'`, the scene loader opens its
  `colormap_lut` zarr array, validates byte length (768 RGB or 1024
  RGBA), and passes the bytes through `getColormapTexture('custom', lut)`
  for Points / Lines / GSplats. Invalid LUTs log a warning and fall
  back to viridis. Custom-LUT cache is disposed on `SceneManager.dispose()`.
- **GPU buffer pool byte-budget eviction.** New `gpuPoolMaxBytes` config
  (default 512 MB). Pooled buffers are evicted (largest-first) once
  total pooled bytes exceed the budget — independent of the count cap.
  `getStats()` now reports activeBytes / pooledBytes / totalBytes /
  largestPooledBytes, surfaced via `__luxarDebug.getState().gpuPool`.
- **EXR export modes.** `captureHDRPixels(mode)` and `captureHDRAsEXR({mode})`
  accept `'visible-ldr' | 'hdr-effects-pre-tone' | 'raw-scene-hdr'`.
  `'raw-scene-hdr'` disables every post-processing effect (incl. bloom/
  DOF/AO). `'hdr-effects-pre-tone'` keeps HDR-space effects but disables
  LDR display effects.
  `'visible-ldr'` keeps everything.
- **Points scalar colormaps end-to-end.** Loader (`loadPoints`) opens
  the `scalars` zarr array when `has_scalars=true`, populates
  `LoadedPointsData.scalars` through every load path; projection
  carries scalars through compaction; accumulator gains a scalar
  buffer (`getScalarBuffer()`); GPU pool detects scalar dtype
  (`'none' | 'Float32Array' | 'Uint8Array'`) and lazily binds the
  `scalar` attribute. The fail-closed guard naturally passes for any
  properly authored scalar dataset.
- **Lines scalar colormaps end-to-end.** Same pattern as Points.
  Loader opens `scalars`; `buildInstanceBuffers` interpolates scalars
  at clipped endpoints exactly like colors/widths/sharpness; WASM
  fallback path mirrors the TS path; GPU pool lazily allocates
  `aStartScalar`/`aEndScalar` instanced attributes; `updateInstancedLinesMesh`
  threads them through the shared `attrSpecs` path.
  Worker projection falls back to main-thread when scalars are
  present (worker payload doesn't carry scalars yet — separate change).
- **Browser-real shader compile + visual smoke tests.** New E2E specs:
  `shader-material-compile.spec.ts` (every material variant compiles +
  produces non-black pixels in a real browser),
  `line-rendering-visual.spec.ts` (cap factor and near-plane safety via
  sampled-pixel assertions), and `gsplat-rendering-visual.spec.ts` (ray
  integral and displayDims order). New helpers `assertNoShaderErrors(page)` and
  `samplePixelAt(page, sel, fx, fy)` in `tests/e2e/helpers.ts`. Visual
  screenshot baselines are deferred (need committed PNGs per
  platform); sampled-pixel assertions are platform-independent.

#### Changed — Viewer shader/material remediation (2026-05-10)

User-visible behavior changes in the Luxar viewer:

- **Point/Line scalar colormaps fail-closed when scalar attributes aren't bound.** Previously, metadata-authored `colormap`+`has_scalars` would activate `USE_COLORMAP` even though the geometry had no `scalar`/`aStartScalar`/`aEndScalar` attribute. Now the viewer logs a warning and falls back to vertex-color rendering.
- **Positions-only Points render visibly.** The GPU buffer pool now fills white/0.5/2.0 defaults for absent color/radius/sharpness arrays instead of leaving zeros (which the fragment shader discards as zero-contribution).
- **Point material parity for max blending.** `PointMaterial.applyBlendingMode` is the new single source of truth; runtime UI mode changes now produce the same `OneFactor/OneFactor` blend factors as creation-time max. The fragment shader gains a `LUXAR_MAX_RGB_CONTRIBUTION` define so `max` mode premultiplies RGB by intensity*opacity.
- **Normal-mode depth writes.** Fully-opaque (opacity ≥ 0.99) `normal` layers now write depth so additive layers behind them are correctly occluded.
- **Line body intensity reaches 1.0 as documented.** Cap factor moved from vertex to fragment shader.
- **Line near-plane safety.** Lines whose endpoints cross or sit very close to the camera no longer produce screen-filling artifacts.
- **Line bounds include rendered footprint** so thick lines aren't culled prematurely.
- **Line max blending RGB contribution** (same `LUXAR_MAX_RGB_CONTRIBUTION` define as Points).
- **GSplat `displayDims` order is preserved** to match Points/Lines. Previously dims were silently sorted.
- **GSplat sum-projection ray integral uses precision** (`1/sqrt(rᵀΣ⁻¹r)`), not covariance variance (`sqrt(rᵀΣr)`). Fixes physically incorrect brightness for rotated anisotropic splats viewed off-eigenaxis.
- **Conditional Gaussian slicing for correlated nD splats remains marginal+attenuation** (documented inline). Conditional `μ_D|H` / `Σ_D|H` math is scheduled as a follow-up.
- **Disabling a colormap clears uniform state**; clones of disabled materials no longer resurrect the disabled texture.
- **`LineMaterial.clone` copies `uIsOrtho`**.
- **GPU buffer pool is disposed** in `SceneLoader.dispose()`. Dataset switches no longer leak `InstancedBufferGeometry` references.
- **No-op blending changes don't recompile shaders.**
- **AO Quality control is no longer a no-op when AO is already enabled.**
- **Effective AA reporting via `getEffectiveAA(smaa, fxaa)`.**
- **Lines counted in `__luxarDebug.getState()`** (`totalLines` + `lineMeshes`).

Documentation: `E2E_TESTING_GUIDE.md` now references `getBufferedMessages()` (was stale `getMessages()`); `HDR_GUIDE.md` reads tone-mapping/exposure from `renderingControls.settings` rather than the nonexistent `state.rendering`.

#### Changed — GSplats default device on macOS

- `GaussianSplatModel` (and the gsplat fitting API) now auto-selects MPS on macOS when no explicit device is provided and `use_metal=True` (the default). Pre-rewrite, MPS was never auto-selected. Pass `use_metal=False` to keep CPU as the default on Macs that prefer it.
- Centralized device selection in `luxar.gsplats.utils.device.resolve_torch_device(...)`; remaining hand-rolled `torch.cuda.is_available()` / `torch.backends.mps.is_available()` ternaries in `utils/demos.py`, `gsplats/multiscale/decompose.py`, `gsplats/seeds/gpu_ops.py`, `gsplats/preprocessing/denoise_pipeline.py`, and `gsplats/fitting/preprocessing.py` (FPS GPU gate) now route through it. The denoise and FPS paths consequently honor MPS where they previously ignored it.

#### Fixed — Encoding, viewer, and GSplats hardening

- Tightened Luxar encoding metadata validation in Python and TypeScript: present `encoding` objects must declare a known `name`, direct dtype names are no longer treated as quantization, `array_ref` targets must be explicit, and malformed LUT/quantization metadata now fails loudly.
- Preserved Python-written integer color arrays (`uint8`/`uint16`) as direct SDR storage in the viewer and added cross-language fixture coverage.
- Hardened viewer loading/cache paths with bounded cache fetch retries, observable async cleanup failures, stricter `extend_to_all` dimension validation, canonical `sharpnesses` loading, and improved WebGL context-restoration resource recreation.
- **Embedder-visible**: `LuxarApp.init({ updateBrowserUrl })` default flipped from `true` to `false` so embedded callers no longer have host-page URL silently rewritten on dataset selection. The standalone bootstrap (`bootstrap.ts`) explicitly opts in. Existing embedded callers that depended on URL reflection must now pass `updateBrowserUrl: true` explicitly. (000bf00a)
- Fixed GSplats batch planning/loading for folded channel-like axes in >5D Zarr arrays, persisting channel-axis metadata and using real selected timepoint/channel indices in batch jobs.
- Bounded the PyTorch GSplat support-grid cache by adaptive per-device byte budgets instead of entry count, with HPC-tunable environment variables and oversized-entry skip behavior to prevent unbounded CPU/GPU memory growth.
- Avoided runtime `torch.compile` CPU toolchain failures by using eager PyTorch on CPU/MPS loss kernels and compiling opportunistically only for CUDA with fallback.

#### Changed — Metal gsplat backend parity, reliability, and performance

- Refactored `GaussianSplatModelMetal` into an MPS-only `GaussianSplatModel` subclass that mirrors the CUDA/base model interface for parameter management (`current_params`, `append_`, `prune_`, `replace_with`, state dicts, constraints) while using custom Metal kernels for 3D MPS tensors and PyTorch rendering for other supported 2D-8D MPS shapes.
- Rewrote the custom 3D Metal renderer from a tile-binned voxel-centric pipeline to a CUDA-style splat-centric pipeline: one Metal threadgroup owns one splat in forward and one splat gradient row in backward.
- Removed the old Metal hot-path tile machinery (`preprocess_3d`, `bin_3d`, tile counts/offsets/content, PyTorch prefix sum, CPU `.item()` allocation sync) and removed the packed-conic `[Z,Y,X]`↔`[X,Y,Z]` reorder; kernels now use native `[Z,Y,X]` Cholesky factors and compute packed conics internally per splat.
- Replaced voxel-centric global parameter-gradient atomics in backward with threadgroup reductions, an inline native `d_conic -> d_L` VJP, and one write per splat gradient. The unconstrained 3D Metal training path now consumes raw model parameters directly, applies sigmoid/softplus transforms and their VJPs inside Metal, and accepts scalar-expanded loss gradients without materializing dense `grad_output`. On the `128³ @ 32k splats` M4 Max benchmark this reduced forward+backward from roughly 133 ms to roughly 2.8-3.0 ms while keeping CPU-reference error around `max_abs_diff≈1.8e-5`.
- Fixed native Metal loading by passing the absolute `default.metallib` path into the extension, rebuilding stale Metal artifacts automatically when sources change, and touching generated artifacts after no-op distutils rebuilds so imports do not repeatedly auto-compile.
- Added/updated Metal tests covering 2D/4D PyTorch rendering, CPU/device-transfer rejection, explicit FP16/dtype rejection, dynamic splat operations, clean nested state dict compatibility, native `[Z,Y,X]` conic ordering, and splat-centric gradient behavior.

#### Added — HuRI PPI flow-field demo

- New `demo_ppi_flow_field.py` turns the HuRI protein-protein interaction graph into a signed-flow UMAP landscape: PageRank orients interactions low→high, a signed sparse adjacency/flow-profile matrix drives the 3D embedding, and protein nodes are forward-advected through a smoothed vector field.
- Vector field construction uses KD-tree edge-sample candidate lookup, exact point-to-segment distances for candidate edges, regularized inverse-cubic weighting, Gaussian component smoothing, and vectorized RK4 integration. The default `full` preset builds the requested 256³ grid; `--preset preview` builds a faster 128³ grid.
- Added `networkx>=3.0` to demo development dependencies for HuRI/CAIDA/PPI graph analysis demos.

### April 2026

#### Added — Cosmicflows-4 Laniakea demo

- New `demo_cosmicflows_laniakea.py` recreates the Cosmicflows-4 / Laniakea flow visualization with 55,486 local-universe galaxies and a full preset that reproduces 29,555 RK4 streamlines from the public `manlius/laniakea` data pipeline.
- Downloads and caches the EDD galaxy table, CF4 velocity field, and basin-of-attraction grid; writes galaxies as Points and each basin's flow as toggleable indexed Lines with HDR additive rendering.

#### Added — Native bundles for distributable scenes

**`luxar export --native macos|linux-amd64|linux-arm64`**
- New flag on `luxar export` that wraps the viewer + zarr around a Go-compiled launcher binary instead of emitting the Python `serve.py` folder. End users double-click and get a real native window — no Python or browser-tab involvement on their machine.
- macOS produces a standard `.app` bundle (`Contents/Info.plist`, `MacOS/launcher`, `Resources/{viewer, data, AppIcon.icns}`) with Cmd+Q / Dock close-button → graceful HTTP server shutdown via `webview.Terminate`. No `LSUIElement`, so the app is fully Dock- and Cmd+Tab-visible.
- Linux produces a portable folder (`<App>-linux-<arch>/{luxar-launcher, viewer/, data/, <App>.png, README.txt}`); the `<App>.png` follows the FreeDesktop icon convention.
- `--name NAME` overrides the default bundle name (which defaults to the zarr stem).

**Embedded launcher**
- New `packages/luxar-launcher/` Go module (~150 lines, `github.com/webview/webview_go` + Go stdlib). Single source compiled to per-OS native binaries.
- Native window via system WebView (WKWebView on macOS, WebKitGTK on Linux). Static HTTP server on a free localhost port, CORS-enabled to mirror `luxar serve`.
- Runtime fallback: `LUXAR_LAUNCHER_NO_WEBVIEW=1` opens the user's default browser instead — useful for headless smoke tests and minimal Linux installs without `libwebkit2gtk`.
- 🌌 emoji icon (matching the viewer's favicon) rendered to PNG via Apple Color Emoji + Pillow, packed to `.icns` via `iconutil`. Source assets committed under `packages/luxar/src/luxar/cli/_launcher_assets/` so wheel installs ship a working icon.

**Build pipeline**
- New Make targets: `make install-go` (Homebrew on macOS, official tarball into `~/.local/go` on Linux — no sudo), `make build-launchers` (CGO=1 host-only build; emits `darwin-universal` via `lipo` on macOS, `linux-<arch>` on Linux), `make clean-launchers`.
- Wired into `check-deps`, `clean-all`, `clean-setup`, `help`, and the `setup-dev` "optional accelerators" footer — same treatment as `make install-rust` and `make setup-cuda`.
- Wheel packaging: launcher binaries (`cli/_launchers/`) and icon assets (`cli/_launcher_assets/`) live inside the Python package and ride along into wheel builds automatically when present at build time.

**Docs**
- New `docs/tutorials/distributing_scenes.rst` covering folder export, native bundles, multi-platform sharing, Gatekeeper / `xattr -cr` recovery, and lifecycle; the tutorial is linked from the main Sphinx docs index.
- New `Native Launcher Setup Details` section in `docs/guides/developer/BUILD_SYSTEM_SPEC.md`.
- New module entry `luxar.cli.native_app` in `docs/api/cli.rst`.
- New per-package READMEs at `packages/luxar-launcher/`, `cli/_launchers/`, `cli/_launcher_assets/`.

**Tests** — 14 new tests in `packages/luxar/src/luxar/cli/tests/test_native_app.py` covering bundle layout, plist correctness (no `LSUIElement`, `CFBundleIconFile = AppIcon`), icon distribution via the package, missing-launcher CLI error handling, name defaulting from `source.stem`, and the `--overwrite`-must-not-wipe-on-failure invariant.

#### Changed — Layer Semantics

**Rendering attribute composition (was: override)**
- Rendering attributes (`opacity`, `gamma`, `intensity`, `offset`, `blending_mode`) now compose along the scene graph root-to-leaf per the Luxar spec, rather than the previous override-only behavior
- `opacity`/`gamma`/`intensity` multiply through the chain; `offset` adds; `blending_mode` uses the nearest ancestor that sets it
- Wired via new `packages/luxar-viewer/src/data/attrs-composer.ts` and `SceneLoader.applyEffectiveAttrs()`; also applied in the progressive GSplats LOD pass-through
- The Layers panel recomposes per-leaf on every slider change so edits to group/ancestor layers flow into every descendant

**Group layers**
- A `group` node with `layer=True` is now exposed in the Layers panel as a composite layer whose controls fan out to every data descendant (points/lines/gsplats)
- Viewer's `LayerStateManager` was previously silently ignoring groups — fixed

#### Changed — GSplats API

**API default loss flipped from MSE to L1.** `fit_gaussian_splats()`,
`GaussianSplatFitter.fit()`, and `LossConfig` now default to
`loss_type="l1"` (was `"mse"`). The change is motivated by the
loss-comparison study (Supp. Doc. 5; `analysis/loss_comparison/`),
which shows L1 reaches equal-or-higher held-out PSNR than MSE on every
microscopy dataset tested, by up to +1.03 dB on the cleanest data and
+0.78 dB on the noisiest. Callers that explicitly pass `loss_type="mse"`
or `"poisson"` are unaffected. Test `LossConfig.test_default_values`
updated to assert `"l1"`.

#### Added — Layer API

- `Node.layer` is now a settable property (previously creation-only): `points.layer = True`
- New `Node.visible` authoring property (default `True`); `add_points(..., layer=True, visible=False)` starts the layer hidden in the panel
- `validate_colormap` now fails fast on unknown names (catches typos like `colormap='viridus'` at authoring time instead of downstream)
- Layers panel gained an **opacity** slider alongside gamma/range/blend/colormap
- Pressing **L** while focus is inside the Layers panel no longer closes it (previously a slider-drag could accidentally dismiss the panel)
- Docs: `LUXAR_ZARR_FORMAT.md` now has a Layers section; `data/README.md` documents composition; `layers/README.md` reflects groups + opacity

#### Major Features

**Screen-Space Overlay System**
- New `scene.add_text()`, `scene.add_image()`, `scene.add_html()` Python API for screen-space annotations
- Overlays rendered as HTML elements over the 3D canvas (below controls)
- Normalized screen coordinates `[0, 1]` with top-left origin, 9-point anchoring
- Viewport-relative sizing (fractions of viewport height/width)
- **Dimension-aware visibility**: `visible_range` parameter shows/hides overlays based on slider positions
- Image overlays: accept file paths, bytes, numpy arrays, PIL Images; stored as raw PNG/JPEG/WebP in zarr
- HTML overlays: restricted safe tag subset with inline styles, XSS sanitization
- Per-overlay configurable fade transitions
- Optional pointer interactivity (`interactive=True`)
- Blend modes for images: normal, multiply, screen, overlay, additive
- Text features: font presets (sans/serif/mono), background boxes, stroke outlines
- 93 new Python tests covering all overlay types, validation, and edge cases

### March 2026

#### Breaking Changes

**Removed sharpness from GSplats**
- Removed `sharpness` attribute from Gaussian Splats (GSplats) geometry type
- Points and Lines retain their sharpness attribute
- GSplats now use the standard Gaussian falloff (equivalent to sharpness=2.0) without per-splat configurability
- Affected formats: `.gsplats.zarr` standalone format and embedded Luxar scene format
- Migration: existing `.gsplats.zarr` files with sharpness arrays will ignore the sharpness data on load

#### Major Features

**Per-Node GOG Color Model & Global EOG Controls**
- **Per-node Gain-Offset-Gamma (GOG)**: Added `intensity` (linear gain), `offset` (black level subtraction), and `gamma` (tonal curve) to all node types (Points, Lines, GSplats)
- **Use case**: Microscopy background subtraction — negative offset suppresses fluorescence floor per channel
- **Shader model**: `adjusted = color * intensity + offset; clip; pow(adjusted, 1/gamma)` with early discard for zero-contribution fragments
- **Global Exposure-Offset-Gamma (EOG)**: Replaced per-shader `hdrMultiplier` with `exposure` (log2 stops), `global_offset`, `global_gamma` applied in post-processing
- **Vendored tone mapping**: `LuxarToneMappingEffect` extends pmndrs ToneMappingEffect with EOG in a single shader pass (zero extra bandwidth cost)
- **Full config propagation**: Python `ViewerConfig` -> zarr -> TypeScript -> UI sliders -> post-processing uniforms
- **UI**: HDR folder now has Exposure (-5 to +5 stops), Offset (-1 to +1), Gamma (0.1 to 10.0), and Tone Mapping selector

**nD Transforms on Non-Displayed Dimensions**
- Per-dimension affine (`scale`, `offset`) and categorical (`permutation`) transforms for non-displayed dimensions
- Python validation in `nd_transform` property with full round-trip zarr serialization
- Viewer uses inverse-query approach: transforms the query (slicePosition + tolerance) from world to local space O(1), rather than transforming all point coordinates O(N)
- No loader internals changed; works transparently with existing spatial indexing

**Recording Panel with Screenshot and Video Export**
- New recording panel UI (`src/ui/recording-panel.ts`) for capturing viewer output
- Screenshot export: single-frame capture (PNG, WebP, JPEG) with configurable resolution
- Video export: record viewport as video for supplementary materials and demos
- Accessible from viewer UI controls

**Scale Bar Overlay with Physical Units**
- Scale bar component (`src/ui/components/scale-bar.ts`) rendered as an overlay on the viewport
- Supports all Luxar physical units (nm, um, mm, cm, m, km, inch, foot, px, au)
- Automatically adapts to current zoom level and camera projection
- Essential for microscopy figure generation

**Tiled Fitting for Large Volumes**
- Cosine-apodized (Hann window) overlapping tiles for fitting arbitrarily large volumes
- Removes GPU memory ceiling: each tile is fit independently, then splats are concatenated
- Enables parallel fitting across tiles (and potentially across GPUs)
- Implementation in `gsplats/fit_tiled_gsplats.py` with tiling utilities in `gsplats/tiling.py`

**GSplat CLI: filter, split, slice, compare Commands**
- `luxar gsplat filter`: Filter splats by amplitude, eccentricity, bounding box, volume, and more
- `luxar gsplat split`: Split gsplat datasets into parts by count or explicit indices
- `luxar gsplat slice`: Slice by coordinate ranges using numpy-style syntax (e.g., `"0:50, :, 10:90"`)
- `luxar gsplat compare`: PSNR/SSIM quality metrics comparing gsplat reconstruction to original volume

**Camera Utilities**
- New `camera-utils.ts` module with `PerspectiveCamera | OrthographicCamera` union type
- Shared utilities for camera setup across perspective and orthographic projections

#### Maintenance

- Added `luxar[gsplats]` optional dependency group and lazy imports for gsplats tooling
- Standardized Python console output on `arbol` and viewer output on `utils/log`
- Filled missing package `README.md` and `SPECIFICATIONS.md` files across Python and viewer packages

### December 2025

#### Major Features

**Modular Theming System** 🎨
- **What**: Complete theming system with runtime theme switching, CSS-based architecture, and modern aesthetics
- **Themes**: 3 production-ready themes (Dark, Light, Frosted Glass)
- **Components**: All 8 UI components fully themed (error dialogs, help overlay, dimension sliders, debug console, dataset browser, data loading monitor, rendering controls)
- **Architecture**: Three-layer system (Theme definitions -> CSS files -> Component logic)
- **Features**:
  - Instant theme switching via UI dropdown (R key -> 🎨 Theme)
  - URL parameter support (`?theme=light`)
  - Automatic persistence via localStorage
  - 130+ CSS utility classes with BEM naming
  - 80+ CSS custom properties (`--luxar-*`)
  - Zero hardcoded colors in components
  - Consistent 0.15s fade-in animation across all UI panels
- **Benefits**:
  - 400+ inline styles removed (-90% inline styling)
  - CSS bundle: 48KB (6.77KB gzipped) - cacheable separately
  - JS bundle: -13KB reduction
  - Better accessibility (WCAG AAA high-contrast theme)
  - Easier maintenance (single source of truth for colors)
- **Testing**: 1074 unit tests + 21 E2E visual regression tests
- **Polish**: Frosted Glass theme with Apple-inspired frosted glass design, HDR default 1.0
- **Implementation**: 22 commits, 4,200+ lines added, 100% complete with final polish
- **Location**: `src/themes/`, `src/styles/`, UI components
- **Documentation**: Complete implementation plan in `docs/archive/developer-archive/THEMING_IMPLEMENTATION_PLAN.md`

### January 2025

#### Critical Bug Fixes

**Transform Composition Bug**
- **Issue**: Matrix multiplication order was reversed in `compose()` function (left-multiply instead of right-multiply)
- **Impact**: `compose(T1, T2, T3)` was applying T3 first instead of T1 first
- **Fix**: Changed `result = transform @ result` to `result = result @ transform`
- **Location**: `core/transforms.py:271`
- **Lesson**: Order matters for non-commutative transforms (rotate+translate). Always test with order-sensitive operations.

**Points Metadata Loss Bug**
- **Issue**: `Points.__init__()` set `self._metadata` before calling `super().__init__()`, then `Node.__init__()` overwrote it with empty dict
- **Impact**: ALL Points objects lost their metadata (has_colors, has_radii, max_radius, etc.)
- **Fix**: Call `super().__init__()` BEFORE setting `self._metadata` in Points class
- **Location**: `core/points.py:50-58`
- **Lesson**: When subclass and parent both initialize the same attribute, parent must initialize first

**Fullscreen Resize Bug**
- **Issue**: Point sizes changed incorrectly on first fullscreen toggle or window resize
- **Root Cause**: Scene initialization didn't call `updateSize()`, causing different behavior on first resize
- **Solution**: Make initialization call `this.updateSize()` in `scene-manager.ts` init() method
- **Lesson**: Ensure initialization and resize paths are identical to avoid first-time-only bugs

#### Code Cleanup

**Legacy Code Removed**
- Eliminated unused "legacy mode" from Node class (~40 lines dead code)
- Removed `group` parameter and all `if self._group is not None:` branches
- Node now only supports progressive writing mode (simpler, clearer)

**Deprecated Parameters Removed**
- Removed `units` parameter from `LuxarZarrCompiler` (use Dimensions instead)
- Removed `DimensionMetadata` class (use full-featured `Dimension` instead)
- Removed unused version constants (LEGACY, PREVIOUS, FUTURE)
- Total: ~160 lines of dead/deprecated code removed

#### Quality Improvements

**Compiler Refactoring**
- Reduced `write_points()` from 432 to 117 lines (73% reduction)
- Extracted 8 focused helper methods with single responsibilities

**Validation Consolidation**
- Created `validation/types.py` centralizing all validation
- Eliminated ~350 lines of duplication between `protocols.py` and `validation/base.py`
- Clear organization: types.py (basic), base.py (detailed for writing), nd.py (dimensional)

**Transform Handling Centralized**
- Added `read_transform_from_zarr()` companion function
- Single source of truth for NumPy to THREE.js transform conversion
- Eliminated ~50 lines of duplicate transpose logic

**Magic Numbers Extracted**
- Created 18 named constants for spatial index tuning
- All grid sizing heuristics now configurable via constants

**Performance**
- Vectorized HSV to RGB conversion in demos (30-100x faster)

**Documentation**
- Added 80+ inline comments explaining complex algorithms

#### New Features

**Debug Console & Console Logging**
- In-App Debug Console: Press Ctrl+L to toggle debug console
- Ring Buffer Implementation: Console interceptor uses 10,000 message ring buffer
- Early Message Capture: Console messages captured from app initialization
- Standardized Logging: All console logs use format: `[emoji] [Luxar] message`
- Debug Interface: Debug tools available at `window.__luxarDebug` when `?debug` URL param is present

**HDR Color Pipeline**
- Float32 Colors: Changed from Uint8Array to Float32Array for HDR support
- nD Slicing Fix: Updated slicing algorithms to use `sliceColorsFloat32()` for proper HDR colors
- HDR Detection: Added comprehensive HDR capability detection in `utils/hdr-detection.ts`
- Note: WebGL canvas doesn't support true HDR output (limited to 8-bit)

**World-Space Point Sizing**
- Physical Accuracy: Points now use world-space sizing instead of screen-space
- Key Property: Two points with radius r at distance 2r will just touch
- FOV Independence: Points maintain physical size regardless of field of view changes
- Formula: `angularSize = 2 * atan(radius/distance)`, then converted to pixels

**nD Visualization**
- Slicing Tolerance: Use point radius for visibility, not fixed tolerance
- Scene Dimensions: Always define at scene level for consistency
- Keyboard Navigation: Simple 2-step: select dimension (1-9), navigate ([/])
- TypeScript Integration: Scene dimensions loaded from zarr attrs, used for step sizes

**Data Loading Architecture Refactor**
- Removed Lazy Loading: Eliminated LazyDataManager in favor of spatial index-based loading
- Spatial Index Required: All datasets now require spatial indices for efficient loading
- Range-Based Caching: New RangeCache system for intelligent memory management
- Improved Monitoring: Enhanced DataLoadingMonitor with better error handling and disposal
- Cleaner Architecture: Removed intermediate abstractions for simpler, more maintainable code

---

## Earlier History

See git history for changes prior to January 2025.
