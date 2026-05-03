# Changelog

All notable changes to Luxar are documented in this file.

## [Unreleased]

### May 2026

#### Changed — GSplats default device on macOS

- `GaussianSplatModel` (and the gsplat fitting API) now auto-selects MPS on macOS when no explicit device is provided and `use_metal=True` (the default). Pre-rewrite, MPS was never auto-selected. Pass `use_metal=False` to keep CPU as the default on Macs that prefer it.
- Centralized device selection in `luxar.gsplats.utils.device.resolve_torch_device(...)`; remaining hand-rolled `torch.cuda.is_available()` / `torch.backends.mps.is_available()` ternaries in `utils/demos.py`, `gsplats/multiscale/decompose.py`, `gsplats/seeds/gpu_ops.py`, `gsplats/preprocessing/denoise_pipeline.py`, and `gsplats/fitting/preprocessing.py` (FPS GPU gate) now route through it. The denoise and FPS paths consequently honor MPS where they previously ignored it.

#### Fixed — Encoding, viewer, and GSplats hardening

- Tightened Luxar encoding metadata validation in Python and TypeScript: present `encoding` objects must declare a known `name`, direct dtype names are no longer treated as quantization, `array_ref` targets must be explicit, and malformed LUT/quantization metadata now fails loudly.
- Preserved Python-written integer color arrays (`uint8`/`uint16`) as direct SDR storage in the viewer and added cross-language fixture coverage.
- Hardened viewer loading/cache paths with bounded cache fetch retries, observable async cleanup failures, stricter `extend_to_all` dimension validation, canonical `sharpnesses` loading, and improved WebGL context-restoration resource recreation.
- Fixed GSplats batch planning/loading for folded channel-like axes in >5D Zarr arrays, persisting channel-axis metadata and using real selected timepoint/channel indices in batch jobs.
- Bounded the PyTorch GSplat support-grid cache by adaptive per-device byte budgets instead of entry count, with HPC-tunable environment variables and oversized-entry skip behavior to prevent unbounded CPU/GPU memory growth.
- Avoided runtime `torch.compile` CPU toolchain failures by using eager PyTorch on CPU/MPS loss kernels and compiling opportunistically only for CUDA with fallback.

#### Changed — Metal gsplat backend parity, reliability, and performance

- Refactored `GaussianSplatModelMetal` into an MPS-only `GaussianSplatModel` subclass that mirrors the CUDA/base model interface for parameter management (`current_params`, `append_`, `prune_`, `replace_with`, state dicts, constraints) while using custom Metal kernels for 3D MPS tensors and PyTorch rendering for other supported 2D-8D MPS shapes.
- Rewrote the custom 3D Metal renderer from a tile-binned voxel-centric pipeline to a CUDA-style splat-centric pipeline: one Metal threadgroup owns one splat in forward and one splat gradient row in backward.
- Removed the old Metal hot-path tile machinery (`preprocess_3d`, `bin_3d`, tile counts/offsets/content, PyTorch prefix sum, CPU `.item()` allocation sync) and removed the packed-conic `[Z,Y,X]`↔`[X,Y,Z]` reorder; kernels now consume native `[Z,Y,X]` packed conics.
- Replaced voxel-centric global parameter-gradient atomics in backward with threadgroup reductions and one write per splat gradient. On the `128³ @ 32k splats` M4 Max benchmark this reduced forward+backward from roughly 133 ms to roughly 4.4-4.8 ms while keeping CPU-reference error around `max_abs_diff≈1.8e-5`.
- Fixed native Metal loading by passing the absolute `default.metallib` path into the extension and rebuilding stale Metal artifacts automatically when sources change.
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
