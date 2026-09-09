import type { LoaderConfig } from '../../data/data-loader-types';
import type { AppFactories } from './factories';

/**
 * Init-time options for {@link LuxarApp.init}.
 *
 * Typically constructed by `main.ts` from `readUrlParams()`, but any caller
 * can provide values directly — useful for tests, embedding, and notebook
 * integrations where `window.location` is not the right source.
 */
export interface LuxarAppOptions {
  /**
   * Canvas element to render into. The standalone app's main.ts resolves
   * this via `document.getElementById('app')`; embedders pass any
   * HTMLCanvasElement they own.
   */
  canvas: HTMLCanvasElement;
  /**
   * Host element the viewer mounts all of its overlays, panels, toasts,
   * dialogs, and injected SVG filters into. Defaults to `document.body`
   * (the standalone-app behaviour).
   *
   * Embedders pass the element that wraps their `canvas` so the entire
   * viewer DOM subtree lives inside host-owned markup — `dispose()` then
   * removes it cleanly, and the viewer's `position: fixed` overlays are
   * scoped to the container box (the viewer promotes a non-`body` container
   * to a containing block via `contain: layout`, restored on dispose).
   *
   * Note: still one viewer per page — see the README "Embedding" section.
   */
  container?: HTMLElement;
  /** Dataset URL. Defaults to {@link config.defaultZarrPath}. */
  src?: string;
  /** Expose `window.__luxarDebug` and verbose hardware logging. */
  debug?: boolean;
  /** Cache and prefetch flags forwarded to the data loader. */
  loaderConfig?: LoaderConfig;
  /**
   * Session-wide GPU geometry budget in bytes. `null` auto-sizes from device
   * memory, measured heap, and device class; `0` disables byte-budget eviction,
   * and a positive value pins the budget. Defaults to
   * `config.dataLoading.performance.gpuPoolMaxBytes`.
   */
  gpuPoolMaxBytes?: number | null;
  /**
   * Reflect the loaded dataset URL in the browser address bar via
   * `history.replaceState` so the page can be reloaded or shared.
   *
   * Defaults to `false` for programmatic/embedded safety. The standalone
   * bootstrap sets this to `true` explicitly.
   */
  updateBrowserUrl?: boolean;

  /**
   * Absolute URL to the WASM JS shim (`luxar_wasm.js`).
   *
   * Defaults to `wasm/luxar_wasm.js` resolved relative to the bundled JS
   * (`import.meta.url`), which works for Vite, Rollup, webpack 5, and most
   * modern bundlers. Because the chunk carrying the loader sits at a
   * different depth in the app build (`assets/…`) than in the library
   * build's entry chunk (the output root), the loader tries
   * `../wasm/luxar_wasm.js` then `./wasm/luxar_wasm.js` until one imports —
   * one attempt, not two, when the chunk sits at the URL root and the two
   * resolve to the same href (on the Vite dev server neither applies: the
   * shim is fetched from `/wasm/luxar_wasm.js` off the origin, as a single
   * candidate).
   * Embedders whose bundlers don't support `import.meta.url` for asset URLs
   * (or who ship the WASM files from a non-default location) override this.
   */
  wasmPath?: string;

  /**
   * Absolute URL to the data-worker module bundle.
   *
   * Defaults to `new URL('./data-worker.ts', import.meta.url)`. Override
   * if your bundler can't resolve worker URLs that way.
   */
  workerPath?: string;

  /**
   * Open the data-loading monitor in expanded mode on the Cache tab as
   * soon as the scene is wired up. Set by the standalone bootstrap when
   * `?cache-stats` is in the URL; embedders can pass it explicitly when
   * profiling cache behaviour.
   */
  openCacheStats?: boolean;

  /**
   * Optional construction overrides for the heavy components
   * constructed by `init()`. When omitted (or per-key undefined),
   * `defaultFactories` is used and the production path simply calls
   * the matching `new X(...)`. Embedders + tests use this hook to
   * substitute alternate scene managers, recording panels, etc.
   * See `factories.ts`.
   */
  factories?: AppFactories;

  /**
   * Force a specific rendering backend, overriding the default
   * resolution. Mirrors `UrlParams.renderer` — the standalone
   * bootstrap reads `?renderer=webgl|webgpu` and threads it here
   * so per-load A/B testing doesn't need a dev-server restart.
   *
   * - `'webgl'`: `THREE.WebGLRenderer` + GLSL `ShaderMaterial` (the
   *   production default).
   * - `'webgpu'`: `WebGPURenderer` + TSL `NodeMaterial`. Internally
   *   falls back to WebGL2 when no WebGPU adapter.
   * - Undefined: fall back to `VITE_LUXAR_USE_WEBGPU` (opt-in to
   *   WebGPU) / `VITE_LUXAR_USE_LEGACY_WEBGL` (no-op, matches default)
   *   env vars, then the WebGL default.
   */
  renderer?: 'webgl' | 'webgpu';

  /**
   * Diagnostic mode for `renderer: 'webgpu'`: construct
   * `WebGPURenderer({ forceWebGL: true })` so Three.js still uses the
   * WebGPURenderer API surface and TSL `NodeMaterial` shaders, but routes
   * rendering through its internal WebGL2 backend. Mirrors the
   * `?webgpu-force-webgl` URL flag.
   */
  webgpuForceWebGL?: boolean;

  /**
   * Opt-in to WebGPU `timestamp-query` profiling. Construct
   * `WebGPURenderer({ trackTimestamp: true })` so the perf bench can
   * read per-frame GPU duration via
   * `renderer.resolveTimestampsAsync('render')`. Tiny runtime cost
   * (~1-2% per Three.js docs); intended only for the perf-bench spec
   * (`?perf-timestamp` URL flag). Ignored under `WebGLRenderer`.
   */
  perfTimestamp?: boolean;

  /**
   * Pin a fixed device pixel ratio for the whole session. Mirrors
   * `UrlParams.dpr` (`?dpr=1`) — the standalone bootstrap threads it
   * here. When set, the adaptive-DPR manager is disabled, the value is
   * clamped to [0.25, native DPR] and applied as a manual DPR, and the
   * adaptive-resolution toggle is locked so persisted per-scene
   * settings can't silently re-enable adaptation. Intended for
   * deterministic E2E/visual-regression runs and bug repros.
   */
  pinnedDPR?: number;

  /**
   * Substitutive-LOD cross-fade (blend adjacent LOD levels' opacity across
   * a zoom transition instead of a hard swap). Default: true. Mirrors
   * `UrlParams.lodFade` (`?no-lod-fade` disables) — the standalone
   * bootstrap threads it here; embedders set it directly.
   */
  lodFade?: boolean;

  /**
   * Streaming brightness compensation (scale a streaming blendable
   * additive/luminous/volumetric LOD leaf's opacity by 1/e(k) so partial
   * ladders render at full-level brightness). Default: true. Mirrors
   * `UrlParams.lodEnergyComp` (`?no-lod-energy` disables).
   */
  lodEnergyComp?: boolean;

  /**
   * Force the finest LOD level regardless of projected screen coverage
   * (never coarsen, even off-screen). For high-quality still/video capture
   * where a coarse level looks blurry despite the subject being small in
   * frame. Default: false. Mirrors `UrlParams.lodFinest` (`?lod-finest`).
   */
  lodFinest?: boolean;

  /**
   * Bake the scene-derived environment once the load settles and hand the
   * container to `luxar env bake` (`__luxarDebug.environment.lastBake` + a
   * download). Mirrors `UrlParams.bakeEnv` / `probe` / `envResolution`
   * (`?bake-env&probe=…&env-resolution=…`). Undefined = normal viewing.
   */
  bakeEnvironment?: { probe?: string; resolution?: number };

  /**
   * WebGL-only blend warm-up (pre-compile each DISTINCT reachable
   * blend-mode program variant, one compile per macrotask). Default:
   * true on laptops/desktops and false on phones/tablets. Mirrors
   * `UrlParams.blendWarmup` (`?no-blend-warmup` disables) — the
   * standalone bootstrap threads it here; embedders can disable it directly.
   */
  blendWarmup?: boolean;

  /**
   * GSplat depth sorting (worker back-to-front sorting of `normal`-mode
   * splats + the camera-motion re-sort scheduler). Default: true; false
   * pins the identity storage ordering for deterministic runs. Mirrors
   * `UrlParams.depthSort` (`?depthSort=0` disables).
   */
  depthSort?: boolean;

  /**
   * Projected-density guard: per-node keep-fraction thinning (blendable
   * modes, brightness-compensated) and a refinement rung cap on nodes whose
   * elements-per-pixel exceed `config.densityGuard.capElementsPerPixel`.
   * Default: true. Mirrors `UrlParams.densityGuard` (`?no-density-guard`).
   */
  densityGuard?: boolean;

  /**
   * Session-only override of the density guard's blendable cap (elements per
   * drawing-buffer pixel), applied to both the thinning ladder and the
   * refinement rung gate. Undefined ⇒ `config.densityGuard.capElementsPerPixel`.
   * Mirrors `UrlParams.densityCap` (`?density-cap=8`). Not persisted.
   */
  densityCap?: number;

  /**
   * Allow a picked element's authored `link` to be opened on left-click
   * (issue #1917). Defaults to true.
   *
   * Set false — or load with `?no-links` — when embedding scenes you did not
   * author: `.zattrs` is untrusted input, and this is the switch that
   * guarantees no navigation can originate in data. It suppresses the
   * navigation, the two link items in the right-click menu, and the pointer
   * cursor; `Copy` keeps working, because writing to the clipboard is not
   * navigation. The `element-click` / `element-contextmenu` embedder events
   * still fire, so a host can implement its own behaviour instead.
   */
  allowLinks?: boolean;
}
