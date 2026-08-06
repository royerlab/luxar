/**
 * Centralized URL parameter parsing.
 *
 * `window.location.search` is read in exactly one place — `main.ts` — and the
 * result flows through the application as a typed object. Components that need
 * a flag declare it on their options, rather than reaching back to
 * `window.location` themselves. URL writing is centralized here for the same
 * reason.
 *
 * This makes consumers testable (no need to mock `window.location`), the URL
 * contract auditable (every recognized parameter is listed in `UrlParams`),
 * and prepares the viewer for any context where `window.location` is not the
 * right source — embedded iframes, programmatic instantiation, SSR.
 */

/**
 * All URL parameters recognized by the viewer.
 *
 * Every consumer that wants a URL-derived value should accept the relevant
 * field via constructor/init options rather than read `window.location`.
 */
import { parseLineJoinStyle, type LineJoinStyle } from '../types/line-join';

const MAX_SRC_LENGTH = 4096;
const URL_SCHEME_PATTERN = /^[a-zA-Z][a-zA-Z\d+.-]*:/;
function hasUnsafeSrcCharacter(src: string): boolean {
  for (const char of src) {
    const code = char.charCodeAt(0);
    if (code <= 0x1f || code === 0x7f || char === '<' || char === '>') {
      return true;
    }
  }
  return false;
}

/**
 * Validate and normalize a data-source URL from user-controlled input.
 *
 * Accepted forms:
 * - Absolute HTTP(S) URLs: `https://example.com/data.zarr`
 * - Root-relative paths: `/datasets/data.zarr`
 * - Relative paths: `datasets/data.zarr`
 *
 * Rejected forms include unsupported schemes (`file:`, `javascript:`,
 * `data:`, `vbscript:`, `blob:`, etc.), protocol-relative URLs
 * (`//host/path`), control characters, obvious HTML delimiters, empty
 * strings, and excessively long values.
 *
 * Trailing slashes are stripped: the zarr loader appends path components
 * (metadata, chunks) to this string, and stripping here is what lets the
 * viewer accept both spellings while storing the canonical no-trailing-slash
 * form (see CLAUDE.md "Data Source URLs Normalize Trailing Slashes").
 */
export function normalizeDataSourceUrl(rawSrc: string | null): string | null {
  if (rawSrc === null) return null;

  const src = rawSrc.trim();
  if (src.length === 0 || src.length > MAX_SRC_LENGTH) return null;
  if (hasUnsafeSrcCharacter(src)) return null;
  if (src.startsWith('//')) return null;

  const hasScheme = URL_SCHEME_PATTERN.test(src);
  if (hasScheme) {
    try {
      const url = new URL(src);
      if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
      // Return canonical url.href (trailing slash stripped) so
      // mixed-case schemes like `HTTPS://...` flow through downstream
      // helpers as `https://...`. The scene-loader's url-normalization
      // helper only matches lowercase prefixes; returning `src` verbatim
      // would let it treat the URL as a relative path.
      const canonical = url.href.replace(/\/+$/, '');
      return canonical.length === 0 ? null : canonical;
    } catch {
      return null;
    }
  }

  // Strip trailing slashes so the zarr loader builds clean child paths.
  // A bare "/" is dropped to "" and rejected as empty.
  const trimmed = src.replace(/\/+$/, '');
  if (trimmed.length === 0) return null;
  return trimmed;
}

/**
 * All URL parameters recognized by the viewer, as a typed snapshot.
 *
 * Produced once by {@link readUrlParams} and threaded through the app; every
 * consumer that wants a URL-derived value accepts the relevant field via
 * options rather than reading `window.location` itself. Adding a new
 * recognized parameter means adding a field here and a line to
 * {@link readUrlParams}.
 */
export interface UrlParams {
  /** Dataset source URL (`?src=...`). Null when not provided. */
  src: string | null;
  /** Theme override (`?theme=light` etc). Null when not provided. */
  theme: string | null;
  /** Enable the `window.__luxarDebug` interface (`?debug`). */
  debug: boolean;
  /** Disable all cache layers (`?no-cache`). */
  noCache: boolean;
  /** Disable only the SliceCache / S-cache (`?no-slice-cache`). */
  noSliceCache: boolean;
  /** Verbose cache logging (`?cache-debug`). */
  cacheDebug: boolean;
  /** Clear caches on init (`?clear-cache`). */
  clearCache: boolean;
  /**
   * Whether the substitutive-LOD cross-fade is enabled: blend adjacent LOD
   * levels' opacity as the camera zooms across their boundary instead of a hard
   * visibility swap, for blendable (additive/luminous/volumetric) layers
   * (anti-popping). **On by
   * default**; pass `?no-lod-fade` to disable it (e.g. to compare against the
   * hard swap or isolate a rendering issue).
   */
  lodFade: boolean;
  /**
   * Whether streaming brightness compensation is enabled: as a blendable
   * (additive/luminous/volumetric)
   * LOD leaf's additive ladder streams in, scale its opacity by `1/e(k)` so the
   * partial prefix renders at full-level brightness instead of brightening up as
   * chunks arrive (anti-popping on the time axis, orthogonal to `lodFade`'s
   * distance axis). **On by default**; pass `?no-lod-energy` to disable it (e.g.
   * to compare against the uncompensated brightening ramp).
   */
  lodEnergyComp: boolean;
  /**
   * Force the finest LOD level regardless of projected screen coverage
   * (`?lod-finest`). For high-quality still/video capture — the gallery
   * harness appends it — where a coarse level would look blurry even though
   * the subject is small in frame. **Off by default** (opt-in, unlike the
   * three on-by-default flags above).
   */
  lodFinest: boolean;
  /**
   * Whether gsplat depth sorting is enabled (depth-sorting Phases 2-3): the
   * async worker sort that keeps `normal`-mode splats composited back-to-front,
   * plus the per-frame camera-motion re-sort scheduler. **On by default**; pass
   * `?depthSort=0` (also `false`/`off`) to disable it — `normal`-mode gsplats
   * then keep the identity (storage) order, which pins deterministic output for
   * E2E/visual runs and reproduces pre-Phase-2 behavior for comparison.
   */
  depthSort: boolean;
  /** Disable adjacent-chunk prefetching (`?no-prefetch`). */
  noPrefetch: boolean;
  /** Verbose prefetch logging (`?prefetch-debug`). */
  prefetchDebug: boolean;
  /**
   * Auto-open the data-loading monitor in expanded mode on the Cache tab
   * (`?cache-stats`). Useful for measuring L0/L1/L2 hit rates without
   * having to find the monitor's keyboard shortcut first.
   */
  cacheStats: boolean;
  /**
   * Force a specific rendering backend regardless of the default
   * resolution. Useful for per-load A/B comparisons and for diagnosing
   * TSL-vs-GLSL divergences without restarting the dev server.
   *
   * - `?renderer=webgl` — `THREE.WebGLRenderer` + GLSL `ShaderMaterial`
   *   (the production default).
   * - `?renderer=webgpu` — opt into `WebGPURenderer` + TSL
   *   `NodeMaterial`. The renderer internally dispatches to a real
   *   WebGPU adapter when available or falls back to its WebGL2
   *   backend otherwise.
   * - Unset (`null`) — fall back to the build-time
   *   `VITE_LUXAR_USE_WEBGPU` env var (opt-in to WebGPU); if that is
   *   also unset, the default is `webgl`.
   *
   * Any other value is normalized to `null` (defer to env / default).
   */
  renderer: 'webgl' | 'webgpu' | null;
  /**
   * Diagnostic flag (`?webgpu-force-webgl`) that keeps the
   * `WebGPURenderer` / TSL `NodeMaterial` pipeline selected but asks
   * Three.js to back it with its internal WebGL2 backend instead of a
   * native WebGPU adapter. Ignored when `renderer` resolves to `webgl`.
   */
  webgpuForceWebGL: boolean;
  /**
   * Opt-in to GPU timestamp queries (`?perf-timestamp`). Only honored
   * under WebGPURenderer with a backend that exposes the
   * `timestamp-query` feature. When set, the renderer is constructed
   * with `{ trackTimestamp: true }` and the perf bench reads
   * per-frame GPU time via `renderer.resolveTimestampsAsync('render')`.
   * Has a small runtime cost so the perf bench is the only intended
   * caller; never set on the production viewer URL.
   */
  perfTimestamp: boolean;
  /**
   * Override the adaptive GPU-geometry byte budget, in megabytes
   * (`?gpuBudgetMB=1536`). Pins the single VRAM budget shared by the
   * buffer pool and LOD-group retention, bypassing the auto-size
   * heuristic. Useful for large scenes on high-VRAM machines (raise it)
   * or for testing eviction on constrained ones (lower it). `0` disables
   * the byte budget (unbounded resident geometry). Null/invalid (missing
   * or negative) ⇒ auto-size from `navigator.deviceMemory`.
   */
  gpuBudgetMB: number | null;
  /**
   * Override the total in-memory cache pool (L0 + L1 + S-cache), in megabytes
   * (`?cacheBudgetMB=1536`). Used where `performance.memory` is unavailable —
   * WKWebView (the native app) and Safari — so heap-aware sizing has a real
   * budget to split instead of the tiny fixed fallback. The native launcher
   * injects it automatically. Null/invalid ⇒ fall back to the measured heap,
   * then to the fixed config sizes. See `cache/heap-budget.ts`.
   */
  cacheBudgetMB: number | null;
  /**
   * Pin a fixed device pixel ratio and disable adaptive DPR for the
   * session (`?dpr=1`). The value is clamped to [0.25, native DPR] at
   * apply time and the adaptive-resolution toggle is locked off so
   * persisted settings can't silently re-enable it. Primarily for
   * deterministic E2E/visual-regression runs, `agent:debug` sessions,
   * and bug repros. Null/invalid (missing, non-numeric, <= 0) ⇒ normal
   * adaptive behavior.
   */
  dpr: number | null;

  /**
   * Force a line join style for the session (`?lineJoin=none|overlap|miter`).
   *
   * Overrides whatever each node authored, which is exactly its purpose: it is
   * a debugging and workaround lever, so it must win over the scene file.
   * Precedence is `?lineJoin=` > authored node attribute > the built-in
   * default. `null` (missing or unrecognised) means "no override" — distinct
   * from `'none'`, which is an explicit request for no join geometry. See
   * `rendering/materials/line/join-style.ts`.
   */
  lineJoin: LineJoinStyle | null;
}

/**
 * Parse the supplied query string (or `window.location.search` by default)
 * into a typed `UrlParams` snapshot.
 *
 * Pass an explicit `search` string in tests; in production main.ts calls this
 * once with no argument and threads the result through the rest of the app.
 */
export function readUrlParams(search?: string): UrlParams {
  const raw = search ?? (typeof window !== 'undefined' ? window.location?.search : '') ?? '';
  const params = new URLSearchParams(raw);

  return {
    src: normalizeDataSourceUrl(params.get('src')),
    theme: params.get('theme'),
    debug: params.has('debug'),
    noCache: params.has('no-cache'),
    noSliceCache: params.has('no-slice-cache'),
    cacheDebug: params.has('cache-debug'),
    clearCache: params.has('clear-cache'),
    lodFade: !params.has('no-lod-fade'),
    lodEnergyComp: !params.has('no-lod-energy'),
    lodFinest: params.has('lod-finest'),
    depthSort: parseEnabledFlag(params.get('depthSort')),
    noPrefetch: params.has('no-prefetch'),
    prefetchDebug: params.has('prefetch-debug'),
    cacheStats: params.has('cache-stats'),
    renderer: normalizeRendererParam(params.get('renderer')),
    webgpuForceWebGL: params.has('webgpu-force-webgl'),
    perfTimestamp: params.has('perf-timestamp'),
    gpuBudgetMB: parseNonNegativeInt(params.get('gpuBudgetMB')),
    cacheBudgetMB: parseNonNegativeInt(params.get('cacheBudgetMB')),
    dpr: parsePositiveFloat(params.get('dpr')),
    lineJoin: parseLineJoinStyle(params.get('lineJoin')),
  };
}

/**
 * Parse an on-by-default enable flag: only an explicit `0`/`false`/`off`
 * value (case-insensitive) disables; missing or any other value keeps the
 * feature enabled. Used by `?depthSort=0`.
 */
function parseEnabledFlag(raw: string | null): boolean {
  if (raw === null) return true;
  const v = raw.trim().toLowerCase();
  return v !== '0' && v !== 'false' && v !== 'off';
}

/**
 * Parse a strictly-positive float query value; null on missing/invalid/<=0.
 * Used by `?dpr=` where zero or negative pixel ratios are meaningless.
 */
function parsePositiveFloat(raw: string | null): number | null {
  if (raw === null) return null;
  const n = Number.parseFloat(raw);
  return Number.isFinite(n) && n > 0 ? n : null;
}

/**
 * Parse a non-negative integer query value; null on missing/invalid/<0.
 * `0` is a valid value — for `?gpuBudgetMB=0` it flows through to
 * `configureGpuByteBudget` as the explicit "disable the byte budget" signal,
 * matching the config-level `0` semantics.
 */
function parseNonNegativeInt(raw: string | null): number | null {
  if (raw === null) return null;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

/**
 * Validate the `?renderer=` query value. Accept `webgl` and `webgpu`
 * case-insensitively; everything else (including the empty
 * `?renderer` flag-only form) is treated as "no override".
 */
function normalizeRendererParam(raw: string | null): 'webgl' | 'webgpu' | null {
  if (raw === null) return null;
  const v = raw.trim().toLowerCase();
  if (v === 'webgl' || v === 'webgl2') return 'webgl';
  if (v === 'webgpu') return 'webgpu';
  return null;
}

/**
 * Minimal read-only view of `window.location` this module needs to rewrite the
 * `src` query parameter. Narrowed to an interface so callers (and tests) can
 * supply a plain object instead of a real `Location`.
 */
export interface BrowserUrlLocation {
  pathname: string;
  search: string;
  hash?: string;
}

/**
 * Minimal `window.history` surface used to replace the current URL without a
 * navigation. Narrowed to just `replaceState` for testability.
 */
export interface BrowserUrlHistory {
  replaceState(data: unknown, unused: string, url?: string | URL | null): void;
}

/**
 * The browser environment {@link replaceBrowserDataSourceUrl} writes into:
 * a {@link BrowserUrlLocation} to read from and a {@link BrowserUrlHistory} to
 * write to. Defaults to the real `window`; injectable in tests.
 */
export interface BrowserUrlWriter {
  location: BrowserUrlLocation;
  history: BrowserUrlHistory;
}

/**
 * Strip trailing slashes from a `src` value before writing it into the
 * address bar. Parsing normalizes trailing slashes away anyway (see
 * `normalizeDataSourceUrl`), so this keeps the STORED URL in the canonical
 * no-trailing-slash spelling instead of round-tripping a non-canonical form.
 */
function normalizeSrcForUrl(src: string): string {
  return src.replace(/\/+$/, '');
}

/**
 * Build a URL path for the current viewer page with `src` updated.
 *
 * This helper preserves existing query parameters and hash fragments while
 * centralizing the viewer's URL-writing contract. It returns a path-relative
 * URL suitable for `history.replaceState()`. The `src` is normalized so it
 * never carries a trailing slash.
 */
export function buildDataSourceBrowserUrl(src: string, location: BrowserUrlLocation): string {
  const params = new URLSearchParams(location.search);
  params.set('src', normalizeSrcForUrl(src));
  const query = params.toString();
  const hash = location.hash ?? '';
  return `${location.pathname}${query ? `?${query}` : ''}${hash}`;
}

/**
 * Replace the current browser URL with the selected dataset source.
 *
 * Returns `false` if the environment has no browser history/location or if
 * `history.replaceState()` is blocked, e.g. by a sandboxed iframe. Callers
 * should treat failure as non-fatal and continue loading the selected dataset.
 * The `src` is always normalized to drop trailing slashes — callers do not
 * need to pre-normalize.
 */
export function replaceBrowserDataSourceUrl(src: string, target?: BrowserUrlWriter): boolean {
  const writer = target ?? (typeof window !== 'undefined' ? window : undefined);
  if (!writer?.location || !writer.history) return false;

  try {
    writer.history.replaceState({}, '', buildDataSourceBrowserUrl(src, writer.location));
    return true;
  } catch {
    return false;
  }
}
