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
import { parseLinePrimitive, type LinePrimitive } from '../types/line-primitive';
import { ENVIRONMENT_RESOLUTION_MAX, ENVIRONMENT_RESOLUTION_MIN } from '../types/environment';
import type { InputProfileOverride } from '../utils/input-capabilities';

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

const MAX_CONTROL_URL_LENGTH = 2048;

/** The path the hub is served at, relative to the page's own origin. */
export const CONTROL_SOCKET_PATH = '/control';

/** The subset of `window.location` the control-URL validator needs. */
export interface ControlSocketOrigin {
  protocol: string;
  host: string;
}

/**
 * Resolve the remote-control socket URL.
 *
 * `?control` is a **bare flag** in the ordinary case: the hub is served by the
 * same app that served this page (`luxar serve --control` puts it on the viewer
 * app), so the socket address is derivable and there is nothing to validate.
 * That also means it keeps working behind an origin-rooted reverse proxy, under
 * `luxar export` and inside the native launcher, none of which know their own
 * address at authoring time. A path-prefixed proxy uses an explicit same-origin
 * path such as `?control=/exhibit/control`.
 *
 * `?control=<url>` is the split-origin override, and it is **same-origin only**
 * unless `allowCrossOrigin` is also set. The threat is concrete: a crafted
 * `?src=<real>&control=ws://attacker/` link turns the display into something an
 * attacker drives, and hands them `getViewerState()` — the dataset URL, the
 * layer list and the camera. The host comparison blocks the direct cross-origin
 * case, and an exhibit that genuinely splits the origins says so explicitly.
 *
 * @param raw The parameter value. `''` (a bare `?control`) derives the
 *   same-origin address; `null` means the flag was absent.
 * @param origin The page's own protocol and host.
 * @param allowCrossOrigin Whether `?controlAllowCrossOrigin` was present.
 * @returns The socket URL, or `null` when control is off or the value is refused.
 */
export function normalizeControlSocketUrl(
  raw: string | null,
  origin: ControlSocketOrigin,
  allowCrossOrigin = false
): string | null {
  if (raw === null) return null;
  const secure = origin.protocol === 'https:';
  const derived = `${secure ? 'wss:' : 'ws:'}//${origin.host}${CONTROL_SOCKET_PATH}`;

  const value = raw.trim();
  if (value.length === 0) return derived;
  const url = parseControlSocketUrl(value, derived);
  if (url === null) return null;
  return acceptControlSocketUrl(url, origin, secure, allowCrossOrigin) ? url.href : null;
}

/**
 * String-level checks, then a parse. `null` for anything not worth inspecting.
 *
 * Shared by the two control-related validators — the socket URL and the
 * `?panel=` module — because the string-level hazards are identical (length,
 * control characters, a protocol-relative address that would silently inherit
 * our origin). Only the per-scheme rules afterwards differ.
 */
function parseControlSocketUrl(value: string, base: string): URL | null {
  if (value.length > MAX_CONTROL_URL_LENGTH) return null;
  if (hasUnsafeSrcCharacter(value)) return null;
  // Protocol-relative: `new URL` would happily inherit our scheme and host,
  // which is exactly the cross-origin smuggle this validator exists to stop.
  if (value.startsWith('//')) return null;
  try {
    // A bare path (`/control`) resolves against our own origin; anything with a
    // scheme is parsed as absolute and checked by the caller.
    return new URL(value, base);
  } catch {
    return null;
  }
}

/** Whether a parsed socket URL is one this page may dial. */
function acceptControlSocketUrl(
  url: URL,
  origin: ControlSocketOrigin,
  secure: boolean,
  allowCrossOrigin: boolean
): boolean {
  if (url.protocol !== 'ws:' && url.protocol !== 'wss:') return false;
  // An insecure socket from a secure page is blocked by the browser anyway;
  // failing here produces a message instead of a bare SecurityError.
  if (secure && url.protocol === 'ws:') return false;
  if (controlSocketUrlCarriesNoise(url)) return false;
  return url.host === origin.host || allowCrossOrigin;
}

/**
 * Credentials, a fragment, or any query key but `token`.
 *
 * Credentials in the URL would sit in the address bar of a tablet on a plinth,
 * and in its history; the token has its own parameter. A fragment and stray
 * query keys are simply not part of this contract, and accepting them would
 * mean accepting whatever a crafted link put there.
 */
function controlSocketUrlCarriesNoise(url: URL): boolean {
  if (url.username.length > 0 || url.password.length > 0) return true;
  if (url.hash.length > 0) return true;
  // `forEach` rather than `keys()`: this package's lib config has DOM but not
  // DOM.Iterable, so the iterator helpers are not in the type surface.
  let foreignQueryKey = false;
  new URLSearchParams(url.search).forEach((_value, key) => {
    if (key !== 'token') foreignQueryKey = true;
  });
  return foreignQueryKey;
}

/**
 * The page's own origin, for the control-socket validator.
 *
 * Falls back to a loopback placeholder when there is no `window` at all (a
 * node-environment unit test), so the validator's same-origin rule still has
 * something concrete to compare against instead of throwing.
 */
function resolvePageOrigin(origin?: ControlSocketOrigin): ControlSocketOrigin {
  if (origin !== undefined) return origin;
  const location = typeof window !== 'undefined' ? window.location : undefined;
  if (!location) return { protocol: 'http:', host: 'localhost' };
  return { protocol: location.protocol, host: location.host };
}

/**
 * Resolve `?panel=<url>` — an alternative control-panel module.
 *
 * The documented escape hatch, so the first exhibit that outgrows CSS has a
 * supported path instead of forking `control.html` out of the viewer package.
 * The module is `import()`ed and handed the already-connected socket, so it is
 * **executable code**: hence same-origin only, with no opt-out. A cross-origin
 * module would be arbitrary remote code running on a kiosk, which is a
 * different feature and not one anybody asked for. Within our own origin it is
 * no more trusted than the page that loads it.
 */
export function normalizePanelModuleUrl(
  raw: string | null,
  origin: ControlSocketOrigin
): string | null {
  if (raw === null) return null;
  const value = raw.trim();
  // Empty must be refused explicitly: `new URL('', base)` resolves to the base
  // itself, so a blank `?panel=` would otherwise import the origin root.
  if (value.length === 0) return null;
  const url = parseControlSocketUrl(value, `${origin.protocol}//${origin.host}/`);
  if (url === null) return null;
  if (url.protocol !== origin.protocol || url.host !== origin.host) return null;
  if (url.username.length > 0 || url.password.length > 0) return null;
  return url.href;
}

/** A trimmed query value, or null when absent or blank. */
function trimmedParam(params: URLSearchParams, key: string): string | null {
  const raw = params.get(key);
  if (raw === null) return null;
  const value = raw.trim();
  return value.length === 0 ? null : value;
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
  /**
   * Browser tab title (`?title=...`). `luxar serve --open` derives it from
   * the dataset file name so several open viewer tabs are tellable apart;
   * a scene's authored `viewer_config.title` overrides it at load.
   */
  title: string | null;
  /**
   * Remote-control socket URL, already resolved and validated
   * (`?control`, or `?control=ws://host/control` to split the origin).
   * Null when control is off — which is the default — or when the supplied
   * value was refused. See {@link normalizeControlSocketUrl}.
   */
  control: string | null;
  /**
   * Shared secret presented to the hub as `?token=` (`?controlToken=...`),
   * matching `luxar serve --control-token`. Null when the hub is open.
   *
   * A token in a query string lands in browser history and in the address bar
   * of whatever tablet is driving the display. That is acceptable for a LAN
   * kiosk and is not a substitute for not exposing the hub to a network you
   * do not trust.
   */
  controlToken: string | null;
  /**
   * Permit a cross-origin control socket (`?controlAllowCrossOrigin`). Off by
   * default so a crafted link cannot point the display at someone else's hub.
   */
  controlAllowCrossOrigin: boolean;
  /**
   * Alternative control-panel module (`?panel=/my-panel.js`), resolved and
   * validated same-origin. Null when absent or refused.
   *
   * Read only by `control.html`; the viewer itself ignores it. See
   * {@link normalizePanelModuleUrl}.
   */
  panel: string | null;
  /** Enable the `window.__luxarDebug` interface (`?debug`). */
  debug: boolean;
  /**
   * `?kiosk` — lock this display down for unattended public use.
   *
   * A hard override over the scene's authored `ui.kiosk` block, because this
   * is the OPERATOR's channel: a store that predates the block, or one
   * borrowed for an exhibit it was never authored for, still has to be
   * lockable from the launch command. See `config/kiosk.ts`.
   */
  kiosk: boolean;
  /** Disable all cache layers (`?no-cache`). */
  noCache: boolean;
  /** Disable only the SliceCache / S-cache (`?no-slice-cache`). */
  noSliceCache: boolean;
  /**
   * Disable only the L2 OPFS persistent tier (`?no-opfs`); L0/L1/S-cache
   * stay on. The deterministic sibling of the OPFS circuit breaker — use
   * it in environments whose OPFS is known to stall (automated Chromium).
   */
  noOpfs: boolean;
  /** Override the page-wide OPFS read cap (`?opfsReadConcurrency=N`). */
  opfsReadConcurrency: number | null;
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
   * Whether a picked element's authored `link` may be opened on left-click
   * (issue #1917). **On by default**; pass `?no-links` to disable it.
   *
   * The switch an embedder showing third-party scenes wants: `.zattrs` is
   * untrusted, so this guarantees no navigation can originate in the data. It
   * suppresses the navigation, the two link items in the right-click menu and
   * the pointer cursor; `Copy` still works, since the clipboard is not
   * navigation.
   */
  allowLinks: boolean;
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
   * Session-wide replacement-LOD bias (`?lod-bias=<positive number>`), in
   * screen-area units. `2` advances an occupancy-halved ladder by one level;
   * `4` by two. Null keeps the neutral `1` default.
   */
  lodBias: number | null;
  /**
   * WebGL-only blend warm-up. **On by default**; pass `?no-blend-warmup`
   * to disable the off-interaction-path pre-linking of reachable
   * blend-mode program variants.
   */
  blendWarmup: boolean;
  /**
   * Whether gsplat depth sorting is enabled (depth-sorting Phases 2-3): the
   * async worker sort that keeps `normal`-mode splats composited back-to-front,
   * plus the per-frame camera-motion re-sort scheduler. **On by default**; pass
   * `?depthSort=0` (also `false`/`off`) to disable it — `normal`-mode gsplats
   * then keep the identity (storage) order, which pins deterministic output for
   * E2E/visual runs and reproduces pre-Phase-2 behavior for comparison.
   */
  depthSort: boolean;

  /**
   * Projected-density guard (per-node keep-fraction thinning + refinement
   * rung cap on over-drawn nodes; `config.densityGuard`). On by default;
   * `?no-density-guard` disables it for the session — the A/B lever for
   * the audit bench and for reproducing an overdraw report.
   */
  densityGuard: boolean;
  /**
   * Session-only override of the density guard's blendable cap
   * (`?density-cap=8`, elements per drawing-buffer pixel;
   * `config.densityGuard.capElementsPerPixel` is 4). Both consumers follow
   * it — the shader keep-fraction ladder and the refinement rung gate — so
   * a threshold sweep is one URL edit per arm, no rebuild and nothing
   * persisted. The non-blendable cap (1) only moves when the override is
   * below it, so it stays the tighter of the two. Null/invalid ⇒ the
   * configured cap.
   */
  densityCap: number | null;
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
   * injects it automatically. Its implied non-cache remainder also replaces
   * the heap-derived GPU-geometry signal in either direction. Null/invalid ⇒
   * fall back to the measured heap, then to the fixed config sizes. See
   * `cache/heap-budget.ts`.
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
   * Force the session's JS input profile (`?input=touch|mouse`): pointer flags,
   * hover capability, touch points, and device tier. This changes device-class
   * fallback budgets (`touch` only — `mouse` keeps the detected tier),
   * primary-tip pen routing, the Safari gesture-canceller gate, and whether the
   * help overlay lists its Touch section; `touch` additionally applies the
   * mobile rendering budgets (adaptive-DPR floor and refresh ceiling, high-DPR
   * cap, GPU-byte and element-texture ceilings, data-worker count) and skips
   * the blend-variant program warm-up. Stylesheets and non-pen gesture routing
   * keep following the real media features and `PointerEvent.pointerType`, so a
   * faithful check still needs device emulation or a real device. `null`
   * (missing or unrecognised) ⇒ detect from the browser. See
   * `utils/input-capabilities.ts`.
   */
  input: InputProfileOverride | null;

  /**
   * Force a line join style for the session (`?lineJoin=none|miter`).
   *
   * Overrides whatever each node authored, which is exactly its purpose: it is
   * a debugging and workaround lever, so it must win over the scene file.
   * Precedence is `?lineJoin=` > authored node attribute > the built-in
   * default. `null` (missing or unrecognised) means "no override" — distinct
   * from `'none'`, which is an explicit request for no join geometry. See
   * `types/line-join.ts`.
   */
  lineJoin: LineJoinStyle | null;

  /**
   * Select the line rendering primitive for the session
   * (`?linePrimitive=screen-space|capsule`, issue #1352). The session's
   * strongest word: it overrides the `Advanced → Line primitive` policy
   * setting (whose `auto` mode otherwise sizes the scene before material
   * build — see `types/line-primitive.ts`).
   *
   * `capsule` (the default) profiles the 2D point-to-segment distance in
   * pixel space — direction-stable end-on, bisector-cut joins;
   * `screen-space` is the classic flat quad. Session-wide by design (a
   * renderer implementation choice, not scene content — there is no
   * authored per-node attribute). `null` (missing or unrecognised) means
   * the built-in default. See `types/line-primitive.ts`.
   */
  linePrimitive: LinePrimitive | null;

  /**
   * Bake the scene environment (`?bake-env`, driven by `luxar env bake`): once
   * the load settles, capture the scene-derived cube map at `probe` /
   * `envResolution`, expose the container on `__luxarDebug.environment.lastBake`
   * and download it. See `rendering/environment/bake.ts`.
   */
  bakeEnv: boolean;
  /** Probe for the bake (`?probe=auto|node:<path>|x,y,z`). Null → the scene's config or `auto`. */
  probe: string | null;
  /** Cube face size for the bake (`?env-resolution=128`). Null → the scene's config or 128. */
  envResolution: number | null;
}

/**
 * Parse the supplied query string (or `window.location.search` by default)
 * into a typed `UrlParams` snapshot.
 *
 * Pass an explicit `search` string in tests; in production main.ts calls this
 * once with no argument and threads the result through the rest of the app.
 */
export function readUrlParams(search?: string, origin?: ControlSocketOrigin): UrlParams {
  const raw = search ?? (typeof window !== 'undefined' ? window.location?.search : '') ?? '';
  const params = new URLSearchParams(raw);
  // The control socket's address is derived from the page's own origin, so the
  // validator needs it. Injectable for the same reason `search` is: a test (and
  // an embedder) must be able to parse without touching window.location.
  const pageOrigin = resolvePageOrigin(origin);
  const allowCrossOriginControl = params.has('controlAllowCrossOrigin');

  return {
    src: normalizeDataSourceUrl(params.get('src')),
    theme: params.get('theme'),
    title: params.get('title')?.trim() || null,
    control: normalizeControlSocketUrl(params.get('control'), pageOrigin, allowCrossOriginControl),
    controlToken: trimmedParam(params, 'controlToken'),
    controlAllowCrossOrigin: allowCrossOriginControl,
    panel: normalizePanelModuleUrl(params.get('panel'), pageOrigin),
    debug: params.has('debug'),
    kiosk: params.has('kiosk'),
    noCache: params.has('no-cache'),
    noSliceCache: params.has('no-slice-cache'),
    noOpfs: params.has('no-opfs'),
    opfsReadConcurrency: parsePositiveInt(params.get('opfsReadConcurrency')),
    cacheDebug: params.has('cache-debug'),
    clearCache: params.has('clear-cache'),
    lodFade: !params.has('no-lod-fade'),
    allowLinks: !params.has('no-links'),
    lodEnergyComp: !params.has('no-lod-energy'),
    lodFinest: params.has('lod-finest'),
    lodBias: parsePositiveFloat(params.get('lod-bias')),
    blendWarmup: !params.has('no-blend-warmup'),
    depthSort: parseEnabledFlag(params.get('depthSort')),
    densityGuard: !params.has('no-density-guard'),
    densityCap: parsePositiveFloat(params.get('density-cap')),
    noPrefetch: params.has('no-prefetch'),
    prefetchDebug: params.has('prefetch-debug'),
    cacheStats: params.has('cache-stats'),
    renderer: normalizeRendererParam(params.get('renderer')),
    webgpuForceWebGL: params.has('webgpu-force-webgl'),
    perfTimestamp: params.has('perf-timestamp'),
    gpuBudgetMB: parseNonNegativeInt(params.get('gpuBudgetMB')),
    cacheBudgetMB: parseNonNegativeInt(params.get('cacheBudgetMB')),
    dpr: parsePositiveFloat(params.get('dpr')),
    input: normalizeInputParam(params.get('input')),
    lineJoin: parseLineJoinStyle(params.get('lineJoin')),
    linePrimitive: parseLinePrimitive(params.get('linePrimitive')),
    bakeEnv: params.has('bake-env'),
    probe: params.get('probe')?.trim() || null,
    envResolution: clampEnvironmentResolution(parseNonNegativeInt(params.get('env-resolution'))),
  };
}

function clampEnvironmentResolution(value: number | null): number | null {
  if (value === null) return null;
  return Math.min(ENVIRONMENT_RESOLUTION_MAX, Math.max(ENVIRONMENT_RESOLUTION_MIN, value));
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

function parsePositiveInt(raw: string | null): number | null {
  if (raw === null) return null;
  const value = Number(raw);
  return Number.isInteger(value) && value > 0 ? value : null;
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
 * Validate the `?input=` query value. Accept `touch` and `mouse`
 * case-insensitively; everything else (including the flag-only `?input`
 * form) means "no override — detect".
 */
function normalizeInputParam(raw: string | null): InputProfileOverride | null {
  if (raw === null) return null;
  const v = raw.trim().toLowerCase();
  if (v === 'touch' || v === 'mouse') return v;
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
 *
 * `title` is the one parameter that does NOT survive: it names the dataset
 * the server started with, so carrying it onto a different `src` would make
 * a shared or reloaded URL title the tab after a scene it no longer shows.
 * The live tab is retitled for the incoming dataset at the same moment —
 * see `core/document-title.ts`.
 */
export function buildDataSourceBrowserUrl(src: string, location: BrowserUrlLocation): string {
  const params = new URLSearchParams(location.search);
  params.set('src', normalizeSrcForUrl(src));
  params.delete('title');
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
