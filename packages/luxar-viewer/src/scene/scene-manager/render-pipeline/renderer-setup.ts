/**
 * Renderer-construction helpers extracted from SceneManager.
 *
 * Four concerns live here:
 *
 *   - `configureRendererBackend` — point the backend-dependent rendering
 *     paths (texture layout, ordering applies, RenderObject eviction,
 *     element-texture row uploads) at a renderer; shared with layer mode.
 *   - `selectBackend` — apply the URL-param / env-var / default
 *     precedence ladder to pick WebGL or WebGPU.
 *   - `createWebGLRenderer` — construct a `THREE.WebGLRenderer`
 *     from the configured WebGL2 context, build the renderer-
 *     capabilities snapshot, configure HDR.
 *   - `createWebGPURenderer` — construct a `WebGPURenderer`,
 *     request a high-limit core adapter / device (workaround for
 *     compat-mode limits), init() it, and build the
 *     capabilities snapshot.
 *
 * Each `create*Renderer` function returns
 * `{ renderer, capabilities }`. Host (SceneManager) is responsible
 * for the post-construction wiring (materialManager.setCaps,
 * resizer initial-size, setClearColor) so the helpers stay focused
 * on graphics-API negotiation.
 *
 * @module scene/scene-manager/render-pipeline/renderer-setup
 */

import * as THREE from 'three';
import { config } from '../../../config';
import {
  createRendererCapabilities,
  probeUnmaskedGpuRenderer,
  type Renderer,
  type RendererCapabilities,
} from '../../../rendering/renderer-capabilities';
import { configureElementTextureLayout } from '../../../rendering/element-texture-layout';
import { configureSortedIndexChunkedApply } from '../../../rendering/element-storage';
import { installUploadCounters } from '../../../rendering/upload-counters';
import { installElementTextureRowUploads } from '../../../rendering/element-texture-row-upload';
import { configureRenderObjectEviction } from '../../../data/scene-loader/commit/invalidate-render-object';
import {
  configureHDRRenderer,
  detectDisplayCapabilities,
  logHDRCapabilities,
} from '../../../utils/hdr/hdr-detection';
import { resolveHdrHeadroom } from '../../../utils/hdr/hdr-headroom';
import { platformName } from '../../../utils/platform';
import { log, Modules } from '../../../utils/log';
import {
  clampHdrHeadroom,
  DEFAULT_HDR_HEADROOM,
  type HdrOutputState,
} from '../../../types/hdr-output';
import { notifier } from '../../../utils/cross-layer/notifier';

/**
 * Point the backend-dependent rendering paths at `renderer`. Every renderer
 * Luxar draws through needs this — the app's own (both `create*Renderer`
 * below) and a host's in layer mode (`core/layer/luxar-layer.ts`) — because
 * each switch is a module default tuned for classic WebGL:
 *
 *   - element textures are laid out against the device's `maxTextureSize`;
 *   - both WebGPU backends ignore texture update ranges, so element textures
 *     upload only their dirty rows there (#2944) — a no-op on classic WebGL;
 *   - chunked ordering applies need honoured attribute update ranges and the
 *     `onUploadCallback` back-pressure, which neither WebGPU backend fires —
 *     classic WebGL only (see element-storage's chunked-apply note);
 *   - RenderObject eviction (dispose dispatch) flushes the stale vertexBuffers
 *     chainMap cache only WebGPU keeps; on classic WebGL the same event
 *     destroys the compiled program (a shader recompile), so WebGPU only.
 *
 * Call after the renderer is initialised (WebGPU's backend exists only after
 * `init()`).
 *
 * The switches are page-wide and the last call wins, so every Luxar host on a
 * page (a LuxarApp, LuxarLayers) must render through ONE backend. Not enforced
 * — a page may re-init on another backend once the first host is gone — but a
 * call that switches the backend warns once.
 */
export function configureRendererBackend(
  renderer: Renderer,
  capabilities: RendererCapabilities
): void {
  warnOnBackendMix(capabilities.apiSurface);
  configureElementTextureLayout(capabilities.maxTextureSize);
  installElementTextureRowUploads(renderer);
  configureSortedIndexChunkedApply(capabilities.apiSurface === 'webgl2');
  configureRenderObjectEviction(capabilities.apiSurface === 'webgpu');
}

/** The backend the page-wide switches were last configured for. */
let configuredApiSurface: RendererCapabilities['apiSurface'] | null = null;
let warnedBackendMix = false;

function warnOnBackendMix(apiSurface: RendererCapabilities['apiSurface']): void {
  const previous = configuredApiSurface;
  configuredApiSurface = apiSurface;
  if (previous === null || previous === apiSurface || warnedBackendMix) return;
  warnedBackendMix = true;
  log.warning(
    Modules.RENDERER,
    `Renderer backend switched from ${previous} to ${apiSurface}: the backend switches are ` +
      'page-wide, so if another Luxar host on this page still renders through ' +
      `${previous} it now runs on ${apiSurface}'s settings. Use one backend per page.`
  );
}

/**
 * URL-param / env-var / default precedence ladder for backend
 * selection. The result drives whether `createWebGLRenderer` or
 * `createWebGPURenderer` is invoked. Kept as a pure function so
 * the selection logic is unit-testable independent of WebGL /
 * WebGPU construction.
 */
export interface BackendSelection {
  /** Backend to use: 'webgl' or 'webgpu'. */
  backend: 'webgl' | 'webgpu';
  /** Where the selection came from, for logging. */
  source: 'url-param' | 'env-var' | 'default';
  /**
   * WebGPU only on a NATIVE adapter: without one, use `WebGLRenderer` rather
   * than letting three run TSL on its internal WebGL2 backend. True for every
   * selection except an explicit `?renderer=webgpu`, which keeps the WebGL2
   * bridge reachable for diagnostics (and the render gate's `webgpu-gl` arm).
   */
  requireNativeAdapter: boolean;
}

/**
 * The backend a session gets when nothing selects one. WebGL until native
 * WebGPU meets the go/no-go criteria of issue #3146; a `'webgpu'` default
 * still falls back to `WebGLRenderer` wherever no native adapter exists.
 */
const DEFAULT_BACKEND: BackendSelection['backend'] = 'webgl';

/**
 * Select the rendering backend.
 *
 * Precedence (highest to lowest):
 *   1. Per-instance `rendererOverride` (URL `?renderer=webgl|webgpu`, or the
 *      Settings › Advanced › Renderer choice).
 *   2. `VITE_LUXAR_USE_WEBGPU=1` / `VITE_LUXAR_USE_WEBGPU_RENDERER=1` env vars.
 *   3. `VITE_LUXAR_USE_LEGACY_WEBGL=1` env var (forces WebGL).
 *   4. `defaultBackend` (`DEFAULT_BACKEND`, WebGL today).
 *
 * Only an explicit `'webgpu'` override may run on three's WebGL2 bridge; every
 * other WebGPU selection requires a native adapter.
 */
export function selectBackend(
  rendererOverride: 'webgl' | 'webgpu' | undefined,
  defaultBackend: BackendSelection['backend'] = DEFAULT_BACKEND
): BackendSelection {
  if (rendererOverride === 'webgl') {
    return { backend: 'webgl', source: 'url-param', requireNativeAdapter: true };
  }
  if (rendererOverride === 'webgpu') {
    return { backend: 'webgpu', source: 'url-param', requireNativeAdapter: false };
  }
  if (
    import.meta.env.VITE_LUXAR_USE_WEBGPU === '1' ||
    import.meta.env.VITE_LUXAR_USE_WEBGPU_RENDERER === '1'
  ) {
    return { backend: 'webgpu', source: 'env-var', requireNativeAdapter: true };
  }
  if (import.meta.env.VITE_LUXAR_USE_LEGACY_WEBGL === '1') {
    return { backend: 'webgl', source: 'env-var', requireNativeAdapter: true };
  }
  return { backend: defaultBackend, source: 'default', requireNativeAdapter: true };
}

/**
 * Adapter limits forwarded verbatim into the device's `requiredLimits`
 * when the adapter advertises them as numbers (absent / non-numeric
 * entries are omitted so the device keeps the spec default).
 *
 * `maxTextureDimension2D` is the per-node element-data ceiling:
 * element textures (points/lines/gsplats) grow DOWNWARD in rows, and
 * `renderer-capabilities` reads the LIVE DEVICE limit to derive every
 * capacity clamp. The WebGPU default is 8192 rows — for lines
 * (6 texels/segment, 682 segments/row at width 4096) that caps a node
 * at 5,586,944 segments while the same GPU's WebGL context exposes
 * 16384 rows = 11.2 M. Same forward-the-adapter policy as the buffer
 * limits.
 *
 * `maxVertexBuffers` is deliberately NOT here — it is clamped to 16
 * rather than forwarded verbatim (see `createWebGPURenderer`).
 */
export const FORWARDED_ADAPTER_LIMIT_NAMES = [
  'maxBufferSize',
  'maxStorageBufferBindingSize',
  'maxTextureDimension2D',
] as const;

/** One of the adapter limits forwarded verbatim into `requiredLimits`. */
export type ForwardedAdapterLimitName = (typeof FORWARDED_ADAPTER_LIMIT_NAMES)[number];

/**
 * Pick the {@link FORWARDED_ADAPTER_LIMIT_NAMES} entries the adapter
 * advertises as numbers, shaped for a `requestDevice` descriptor's
 * `requiredLimits`. Absent / non-numeric entries are omitted so the
 * device keeps the WebGPU spec default for them.
 */
export function forwardedAdapterLimits(
  // Structural over the three explicit names only — a real
  // `GPUSupportedLimits` (no string index signature) satisfies this,
  // so the signature survives a future move off the file-local
  // `GPUAdapterLike` shim onto @webgpu/types.
  limits: Partial<Record<ForwardedAdapterLimitName, number>> | undefined
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const key of FORWARDED_ADAPTER_LIMIT_NAMES) {
    const value = limits?.[key];
    if (typeof value === 'number') out[key] = value;
  }
  return out;
}

/** Result of a successful renderer construction. */
export interface CreatedRenderer {
  renderer: Renderer;
  capabilities: RendererCapabilities;
}

/** Options that `createWebGLRenderer` honours. */
export interface CreateWebGLOptions {
  /**
   * Debug mode (`?debug`). Keeps three's `checkShaderErrors` on, so a broken
   * shader gets three's formatted report with the offending source lines.
   */
  debug?: boolean;
}

/**
 * Construct a `THREE.WebGLRenderer` from a WebGL2 context obtained
 * via `canvas.getContext('webgl2', config.webgl.context)`.
 *
 * Outside debug mode `renderer.debug.checkShaderErrors` is turned OFF. With it
 * on, three reads `getProgramInfoLog` and both shader logs straight after every
 * link — a synchronous GPU round trip that defeats the driver's parallel
 * compile and measured 15-91 ms of main-thread stall on a cold start. A broken
 * shader still surfaces without it: the browser logs the failed link as a WebGL
 * warning and the draw is skipped.
 *
 * Returns the renderer plus a `RendererCapabilities` snapshot. The
 * caller wires materialManager / HDR / clear color / initial resize.
 */
export async function createWebGLRenderer(
  canvas: HTMLCanvasElement,
  options: CreateWebGLOptions = {}
): Promise<CreatedRenderer> {
  // Try to get HDR canvas context first using config values.
  //
  // Allow-list rule: a `getContext` call is permitted ONLY if it
  // runs before the renderer exists (no `capabilities` to route
  // through yet).
  let gl: WebGLRenderingContext | null = null;
  try {
    gl = canvas.getContext('webgl2', config.webgl.context) as WebGLRenderingContext | null;

    if (!gl) {
      log.warning(Modules.SCENE_MANAGER, 'WebGL2 context creation failed, falling back to default');
    }
  } catch (error) {
    log.error(Modules.SCENE_MANAGER, 'Error creating WebGL2 context:', error);
    notifier.error('Failed to create WebGL2 context. Your browser may not support WebGL2.');
  }

  const renderer = new THREE.WebGLRenderer({
    canvas,
    context: gl || undefined,
    alpha: config.webgl.context.alpha,
    antialias: config.webgl.context.antialias,
    depth: config.webgl.context.depth,
    stencil: config.webgl.context.stencil,
    powerPreference: config.webgl.context.powerPreference,
    preserveDrawingBuffer: config.webgl.context.preserveDrawingBuffer,
    premultipliedAlpha: config.webgl.context.premultipliedAlpha,
    ...config.webgl.renderer,
  });
  renderer.debug.checkShaderErrors = options.debug === true;

  // Tally every CPU→GPU upload for the perf counters (instrumentation only).
  installUploadCounters(renderer);

  const capabilities = createRendererCapabilities(renderer);
  configureRendererBackend(renderer, capabilities);

  log.info(
    Modules.RENDERER,
    `Rendering API: ${capabilities.apiSurface} (${capabilities.backendKind})`
  );

  const hdrCapabilities = capabilities.hdr;
  logHDRCapabilities(hdrCapabilities);
  configureHDRRenderer(renderer, hdrCapabilities);

  return { renderer, capabilities };
}

/**
 * Options that `createWebGPURenderer` honours.
 */
export interface CreateWebGPUOptions {
  /** When true, log hardware point-size limits for diagnostics. */
  debug?: boolean;
  /**
   * Diagnostic mode: keep `WebGPURenderer` + TSL `NodeMaterial`
   * dispatch but make Three use its internal WebGL2 backend.
   * Mutually exclusive with a pre-built `device`.
   */
  webgpuForceWebGL?: boolean;
  /**
   * Opt into GPU timestamp queries. Three.js gates the pool's
   * allocation on `device.features.has('timestamp-query')`, so
   * setting this is safe on drivers without the feature.
   */
  perfTimestamp?: boolean;
  /**
   * Honour an explicit `?renderer=webgpu` override even when the
   * adapter advertises maxVertexBuffers < 8. Used by the WebGPU
   * auto-fallback logic to decide whether to drop to WebGL or
   * push through on a degenerate adapter.
   */
  rendererOverride?: 'webgl' | 'webgpu';
  /**
   * Return `{ fallback: true }` when no native WebGPU device can be obtained
   * (no `navigator.gpu`, no adapter, or `requestDevice` failed) instead of
   * constructing a `WebGPURenderer` that three would run on its WebGL2
   * backend. Ignored with `webgpuForceWebGL`, which asks for that backend.
   */
  requireNativeAdapter?: boolean;
  /** PROTOTYPE: request extended-range (HDR) output (`?hdr`); gated by `planHdrOutput`. */
  hdr?: boolean;
  /** Pinned HDR headroom (`?hdr=<n>`); omitted = resolved per display. */
  hdrHeadroom?: number;
}

/**
 * Whether this browser honours `toneMapping: { mode: 'extended' }` on an
 * `rgba16float` canvas. Probed on a throwaway 1x1 canvas with the device the
 * renderer will use, before the renderer exists (three configures the real
 * canvas lazily, at the first frame). An unknown dictionary member is dropped
 * silently, so the configuration is read back; a browser without
 * `getConfiguration()` cannot be verified and counts as unsupported, and one
 * that rejects the format throws here, harmlessly.
 */
function supportsExtendedToneMapping(device: unknown): boolean {
  try {
    const probe = document.createElement('canvas');
    probe.width = probe.height = 1;
    const ctx = probe.getContext('webgpu') as {
      configure(config: Record<string, unknown>): void;
      unconfigure?: () => void;
      getConfiguration?: () => { toneMapping?: { mode?: string } } | null;
    } | null;
    if (!ctx || device === undefined) return false;
    ctx.configure({ device, format: 'rgba16float', toneMapping: { mode: 'extended' } });
    const mode = ctx.getConfiguration?.()?.toneMapping?.mode;
    ctx.unconfigure?.();
    return mode === 'extended';
  } catch {
    return false;
  }
}

/**
 * Decide, before the renderer exists, whether the HDR output requested by
 * `?hdr` can be honoured, and with which starting headroom. Undefined means
 * ordinary SDR output.
 *
 * Needs native WebGPU: only `GPUCanvasContext.configure` takes the
 * `toneMapping: { mode: 'extended' }` that lets canvas values above 1.0 reach
 * the display, and three only asks for it with `outputType: HalfFloatType`
 * (an `rgba16float` canvas). Its WebGL2 backend has no such switch, so no
 * adapter (three would fall back to it) means no HDR. And the display must
 * report `(dynamic-range: high)`: on an SDR display an extended canvas just
 * clips at 1.0, which is strictly worse than the SDR tone curve.
 */
function planHdrOutput(options: CreateWebGPUOptions, device: unknown): HdrOutputState | undefined {
  if (options.hdr !== true) return undefined;
  const blocker = hdrOutputBlocker(options, device);
  if (blocker !== null) {
    log.info(Modules.HDR, `${blocker}; using an SDR canvas.`);
    return undefined;
  }
  const pinned = options.hdrHeadroom !== undefined;
  // Only the Apple-silicon rule reads the GPU string, and only when nothing
  // pinned the headroom, so its throwaway WebGL2 context is rarely paid.
  const gpuRenderer = !pinned && platformName() === 'macOS' ? probeUnmaskedGpuRenderer() : null;
  const headroom = pinned
    ? clampHdrHeadroom(options.hdrHeadroom ?? DEFAULT_HDR_HEADROOM)
    : resolveHdrHeadroom(gpuRenderer);
  return { headroom, pinned, gpuRenderer };
}

/** Why requested HDR output cannot be honoured, or null when it can. */
function hdrOutputBlocker(options: CreateWebGPUOptions, device: unknown): string | null {
  if (options.hdrHeadroom !== undefined && clampHdrHeadroom(options.hdrHeadroom) <= 1) {
    return 'Headroom pinned to 1x (no range above SDR white)';
  }
  if (options.webgpuForceWebGL === true || device === undefined) {
    return 'HDR output needs a native WebGPU device';
  }
  if (!detectDisplayCapabilities().hdr) {
    return 'Display does not report HDR (dynamic-range: high)';
  }
  // Probed BEFORE the renderer exists: `outputType` commits the real canvas to
  // rgba16float, so a browser that cannot do extended tone mapping must never
  // get one (it would pay double swapchain bandwidth — measured +0.22 ms/frame
  // at 4K — or fail to configure the format at its first frame).
  if (!supportsExtendedToneMapping(device)) return 'Canvas extended tone mapping unavailable';
  return null;
}

/**
 * After `init()`: keep the HDR plan only if three really runs its WebGPU
 * backend; otherwise fall back to SDR tone mapping (values stay in [0, 1]).
 * A defensive check — the plan already required a native device, which three
 * uses as given.
 */
function confirmHdrOutput(
  renderer: Renderer,
  plan: HdrOutputState | undefined
): HdrOutputState | undefined {
  if (plan === undefined) return undefined;
  const backend = (renderer as { backend?: { isWebGPUBackend?: boolean } }).backend;
  if (backend?.isWebGPUBackend !== true) {
    log.info(Modules.HDR, 'WebGPURenderer runs its WebGL2 backend; using SDR output.');
    return undefined;
  }
  log.info(
    Modules.HDR,
    `HDR output: rgba16float extended-range canvas, headroom ${plan.headroom.toFixed(2)}x SDR white` +
      (plan.pinned ? ' (pinned)' : '')
  );
  return plan;
}

/**
 * True (and logged) when `requireNativeAdapter` asked for a native WebGPU
 * device and none was obtained, so the caller falls back to WebGL.
 */
function lacksRequiredNativeDevice(
  options: CreateWebGPUOptions,
  hasGpu: boolean,
  hasAdapter: boolean,
  device: unknown
): boolean {
  if (device !== undefined || options.requireNativeAdapter !== true) return false;
  if (options.webgpuForceWebGL === true) return false;
  const why = !hasGpu
    ? 'navigator.gpu is unavailable'
    : hasAdapter
      ? 'requestDevice failed'
      : 'no adapter';
  log.info(Modules.RENDERER, `No native WebGPU device (${why}); using WebGLRenderer.`);
  return true;
}

/**
 * What three hands `renderer.onError` for an uncaptured WebGPU error
 * (`device.onuncapturederror` in `WebGPUBackend.init`): the error class name
 * (`GPUValidationError`, `GPUOutOfMemoryError`, `GPUInternalError`) and its
 * message. `@types/three` 0.185 declares the parameter as a plain `string`,
 * which the r185 runtime never passes; the handler accepts both.
 */
export interface UncapturedGpuErrorInfo {
  api: string;
  type: string;
  message: string;
  originalEvent?: unknown;
}

/** Identical errors logged in full before the router goes quiet on that message. */
export const GPU_ERROR_REPEAT_LIMIT = 3;
/** Distinct messages tracked; past this every new message is suppressed too. */
export const GPU_ERROR_DISTINCT_LIMIT = 32;

/**
 * Build the `renderer.onError` handler: each uncaptured GPU error goes to
 * `log.error(Modules.RENDERER, …)` instead of three's bare console `error()`.
 *
 * A validation error raised by a pipeline or bind group usually fires on EVERY
 * frame, which would bury the console at 60 lines a second. So the first
 * {@link GPU_ERROR_REPEAT_LIMIT} occurrences of each distinct message are
 * logged, followed by one note that further copies are suppressed; after
 * {@link GPU_ERROR_DISTINCT_LIMIT} distinct messages a single note stops all
 * further reporting. The first occurrence is what carries the information;
 * an occurrence count is not worth the console it costs.
 */
export function createUncapturedGpuErrorHandler(): (info: UncapturedGpuErrorInfo | string) => void {
  const seen = new Map<string, number>();
  let saturated = false;
  return (info) => {
    const text =
      typeof info === 'string' ? info : `Uncaptured ${info.api} ${info.type}: ${info.message}`;
    const count = (seen.get(text) ?? 0) + 1;
    if (count === 1 && seen.size >= GPU_ERROR_DISTINCT_LIMIT) {
      if (!saturated) {
        saturated = true;
        log.error(
          Modules.RENDERER,
          `${GPU_ERROR_DISTINCT_LIMIT} distinct GPU errors reported; suppressing all further GPU errors`
        );
      }
      return;
    }
    seen.set(text, count);
    if (count <= GPU_ERROR_REPEAT_LIMIT) {
      log.error(Modules.RENDERER, text);
    } else if (count === GPU_ERROR_REPEAT_LIMIT + 1) {
      log.error(Modules.RENDERER, `Suppressing further repeats of: ${text}`);
    }
  };
}

/**
 * Result of `createWebGPURenderer`. Either a successful WebGPU
 * renderer + capabilities pair OR a signal that the caller should fall
 * back to `createWebGLRenderer`: the adapter was below the spec minimum,
 * or a native adapter was required and none was available.
 */
export type CreateWebGPUResult = (CreatedRenderer & { fallback: false }) | { fallback: true };

/**
 * Construct a `WebGPURenderer`. Negotiates a "core" adapter with
 * raised vertex-buffer / buffer-size / texture-dimension limits to
 * bypass three's compat-mode defaults (see
 * {@link forwardedAdapterLimits}). Performs `await renderer.init()`
 * before returning.
 *
 * Returns `{ fallback: true }` so the caller can drop to WebGL when the
 * adapter advertises `maxVertexBuffers < 8` (WebGPU spec minimum) and
 * `rendererOverride !== 'webgpu'`, or when `requireNativeAdapter` is set and
 * no native device was obtained. Both return before `three/webgpu` is
 * imported, so a fallback session never downloads that chunk.
 */
export async function createWebGPURenderer(
  canvas: HTMLCanvasElement,
  options: CreateWebGPUOptions = {}
): Promise<CreateWebGPUResult> {
  const { webgpuForceWebGL = false, perfTimestamp = false, rendererOverride } = options;

  log.info(Modules.SCENE_MANAGER, 'Constructing WebGPURenderer…');
  if (webgpuForceWebGL) {
    log.info(
      Modules.RENDERER,
      'WebGPURenderer forceWebGL enabled: using Three.js WebGL2 backend with TSL materials.'
    );
  }

  // ============================================================
  // Bypass Three.js's `featureLevel: 'compatibility'` default.
  // See the long-form rationale in the original setupWebGPURenderer
  // implementation: three still hard-codes compat mode at the pinned r185,
  // as it did at r184, and that caps
  // `maxVertexBuffers=8`, breaking the line material that binds 12+
  // attributes on hardware that natively supports many more.
  // Workaround: build a core adapter + device with raised limits
  // and pass the device to `WebGPURenderer`.
  // ============================================================
  type GPUAdapterLike = {
    readonly features: ReadonlySet<string>;
    readonly limits: Record<string, number | undefined>;
    requestDevice: (descriptor: {
      requiredFeatures?: string[];
      requiredLimits?: Record<string, number>;
    }) => Promise<unknown>;
  };
  type NavigatorWithGPU = Navigator & {
    gpu?: {
      requestAdapter?: (options?: {
        featureLevel?: 'core' | 'compatibility';
        powerPreference?: 'low-power' | 'high-performance';
      }) => Promise<GPUAdapterLike | null>;
    };
  };
  const gpu = webgpuForceWebGL ? undefined : (navigator as NavigatorWithGPU).gpu;
  let adapter: GPUAdapterLike | null = null;
  if (gpu?.requestAdapter) {
    try {
      adapter = await gpu.requestAdapter({
        featureLevel: 'core',
        powerPreference: 'high-performance',
      });
    } catch {
      // Older browsers reject the `featureLevel` option — fall through.
    }
    if (!adapter) {
      adapter = await gpu.requestAdapter().catch(() => null);
    }
  }

  const LUXAR_MIN_VERTEX_BUFFERS_REQUIRED = 8;

  let device: unknown;
  let adapterMax: number | undefined;
  if (adapter) {
    adapterMax = adapter.limits?.maxVertexBuffers;

    // Auto-fallback only when even the spec minimum isn't met.
    if (
      typeof adapterMax === 'number' &&
      adapterMax < LUXAR_MIN_VERTEX_BUFFERS_REQUIRED &&
      rendererOverride !== 'webgpu'
    ) {
      log.warning(
        Modules.RENDERER,
        `WebGPU adapter advertises maxVertexBuffers=${adapterMax}, below the WebGPU spec minimum ` +
          `of ${LUXAR_MIN_VERTEX_BUFFERS_REQUIRED}. Auto-falling back to the default WebGLRenderer ` +
          'path. Override with `?renderer=webgpu` to force WebGPU anyway (for debugging).'
      );
      return { fallback: true };
    }

    const requestedMax = typeof adapterMax === 'number' ? Math.min(adapterMax, 16) : undefined;

    const requiredFeatures: string[] = [];
    for (const name of adapter.features) {
      requiredFeatures.push(name);
    }
    // Buffer + element-texture ceilings forwarded from the adapter —
    // see forwardedAdapterLimits for the row-ceiling rationale.
    const requiredLimits: Record<string, number> = forwardedAdapterLimits(adapter.limits);
    if (requestedMax !== undefined) requiredLimits.maxVertexBuffers = requestedMax;
    log.info(
      Modules.RENDERER,
      `WebGPU adapter advertises maxVertexBuffers=${adapterMax}, ` +
        `maxBufferSize=${adapter.limits?.maxBufferSize}, ` +
        `maxStorageBufferBindingSize=${adapter.limits?.maxStorageBufferBindingSize}, ` +
        `maxTextureDimension2D=${adapter.limits?.maxTextureDimension2D}; ` +
        `requesting maxVertexBuffers=${requestedMax}, ` +
        `maxBufferSize=${requiredLimits.maxBufferSize ?? 'default'}, ` +
        `maxStorageBufferBindingSize=${requiredLimits.maxStorageBufferBindingSize ?? 'default'}, ` +
        `maxTextureDimension2D=${requiredLimits.maxTextureDimension2D ?? 'default'}`
    );
    try {
      device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
    } catch (err) {
      log.warning(
        Modules.RENDERER,
        `WebGPU requestDevice failed (${err}).`
      );
    }
  }

  if (lacksRequiredNativeDevice(options, gpu !== undefined, adapter !== null, device)) {
    return { fallback: true };
  }

  // Loaded HERE, not at module scope: a static `import { WebGPURenderer }`
  // makes the ~182 kB gzipped `three-webgpu` chunk a dependency of the entry
  // bundle, which every WebGL session then downloads and never runs (issue
  // #1679). `selectBackend()` is pure and synchronous and has already chosen
  // WebGPU by the time we get here, so this await costs the WebGPU path one
  // chunk fetch and the default path nothing.
  const { WebGPURenderer } = await import('three/webgpu');
  const hdrPlan = planHdrOutput(options, device);

  const gpuRenderer = new WebGPURenderer({
    canvas,
    antialias: config.webgl.context.antialias,
    alpha: config.webgl.context.alpha,
    ...(webgpuForceWebGL ? { forceWebGL: true } : device !== undefined ? { device } : {}),
    ...(perfTimestamp ? { trackTimestamp: true } : {}),
    // rgba16float canvas + `toneMapping.mode = 'extended'` (WebGPUBackend).
    ...(hdrPlan !== undefined ? { outputType: THREE.HalfFloatType } : {}),
  } as ConstructorParameters<typeof WebGPURenderer>[0]);
  // Before init(): three wires `device.onuncapturederror` to this callback
  // there. Only the native backend ever calls it; the WebGL2 bridge has no
  // uncaptured-error channel.
  gpuRenderer.onError = createUncapturedGpuErrorHandler();
  await gpuRenderer.init();
  // After init(): the backend's device / WebGL2 context exists only now.
  installUploadCounters(gpuRenderer);

  const hdrOutput = confirmHdrOutput(gpuRenderer, hdrPlan);
  const capabilities: RendererCapabilities = {
    ...createRendererCapabilities(gpuRenderer),
    ...(hdrOutput !== undefined ? { hdrOutput } : {}),
  };
  configureRendererBackend(gpuRenderer, capabilities);
  log.info(
    Modules.RENDERER,
    `Rendering API: ${capabilities.apiSurface} (${capabilities.backendKind})`
  );

  const hdrCapabilities = capabilities.hdr;
  logHDRCapabilities(hdrCapabilities);
  configureHDRRenderer(gpuRenderer, hdrCapabilities);

  return { fallback: false, renderer: gpuRenderer, capabilities };
}
