/**
 * Renderer-construction helpers extracted from SceneManager.
 *
 * Three concerns live here:
 *
 *   - `selectBackend` — apply the URL-param / env-var / default
 *     precedence ladder to pick WebGL or WebGPU.
 *   - `createWebGLRenderer` — construct a `THREE.WebGLRenderer`
 *     from the configured WebGL2 context, build the renderer-
 *     capabilities snapshot, configure HDR.
 *   - `createWebGPURenderer` — construct a `WebGPURenderer`,
 *     request a high-limit core adapter / device (workaround for
 *     compat-mode limits in r184), init() it, and build the
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
import { WebGPURenderer } from 'three/webgpu';
import { config } from '../../../config';
import {
  createRendererCapabilities,
  type Renderer,
  type RendererCapabilities,
} from '../../../rendering/renderer-capabilities';
import { configureHDRRenderer, logHDRCapabilities } from '../../../utils/hdr/hdr-detection';
import { log, Modules } from '../../../utils/log';
import { notifier } from '../../../utils/cross-layer/notifier';

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
}

/**
 * Select the rendering backend.
 *
 * Precedence (highest to lowest):
 *   1. Per-instance `rendererOverride` (URL `?renderer=webgl|webgpu`).
 *   2. `VITE_LUXAR_USE_WEBGPU=1` / `VITE_LUXAR_USE_WEBGPU_RENDERER=1` env vars.
 *   3. `VITE_LUXAR_USE_LEGACY_WEBGL=1` env var (no-op alias for the default).
 *   4. Default: WebGL.
 */
export function selectBackend(rendererOverride: 'webgl' | 'webgpu' | undefined): BackendSelection {
  if (rendererOverride === 'webgl') return { backend: 'webgl', source: 'url-param' };
  if (rendererOverride === 'webgpu') return { backend: 'webgpu', source: 'url-param' };
  if (
    import.meta.env.VITE_LUXAR_USE_WEBGPU === '1' ||
    import.meta.env.VITE_LUXAR_USE_WEBGPU_RENDERER === '1'
  ) {
    return { backend: 'webgpu', source: 'env-var' };
  }
  if (import.meta.env.VITE_LUXAR_USE_LEGACY_WEBGL === '1') {
    return { backend: 'webgl', source: 'env-var' };
  }
  return { backend: 'webgl', source: 'default' };
}

/** Result of a successful renderer construction. */
export interface CreatedRenderer {
  renderer: Renderer;
  capabilities: RendererCapabilities;
}

/**
 * Construct a `THREE.WebGLRenderer` from a WebGL2 context obtained
 * via `canvas.getContext('webgl2', config.webgl.context)`.
 *
 * Returns the renderer plus a `RendererCapabilities` snapshot. The
 * caller wires materialManager / HDR / clear color / initial resize.
 */
export async function createWebGLRenderer(canvas: HTMLCanvasElement): Promise<CreatedRenderer> {
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

  const capabilities = createRendererCapabilities(renderer);

  log.info(Modules.RENDERER, `Rendering API: ${capabilities.apiSurface}`);

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
}

/**
 * Result of `createWebGPURenderer`. Either a successful WebGPU
 * renderer + capabilities pair OR a signal that the WebGPU adapter
 * was below the spec minimum and the caller should fall back to
 * `createWebGLRenderer`.
 */
export type CreateWebGPUResult = (CreatedRenderer & { fallback: false }) | { fallback: true };

/**
 * Construct a `WebGPURenderer`. Negotiates a "core" adapter with
 * raised vertex-buffer / buffer-size limits to bypass r184's
 * compat-mode defaults. Performs `await renderer.init()` before
 * returning.
 *
 * When the adapter advertises `maxVertexBuffers < 8` (WebGPU spec
 * minimum) and `rendererOverride !== 'webgpu'`, returns
 * `{ fallback: true }` so the caller can drop to WebGL instead.
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
  // implementation: r184 hard-codes compat mode which caps
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
    const adapterMaxBufferSize = adapter.limits?.maxBufferSize;
    const adapterMaxStorageBuffer = adapter.limits?.maxStorageBufferBindingSize;

    const requiredFeatures: string[] = [];
    for (const name of adapter.features) {
      requiredFeatures.push(name);
    }
    const requiredLimits: Record<string, number> = {};
    if (requestedMax !== undefined) requiredLimits.maxVertexBuffers = requestedMax;
    if (typeof adapterMaxBufferSize === 'number') {
      requiredLimits.maxBufferSize = adapterMaxBufferSize;
    }
    if (typeof adapterMaxStorageBuffer === 'number') {
      requiredLimits.maxStorageBufferBindingSize = adapterMaxStorageBuffer;
    }
    log.info(
      Modules.RENDERER,
      `WebGPU adapter advertises maxVertexBuffers=${adapterMax}, ` +
        `maxBufferSize=${adapterMaxBufferSize}, ` +
        `maxStorageBufferBindingSize=${adapterMaxStorageBuffer}; ` +
        `requesting maxVertexBuffers=${requestedMax}, ` +
        `maxBufferSize=${requiredLimits.maxBufferSize ?? 'default'}, ` +
        `maxStorageBufferBindingSize=${requiredLimits.maxStorageBufferBindingSize ?? 'default'}`
    );
    try {
      device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
    } catch (err) {
      log.warning(
        Modules.RENDERER,
        `WebGPU requestDevice failed (${err}); falling through to Three's internal compat-mode init.`
      );
    }
  }

  const gpuRenderer = new WebGPURenderer({
    canvas,
    antialias: config.webgl.context.antialias,
    alpha: config.webgl.context.alpha,
    ...(webgpuForceWebGL ? { forceWebGL: true } : device !== undefined ? { device } : {}),
    ...(perfTimestamp ? { trackTimestamp: true } : {}),
  } as ConstructorParameters<typeof WebGPURenderer>[0]);
  await gpuRenderer.init();

  const capabilities = createRendererCapabilities(gpuRenderer);
  log.info(Modules.RENDERER, `Rendering API: ${capabilities.apiSurface}`);

  const hdrCapabilities = capabilities.hdr;
  logHDRCapabilities(hdrCapabilities);
  configureHDRRenderer(gpuRenderer, hdrCapabilities);

  return { fallback: false, renderer: gpuRenderer, capabilities };
}
