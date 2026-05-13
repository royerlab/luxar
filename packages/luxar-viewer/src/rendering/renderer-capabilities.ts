/**
 * Renderer capabilities — the single seam between Luxar and the raw
 * graphics API.
 *
 * Every other module in the codebase asks `RendererCapabilities` for
 * what the GPU can do, what bit depth the backbuffer has, or to read
 * pixels back. The implementation here speaks WebGL2. When the WebGPU
 * port lands, only this file changes.
 */
import * as THREE from 'three';
import type { WebGPURenderer } from 'three/webgpu';

import { detectDisplayCapabilities, type HDRCapabilities } from '../utils/hdr-detection';

export type { WebGPURenderer } from 'three/webgpu';

/**
 * The graphics-API renderer Luxar uses.
 *
 * `THREE.WebGLRenderer` is the WebGL2 path; `WebGPURenderer` (with
 * `forceWebGL: true` until the TSL ports complete) is the WebGPU
 * path. Consumers hold renderer references typed as `Renderer` so
 * future widening (post-port: drop the WebGLRenderer arm entirely)
 * is a one-line change here.
 */
export type Renderer = THREE.WebGLRenderer | WebGPURenderer;

/**
 * Shape-narrow a `Renderer` to `THREE.WebGLRenderer`. Used in
 * `createRendererCapabilities` to gate raw-GL probes. The check
 * is structural rather than `instanceof` so we don't have to
 * runtime-import `WebGPURenderer` just to type-test against it
 * (the import would force eager loading of `three/webgpu` even
 * when WebGL is the active backend).
 */
function isWebGLRenderer(renderer: Renderer): renderer is THREE.WebGLRenderer {
  return typeof (renderer as THREE.WebGLRenderer).getContext === 'function';
}

/**
 * What downstream code needs to know about the underlying graphics
 * stack. All static fields are captured once at construction.
 */
export interface RendererCapabilities {
  /** Underlying graphics API. Discriminator for callers that must branch. */
  readonly api: 'webgl2' | 'webgpu';
  /** HDR / wide-gamut / float-texture detection. */
  readonly hdr: HDRCapabilities;
  /** Maximum MSAA sample count the GPU supports (0 if unsupported). */
  readonly maxMSAASamples: number;
  /** `[min, max]` `gl_PointSize` range — used for debug logging. */
  readonly pointSizeRange: readonly [number, number];

  /**
   * Read the current canvas backbuffer into a freshly-allocated
   * `Uint8Array` (RGBA bytes, not vertically flipped).
   *
   * Returns a Promise so the WebGPU port (which requires async
   * `buffer.mapAsync`) replaces only this method's body, not the
   * interface contract. Under WebGL2 the inner work is synchronous
   * and the Promise resolves immediately.
   *
   * Implementations are responsible for binding the canvas
   * backbuffer before reading.
   */
  readBackbufferPixels(): Promise<{ pixels: Uint8Array; width: number; height: number }>;
}

/**
 * Build a `RendererCapabilities` snapshot from a constructed
 * renderer. Call once after renderer init; pass the result to
 * consumers (PostProcessingManager, SceneManager, …).
 */
export function createRendererCapabilities(renderer: Renderer): RendererCapabilities {
  const display = detectDisplayCapabilities();

  if (isWebGLRenderer(renderer)) {
    // WebGL2 path: probe raw-GL for capabilities. This is the only
    // place in the codebase that calls `getContext()` post-renderer
    // (the canvas-side pre-renderer call in `scene-manager` and the
    // probe in `webgpu-availability` are the documented exceptions).
    const gl = renderer.getContext() as WebGL2RenderingContext;

    const maxMSAASamplesRaw = gl.getParameter(gl.MAX_SAMPLES) as number | null;
    const maxMSAASamples = typeof maxMSAASamplesRaw === 'number' ? maxMSAASamplesRaw : 0;

    const rawRange = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE);
    const pointSizeRange: readonly [number, number] =
      rawRange &&
      typeof (rawRange as ArrayLike<number>)[0] === 'number' &&
      typeof (rawRange as ArrayLike<number>)[1] === 'number'
        ? [(rawRange as ArrayLike<number>)[0], (rawRange as ArrayLike<number>)[1]]
        : [1, 1024];

    const floatTextures = !!(
      gl.getExtension('EXT_color_buffer_float') ||
      gl.getExtension('EXT_color_buffer_half_float') ||
      gl.getExtension('WEBGL_color_buffer_float')
    );
    const colorDepth = {
      red: (gl.getParameter(gl.RED_BITS) as number) ?? 8,
      green: (gl.getParameter(gl.GREEN_BITS) as number) ?? 8,
      blue: (gl.getParameter(gl.BLUE_BITS) as number) ?? 8,
    };
    const hdr: HDRCapabilities = { ...display, floatTextures, colorDepth };

    return {
      api: 'webgl2',
      hdr,
      maxMSAASamples,
      pointSizeRange,
      readBackbufferPixels() {
        // Bind the canvas backbuffer explicitly. `runPipeline` is
        // defensive about restoring its prior render target, but we
        // can't assume the caller arrived here through that path.
        renderer.setRenderTarget(null);
        const ctx = renderer.getContext();
        const width = ctx.drawingBufferWidth;
        const height = ctx.drawingBufferHeight;
        const pixels = new Uint8Array(width * height * 4);
        ctx.readPixels(0, 0, width, height, ctx.RGBA, ctx.UNSIGNED_BYTE, pixels);
        return Promise.resolve({ pixels, width, height });
      },
    };
  }

  // WebGPU path: WebGPURenderer doesn't expose a `getContext()`
  // returning a WebGL2 context. Most capability probes that the
  // WebGL2 path uses (MAX_SAMPLES, ALIASED_POINT_SIZE_RANGE, GL
  // extensions, channel bit depths) don't have direct WebGPU
  // equivalents — WebGPU's adapter limits cover different things.
  // For now, use sensible defaults; M17 fills in the real
  // backbuffer-readback body and may expose more capabilities
  // when needed.
  const hdr: HDRCapabilities = {
    ...display,
    floatTextures: true, // WebGPU canvas formats include float-texture targets
    colorDepth: { red: 8, green: 8, blue: 8 },
  };

  return {
    api: 'webgpu',
    hdr,
    maxMSAASamples: 4, // WebGPU adapters guarantee at least 4× MSAA
    pointSizeRange: [1, 1024],
    readBackbufferPixels() {
      // M17 will implement this against WebGPU's offscreen target +
      // readRenderTargetPixelsAsync. Until then, the WebGPU path
      // doesn't reach the screenshot/EXR capture code.
      return Promise.reject(
        new Error('readBackbufferPixels: WebGPU body not yet implemented (M17)')
      );
    },
  };
}
