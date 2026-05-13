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
// Type-only re-export of `WebGPURenderer` doubles as a smoke
// check that the `three/webgpu` subpath resolves cleanly under
// the `~0.184.0` pin (requires `tsconfig.moduleResolution:
// "Bundler"`). M2 lands the actual `Renderer` union arm + runtime
// branch in `scene-manager.ts:setupRenderer`. Zero runtime cost
// today — the import is erased at build time.
export type { WebGPURenderer } from 'three/webgpu';

import { detectDisplayCapabilities, type HDRCapabilities } from '../utils/hdr-detection';

/**
 * The graphics-API renderer Luxar uses. Single-arm today; the WebGPU
 * port widens this to `THREE.WebGLRenderer | THREE.WebGPURenderer`.
 * Consumers that hold a renderer reference should type it as
 * `Renderer` rather than `THREE.WebGLRenderer` so the port is a
 * one-edit widening.
 */
export type Renderer = THREE.WebGLRenderer;

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

  // HDR: merge display-side detection (CSS media queries) with
  // renderer-side probes (float-texture extension, color buffer bit
  // depth). The renderer-side probes are the only raw-GL probes
  // outside `readBackbufferPixels`; concentrating them here is the
  // whole point of this module.
  const display = detectDisplayCapabilities();
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
