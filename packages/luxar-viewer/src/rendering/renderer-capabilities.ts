/**
 * Renderer capabilities — the single seam between Luxar and the raw
 * graphics API.
 *
 * Every other module in the codebase asks `RendererCapabilities` for
 * what the GPU can do, what bit depth the backbuffer has, or to read
 * pixels back. The implementation here speaks WebGL2. When the WebGPU
 * port lands, only this file changes.
 *
 * See `delme/` / plan files for the WebGPU migration prep work.
 */
import * as THREE from 'three';

import { detectHDRCapabilities, type HDRCapabilities } from '../utils/hdr-detection';

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
   * Implementations are responsible for binding the canvas backbuffer
   * before reading. Today this is sync (`gl.readPixels`); under WebGPU
   * the implementation will become async (`buffer.mapAsync`) — see the
   * Promise-ification work in Item 3 of the migration plan.
   */
  readBackbufferPixels(): { pixels: Uint8Array; width: number; height: number };
}

/**
 * Build a `RendererCapabilities` snapshot from a constructed
 * `THREE.WebGLRenderer`. Call once after renderer init; pass the result
 * to consumers (PostProcessingManager, SceneManager, …).
 */
export function createRendererCapabilities(
  renderer: THREE.WebGLRenderer
): RendererCapabilities {
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

  const hdr = detectHDRCapabilities(renderer);

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
      return { pixels, width, height };
    },
  };
}
