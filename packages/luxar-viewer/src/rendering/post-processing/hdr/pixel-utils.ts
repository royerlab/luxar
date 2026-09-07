/**
 * Pure helpers for HDR pixel buffer manipulation, plus the unified
 * render-target readback primitive `readPixelsCompactAsync`.
 *
 * The pure helpers (half-float ↔ float32, row flip, WebGPU row-padding
 * compaction) are kept exported because tests cover them directly and
 * because `readPixelsCompactAsync` composes them. New call sites should
 * prefer `readPixelsCompactAsync`, which handles backend dispatch,
 * WebGPU padding, and canonical Y orientation in one place.
 *
 * @module rendering/post-processing/hdr/pixel-utils
 */

import * as THREE from 'three';

import type { Renderer, RendererCapabilities } from '../../renderer-capabilities';

/**
 * Convert an array of half-float values (encoded as Uint16) into
 * Float32 values via {@link THREE.DataUtils.fromHalfFloat}.
 *
 * Used after `gl.readPixels` against a HalfFloat render target — WebGL
 * gives us the raw 16-bit-encoded values; the rest of the HDR pipeline
 * (EXR encoding, downstream math) wants Float32.
 *
 * @param halfData - Source half-float-encoded buffer (Uint16Array view).
 * @returns Float32Array of the same length with decoded values.
 */
export function halfFloatToFloat32(halfData: Uint16Array): Float32Array {
  const out = new Float32Array(halfData.length);
  for (let i = 0; i < halfData.length; i++) {
    out[i] = THREE.DataUtils.fromHalfFloat(halfData[i]);
  }
  return out;
}

/**
 * Convert Float32 values into half-float encoding via
 * {@link THREE.DataUtils.toHalfFloat}.
 *
 * Used when exporting HDR data as a half-float EXR (smaller file size
 * than full Float32 with negligible quality loss for typical scenes).
 *
 * @param floatData - Source Float32Array.
 * @returns Uint16Array of the same length with half-float-encoded values.
 */
export function float32ToHalfFloat(floatData: Float32Array): Uint16Array {
  const out = new Uint16Array(floatData.length);
  for (let i = 0; i < floatData.length; i++) {
    out[i] = THREE.DataUtils.toHalfFloat(floatData[i]);
  }
  return out;
}

/**
 * Flip a row-major RGBA pixel buffer vertically.
 *
 * WebGL's `gl.readPixels` returns rows in bottom-up order (y=0 at the
 * bottom of the framebuffer); a `Canvas2D ImageData` expects top-down
 * (y=0 at the top). This routine swaps rows in-place by row-block
 * copy. It does not change RGBA component order — only the row order.
 *
 * The output is `Uint8ClampedArray` because that's what `ImageData`
 * requires; clamping is a no-op here since the input is already
 * Uint8 (no values outside [0, 255]).
 *
 * @param pixels - Source row-major RGBA pixel buffer (4 bytes/pixel).
 * @param width - Image width in pixels.
 * @param height - Image height in pixels.
 * @returns Vertically-flipped buffer as Uint8ClampedArray.
 */
export function flipPixelsVerticallyRGBA(
  pixels: Uint8Array,
  width: number,
  height: number
): Uint8ClampedArray {
  const rowSize = width * 4;
  const out = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    const srcOffset = y * rowSize;
    const dstOffset = (height - 1 - y) * rowSize;
    out.set(pixels.subarray(srcOffset, srcOffset + rowSize), dstOffset);
  }
  return out;
}

const WEBGPU_BYTES_PER_ROW_ALIGNMENT = 256;

/**
 * Drop WebGPU's per-row buffer padding from a readback typed array.
 *
 * WebGPU's `copyTextureToBuffer` (used internally by
 * `WebGPURenderer.readRenderTargetPixelsAsync`) requires `bytesPerRow`
 * to be a multiple of 256 (WebGPU spec § "Texture & buffer copy
 * alignment"). Three.js sizes the returned typed array to match the
 * padded GPU layout, so when `width * bytesPerTexel` isn't already a
 * multiple of 256 the array contains junk bytes at the end of every
 * row.
 *
 * This helper returns a row-compacted view sized exactly
 * `width * height * bytesPerTexel`. When the input already matches the
 * compact size (no padding required), the input is returned unchanged
 * to avoid an allocation. Both WebGL2-returned arrays (always compact)
 * and WebGPU-returned arrays for aligned widths hit the no-op path.
 *
 * Same math as the `PickingSystem.readbackAndVote` deinterlace loop;
 * extracted so the post-processing capture/screenshot paths can share
 * it (previously they assumed compact rows under WebGPU and produced
 * corrupted output for canvas widths whose row stride wasn't a multiple
 * of 256 bytes).
 *
 * @param raw - The typed array returned by
 *   `WebGPURenderer.readRenderTargetPixelsAsync`.
 * @param width - Image width in pixels.
 * @param height - Image height in pixels.
 * @param bytesPerTexel - 4 for RGBA8/Uint8, 8 for RGBA16F/Uint16, 16
 *   for RGBA32F/Float32.
 * @returns Either `raw` itself (when no compaction is needed) or a new
 *   typed array of the same concrete type sized for the compact layout.
 */
export function compactWebGPUReadbackRows<T extends Uint8Array | Uint16Array | Float32Array>(
  raw: T,
  width: number,
  height: number,
  bytesPerTexel: number
): T {
  const bytesPerRowReal = width * bytesPerTexel;
  const bytesPerRowPadded =
    Math.ceil(bytesPerRowReal / WEBGPU_BYTES_PER_ROW_ALIGNMENT) * WEBGPU_BYTES_PER_ROW_ALIGNMENT;
  if (bytesPerRowPadded === bytesPerRowReal) {
    return raw;
  }
  const bytesPerElement = raw.BYTES_PER_ELEMENT;
  const elementsPerRowPadded = bytesPerRowPadded / bytesPerElement;
  const elementsPerRowReal = bytesPerRowReal / bytesPerElement;
  const compactElementCount = elementsPerRowReal * height;
  if (raw.length === compactElementCount) {
    return raw;
  }
  // Allocate the same concrete typed-array kind as the input. The
  // generic constraint guarantees the constructor lookup is safe.
  const Ctor = raw.constructor as new (length: number) => T;
  const out = new Ctor(compactElementCount);
  for (let row = 0; row < height; row++) {
    const srcStart = row * elementsPerRowPadded;
    const dstStart = row * elementsPerRowReal;
    out.set(raw.subarray(srcStart, srcStart + elementsPerRowReal), dstStart);
  }
  return out;
}

// =============================================================================
// readPixelsCompactAsync — unified readback primitive
// =============================================================================

/**
 * Flip an RGBA typed-array vertically by row swap. Generic over the
 * three typed-array kinds the readback path produces. Same algorithm
 * as {@link flipPixelsVerticallyRGBA} but preserves the input's
 * concrete typed-array kind (the Uint8 variant returns
 * Uint8ClampedArray for ImageData compatibility; this one returns the
 * same kind as the input so it composes naturally with the readback).
 */
/**
 * Flip rows of an RGBA typed array. When `out` is provided it MUST be
 * a different buffer than `pixels` — the algorithm interleaves reads
 * and writes across rows (row 0 → row N-1, etc.), so passing the same
 * buffer would self-corrupt. Callers wanting an in-place flip should
 * stage through a scratch buffer.
 */
function flipRowsTyped<T extends Uint8Array | Uint16Array | Float32Array>(
  pixels: T,
  width: number,
  height: number,
  out?: T
): T {
  const elementsPerRow = width * 4;
  const Ctor = pixels.constructor as new (length: number) => T;
  const dst = out ?? new Ctor(elementsPerRow * height);
  for (let y = 0; y < height; y++) {
    const srcOffset = y * elementsPerRow;
    const dstOffset = (height - 1 - y) * elementsPerRow;
    dst.set(pixels.subarray(srcOffset, srcOffset + elementsPerRow), dstOffset);
  }
  return dst;
}

/** Render-target texel formats the unified readback supports. */
export type TexelKind = 'rgba8' | 'rgba16f' | 'rgba32f';

const BYTES_PER_TEXEL: Record<TexelKind, number> = {
  rgba8: 4,
  rgba16f: 8,
  rgba32f: 16,
};

type TexelArray<K extends TexelKind> = K extends 'rgba8'
  ? Uint8Array
  : K extends 'rgba16f'
    ? Uint16Array
    : K extends 'rgba32f'
      ? Float32Array
      : never;

const TEXEL_CTOR: { [K in TexelKind]: new (length: number) => TexelArray<K> } = {
  rgba8: Uint8Array,
  rgba16f: Uint16Array,
  rgba32f: Float32Array,
} as const;

/** Arguments to {@link readPixelsCompactAsync}. */
export interface ReadPixelsOpts<K extends TexelKind = TexelKind> {
  /**
   * Source render target. Typed as the base `RenderTarget` so a cube target from
   * either backend (`WebGLCubeRenderTarget`, or `three/webgpu`'s `CubeRenderTarget`,
   * which is NOT a `WebGLRenderTarget`) is accepted; see `faceIndex`.
   */
  target: THREE.RenderTarget;
  /** Pixel-format discriminator. Determines the returned typed-array kind. */
  kind: K;
  /**
   * Cube face to read when `target` is a cube render target (0..5 in three's
   * px, nx, py, ny, pz, nz order). Threaded into WebGL's `activeCubeFaceIndex`
   * argument and WebGPU's `faceIndex` argument — the two signatures put it in
   * different slots (after the destination buffer vs after `textureIndex`).
   * Ignored for a 2D target.
   */
  faceIndex?: number;
  /**
   * X offset in pixels (top-down convention). Defaults to 0. Full-target
   * reads (the common case) leave this at 0; sub-region readers (e.g. the
   * picking 5×5 voter) supply canvas-space top-down coords and the
   * primitive converts to the backend's framebuffer convention internally.
   */
  x?: number;
  /**
   * Y offset in pixels (canonical **top-down**, row 0 = top of source).
   * The primitive flips this to the bottom-up framebuffer convention
   * internally when `caps.framebufferYDown === false`. Defaults to 0.
   */
  y?: number;
  /** Region width in pixels. Defaults to `target.width`. */
  width?: number;
  /** Region height in pixels. Defaults to `target.height`. */
  height?: number;
  /**
   * Default `false`. The primitive's canonical return convention is
   * rows in **top-down** order (row 0 = top of the source target).
   * Pass `true` to receive bottom-up rows instead — used by callers
   * whose downstream consumers expect scene-space row order (e.g. EXR
   * export).
   */
  flipY?: boolean;
  /**
   * Optional pre-allocated destination buffer for the raw readback. Must
   * be exactly `width * height * 4` elements of the right typed-array
   * kind for `kind`. When provided, no fresh allocation is made for the
   * raw readback; ideal for hot paths (e.g. picking) that fire at
   * mouse-event rates.
   */
  out?: TexelArray<K>;
  /**
   * Optional pre-allocated destination buffer for the row-flipped
   * output (only consulted when `flipY` is the default `false`). Must
   * be the same size and kind as `out`, and a different buffer than
   * `out` (the row-flip interleaves reads/writes across rows and
   * cannot operate in place). When provided, the row-flip writes into
   * this buffer instead of allocating one. The returned `pixels` is
   * this buffer (not the raw `out`).
   */
  flipOut?: TexelArray<K>;
}

/** Return value of {@link readPixelsCompactAsync}. */
export interface ReadPixelsResult<K extends TexelKind = TexelKind> {
  pixels: TexelArray<K>;
  width: number;
  height: number;
}

type WebGPUReadback = {
  readRenderTargetPixelsAsync(
    target: THREE.RenderTarget,
    x: number,
    y: number,
    width: number,
    height: number,
    textureIndex?: number,
    faceIndex?: number
  ): Promise<Uint8Array | Uint16Array | Float32Array>;
};

/**
 * Unified pixel readback. Hides three backend differences from callers:
 *
 * 1. **Method-signature dispatch** — WebGL2's `readRenderTargetPixelsAsync`
 *    takes a destination buffer; WebGPU's returns the result.
 * 2. **WebGPU row padding** — WebGPU's `copyTextureToBuffer` rounds
 *    `bytesPerRow` up to 256; this primitive runs
 *    {@link compactWebGPUReadbackRows} unconditionally (no-op for
 *    aligned widths or under WebGL2).
 * 3. **Framebuffer Y orientation** — both renderer surfaces return
 *    **bottom-up** rows from `readRenderTargetPixelsAsync`:
 *    WebGLRenderer is calling `gl.readPixels` underneath, and
 *    WebGPURenderer's compat layer maintains the same contract on
 *    both its real-WebGPU and WebGL2 backends. (`caps.framebufferYDown`
 *    describes the *sampling* convention used by
 *    {@link createFullscreenTriangleGeometry}, NOT the readback
 *    memory layout — empirically the two have diverged on
 *    WebGPURenderer.) The primitive canonicalises to **top-down**
 *    by default. Pass `flipY: true` to get bottom-up rows out
 *    instead — only the EXR exporter does this today, to preserve
 *    the orientation external tools (Nuke, Houdini, oiiotool)
 *    expect.
 */
export async function readPixelsCompactAsync<K extends TexelKind>(
  renderer: Renderer,
  caps: RendererCapabilities,
  opts: ReadPixelsOpts<K>
): Promise<ReadPixelsResult<K>> {
  const target = opts.target;
  const xTopDown = opts.x ?? 0;
  const yTopDown = opts.y ?? 0;
  const width = opts.width ?? target.width;
  const height = opts.height ?? target.height;
  const flipY = opts.flipY ?? false;
  const bytesPerTexel = BYTES_PER_TEXEL[opts.kind];
  const Ctor = TEXEL_CTOR[opts.kind];
  const compactLength = width * height * 4;

  // Both renderer surfaces use `gl.readPixels`-style bottom-up
  // addressing for the input `(x, y)` argument. WebGPURenderer's
  // compat layer maintains the WebGL convention on both backends —
  // verified by the y-orientation E2E spec.
  const x = xTopDown;
  const y = target.height - yTopDown - height;

  let pixels: TexelArray<K>;
  if (caps.apiSurface === 'webgl2') {
    // WebGL2 signature: pass destination, fill in-place. The returned
    // buffer is always compact (no row-padding under WebGL).
    pixels = (opts.out ?? new Ctor(compactLength)) as TexelArray<K>;
    await (renderer as THREE.WebGLRenderer).readRenderTargetPixelsAsync(
      target as THREE.WebGLRenderTarget,
      x,
      y,
      width,
      height,
      pixels,
      opts.faceIndex
    );
  } else {
    // WebGPU signature: returns a typed array of the right kind, but
    // possibly padded out to 256-byte rows. compactWebGPUReadbackRows
    // is a no-op when the row stride is already aligned. The
    // WebGPU surface doesn't accept a destination buffer, so `opts.out`
    // is honoured by copying the result into it after compaction.
    const raw = (await (renderer as unknown as WebGPUReadback).readRenderTargetPixelsAsync(
      target,
      x,
      y,
      width,
      height,
      0,
      // `undefined` lets the renderer's own default (face 0) apply.
      opts.faceIndex
    )) as TexelArray<K>;
    const compact = compactWebGPUReadbackRows(raw, width, height, bytesPerTexel) as TexelArray<K>;
    if (opts.out) {
      opts.out.set(compact as unknown as ArrayLike<number>);
      pixels = opts.out;
    } else {
      pixels = compact;
    }
  }

  // Raw readback is bottom-up on both backends. Canonical out-orientation
  // is top-down (`wantsTopDown = !flipY`), so flip iff the caller wants
  // top-down output. Bottom-up output (`flipY: true`) passes through.
  if (!flipY) {
    pixels = flipRowsTyped(pixels, width, height, opts.flipOut);
  }

  return { pixels, width, height };
}
