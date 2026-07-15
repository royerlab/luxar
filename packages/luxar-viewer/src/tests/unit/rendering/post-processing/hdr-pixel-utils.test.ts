/**
 * Unit tests for the post-processing HDR pixel utilities.
 *
 * Pure conversions over typed arrays — no GL context needed; we feed
 * known values and check round-trips and orientation.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  halfFloatToFloat32,
  float32ToHalfFloat,
  flipPixelsVerticallyRGBA,
  compactWebGPUReadbackRows,
  readPixelsCompactAsync,
} from '../../../../rendering/post-processing/hdr/pixel-utils';
import type { Renderer, RendererCapabilities } from '../../../../rendering/renderer-capabilities';

describe('halfFloatToFloat32', () => {
  it('returns an empty Float32Array for empty input', () => {
    const out = halfFloatToFloat32(new Uint16Array(0));
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(0);
  });

  it('preserves the array length', () => {
    expect(halfFloatToFloat32(new Uint16Array(5)).length).toBe(5);
  });

  it('decodes half-float bit patterns to expected float values', () => {
    // Encode a few canonical floats via THREE, then decode and compare.
    const source = new Float32Array([0, 1, -1, 0.5, -0.25, 65504]);
    const encoded = new Uint16Array(source.length);
    for (let i = 0; i < source.length; i++) {
      encoded[i] = THREE.DataUtils.toHalfFloat(source[i]);
    }
    const decoded = halfFloatToFloat32(encoded);
    for (let i = 0; i < source.length; i++) {
      // Half-float has limited precision; allow small rounding error.
      expect(decoded[i]).toBeCloseTo(source[i], 1);
    }
  });
});

describe('float32ToHalfFloat', () => {
  it('returns an empty Uint16Array for empty input', () => {
    const out = float32ToHalfFloat(new Float32Array(0));
    expect(out).toBeInstanceOf(Uint16Array);
    expect(out.length).toBe(0);
  });

  it('preserves the array length', () => {
    expect(float32ToHalfFloat(new Float32Array(7)).length).toBe(7);
  });

  it('round-trips losslessly for half-float-representable values', () => {
    const source = new Float32Array([0, 1, -1, 0.5, -0.25, 16, -2048]);
    const encoded = float32ToHalfFloat(source);
    const decoded = halfFloatToFloat32(encoded);
    for (let i = 0; i < source.length; i++) {
      // Each chosen value is exactly representable in half-float.
      expect(decoded[i]).toBe(source[i]);
    }
  });
});

describe('flipPixelsVerticallyRGBA', () => {
  it('returns a Uint8ClampedArray of the same total size', () => {
    const pixels = new Uint8Array(2 * 3 * 4); // 2x3 RGBA
    const out = flipPixelsVerticallyRGBA(pixels, 2, 3);
    expect(out).toBeInstanceOf(Uint8ClampedArray);
    expect(out.length).toBe(2 * 3 * 4);
  });

  it('flips a 1×2 image (just two rows swap)', () => {
    // 1x2 RGBA: rows = [10,20,30,40] (y=0, bottom) and [50,60,70,80] (y=1, top).
    // After flip, y=0 should hold the top row and y=1 should hold the bottom row.
    const pixels = new Uint8Array([10, 20, 30, 40, 50, 60, 70, 80]);
    const out = flipPixelsVerticallyRGBA(pixels, 1, 2);
    expect(Array.from(out)).toEqual([50, 60, 70, 80, 10, 20, 30, 40]);
  });

  it('flips a 2×3 image — rows reorder, RGBA byte order within each row preserved', () => {
    // 2 wide × 3 tall, 4 components per pixel. Rows are 8 bytes each.
    // y=0 row: [1..8], y=1 row: [9..16], y=2 row: [17..24].
    const pixels = new Uint8Array([
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8, // y=0
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16, // y=1
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24, // y=2
    ]);
    const out = flipPixelsVerticallyRGBA(pixels, 2, 3);
    expect(Array.from(out)).toEqual([
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24, // y=0 ← was y=2
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16, // y=1 unchanged
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8, // y=2 ← was y=0
    ]);
  });

  it('returns an all-zero buffer for zero-area inputs without throwing', () => {
    expect(flipPixelsVerticallyRGBA(new Uint8Array(0), 0, 0).length).toBe(0);
    expect(flipPixelsVerticallyRGBA(new Uint8Array(4), 1, 0).length).toBe(0);
  });
});

describe('compactWebGPUReadbackRows', () => {
  // WebGPU pads `bytesPerRow` to a multiple of 256. The helper drops
  // that padding so callers see a compact row layout regardless of the
  // backend. Widths whose `width * bytesPerTexel` is already a multiple
  // of 256 should take the no-op fast path (return the input array
  // unchanged).

  it('returns the input unchanged when width × bytesPerTexel is already 256-aligned (RGBA8 width=64)', () => {
    // 64 × 4 = 256 → no padding required.
    const raw = new Uint8Array(64 * 1 * 4);
    raw[0] = 7;
    raw[raw.length - 1] = 9;
    const out = compactWebGPUReadbackRows(raw, 64, 1, 4);
    expect(out).toBe(raw);
  });

  it('compacts a 5×5 RGBA32F readback (80 B/row → 256 B padded)', () => {
    // Mirrors picking-system.ts: PICK_SIZE=5, bytesPerTexel=16.
    const elementsPerRowReal = 5 * 4; // 20 floats
    const elementsPerRowPadded = 256 / 4; // 64 floats
    const raw = new Float32Array(elementsPerRowPadded * 5);
    // Fill the compact region with row-index codes so we can verify
    // ordering after compaction. Padding columns get a sentinel.
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < elementsPerRowPadded; col++) {
        raw[row * elementsPerRowPadded + col] = col < elementsPerRowReal ? row * 100 + col : -999;
      }
    }
    const out = compactWebGPUReadbackRows(raw, 5, 5, 16);
    expect(out).toBeInstanceOf(Float32Array);
    expect(out).not.toBe(raw);
    expect(out.length).toBe(elementsPerRowReal * 5);
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < elementsPerRowReal; col++) {
        expect(out[row * elementsPerRowReal + col]).toBe(row * 100 + col);
      }
    }
    // The sentinel padding values must NOT leak into the compact output.
    for (let i = 0; i < out.length; i++) {
      expect(out[i]).not.toBe(-999);
    }
  });

  it('compacts a 853×2 RGBA8 readback (3412 B/row → 3584 B padded)', () => {
    // 853 × 4 = 3412; ceil(3412 / 256) * 256 = 3584; 172 bytes of pad.
    const width = 853;
    const height = 2;
    const elementsPerRowReal = width * 4;
    const elementsPerRowPadded = 3584;
    const raw = new Uint8Array(elementsPerRowPadded * height);
    // Encode (row << 24 | col) into the compact region; padding gets 0xFF.
    raw.fill(0xff);
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < elementsPerRowReal; col++) {
        raw[row * elementsPerRowPadded + col] = (row * 13 + col) & 0xff;
      }
    }
    const out = compactWebGPUReadbackRows(raw, width, height, 4);
    expect(out.length).toBe(elementsPerRowReal * height);
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < elementsPerRowReal; col++) {
        expect(out[row * elementsPerRowReal + col]).toBe((row * 13 + col) & 0xff);
      }
    }
  });

  it('compacts a 853×2 RGBA16F readback (6824 B/row → 6912 B padded)', () => {
    // 853 × 8 = 6824; ceil(6824 / 256) * 256 = 6912; 88 bytes of pad
    // = 44 Uint16 entries.
    const width = 853;
    const height = 2;
    const elementsPerRowReal = width * 4;
    const elementsPerRowPadded = 6912 / 2; // Uint16
    const raw = new Uint16Array(elementsPerRowPadded * height);
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < elementsPerRowPadded; col++) {
        raw[row * elementsPerRowPadded + col] = col < elementsPerRowReal ? row * 1000 + col : 0;
      }
    }
    const out = compactWebGPUReadbackRows(raw, width, height, 8);
    expect(out).toBeInstanceOf(Uint16Array);
    expect(out.length).toBe(elementsPerRowReal * height);
    expect(out[0]).toBe(0);
    expect(out[elementsPerRowReal - 1]).toBe(elementsPerRowReal - 1);
    expect(out[elementsPerRowReal]).toBe(1000); // first byte of row 1
  });

  it('compacts a 853×3 RGBA32F readback (13648 B/row → 13824 B padded)', () => {
    // 853 × 16 = 13648; ceil(13648 / 256) * 256 = 13824; 176 bytes of pad
    // = 44 Float32 entries.
    const width = 853;
    const height = 3;
    const elementsPerRowReal = width * 4;
    const elementsPerRowPadded = 13824 / 4;
    const raw = new Float32Array(elementsPerRowPadded * height);
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < elementsPerRowPadded; col++) {
        raw[row * elementsPerRowPadded + col] =
          col < elementsPerRowReal ? row + col * 0.001 : Number.NaN;
      }
    }
    const out = compactWebGPUReadbackRows(raw, width, height, 16);
    expect(out.length).toBe(elementsPerRowReal * height);
    // No NaN sentinels should have leaked from the padding band.
    for (let i = 0; i < out.length; i++) {
      expect(Number.isNaN(out[i])).toBe(false);
    }
    // Spot-check a few entries.
    expect(out[0]).toBeCloseTo(0, 4);
    expect(out[elementsPerRowReal - 1]).toBeCloseTo((elementsPerRowReal - 1) * 0.001, 4);
    expect(out[elementsPerRowReal]).toBeCloseTo(1, 4); // first entry of row 1
  });

  it('returns input unchanged when length already matches compact (WebGL2 caller path)', () => {
    // The WebGL2 backend returns a compact array even for unaligned
    // widths. The helper must detect this and short-circuit.
    const width = 5;
    const height = 5;
    const raw = new Float32Array(width * height * 4); // compact RGBA32F
    const out = compactWebGPUReadbackRows(raw, width, height, 16);
    expect(out).toBe(raw);
  });
});

// =============================================================================
// readPixelsCompactAsync
// =============================================================================
//
// Hand-rolled fakes for renderer + caps. The primitive is pure-logic
// glue around two `readRenderTargetPixelsAsync` signatures plus the
// existing flip/compact helpers, so we exercise the dispatch matrix
// without needing a real GL/GPU context.

function makeCaps(
  apiSurface: 'webgl2' | 'webgpu',
  framebufferYDown: boolean
): RendererCapabilities {
  return {
    apiSurface,
    framebufferYDown,
    hdr: {
      p3Gamut: false,
      rec2020Gamut: false,
      hdr: false,
      deepColor: false,
      floatTextures: true,
      colorDepth: { red: 8, green: 8, blue: 8 },
      recommendedColorSpace: 'srgb',
    },
    maxTextureSize: 4096,
    maxMSAASamples: 4,
    pointSizeRange: [1, 1024],
    readBackbufferPixels: () => Promise.resolve({ pixels: new Uint8Array(), width: 0, height: 0 }),
  };
}

function makeTarget(width: number, height: number): THREE.WebGLRenderTarget {
  return { width, height } as unknown as THREE.WebGLRenderTarget;
}

/**
 * Build a 1-wide × `height`-tall RGBA8 buffer with row `i` filled by
 * `(i, 0, 0, 255)`. With row 0 = (0,0,0,255), row 1 = (1,0,0,255), etc.
 * The R channel is a row-index probe — flipping rows is detectable by
 * inspecting the R channel order in the output.
 */
function makeRowProbeBuffer(height: number): Uint8Array {
  const buf = new Uint8Array(height * 4);
  for (let row = 0; row < height; row++) {
    buf[row * 4 + 0] = row;
    buf[row * 4 + 3] = 255;
  }
  return buf;
}

function extractRowIndices(buf: Uint8Array): number[] {
  const rows = buf.length / 4;
  const out: number[] = [];
  for (let i = 0; i < rows; i++) out.push(buf[i * 4]);
  return out;
}

describe('readPixelsCompactAsync', () => {
  describe('WebGL2 (bottom-up framebuffer)', () => {
    // Under WebGL2, `readRenderTargetPixelsAsync` writes the raw
    // readback into the destination buffer in bottom-up order: the
    // first row of the buffer is the BOTTOM of the source target.
    // Building a row-probe with R = row-index in *output* terms means
    // we feed the mock a bottom-up pattern.

    it('returns top-down rows by default (flips the bottom-up GL readback)', async () => {
      const height = 4;
      const target = makeTarget(1, height);
      const caps = makeCaps('webgl2', false);
      // WebGL semantics: row 0 of the buffer is the bottom of the
      // image. To produce a result where R encodes top-down row index,
      // the renderer must write the *reverse* into the buffer.
      const bottomUp = new Uint8Array([3, 0, 0, 255, 2, 0, 0, 255, 1, 0, 0, 255, 0, 0, 0, 255]);
      const renderer = {
        readRenderTargetPixelsAsync: vi
          .fn()
          .mockImplementation(
            (
              _t: THREE.WebGLRenderTarget,
              _x: number,
              _y: number,
              _w: number,
              _h: number,
              dst: Uint8Array
            ) => {
              dst.set(bottomUp);
              return Promise.resolve();
            }
          ),
      } as unknown as Renderer;

      const result = await readPixelsCompactAsync(renderer, caps, {
        target,
        kind: 'rgba8',
      });

      // Top-down rows: row 0 of the output is the TOP of the source,
      // which is what the test probe encodes as R=0.
      expect(extractRowIndices(result.pixels)).toEqual([0, 1, 2, 3]);
      expect(result.width).toBe(1);
      expect(result.height).toBe(height);
    });

    it('returns bottom-up rows when flipY=true (skips the flip)', async () => {
      const height = 4;
      const target = makeTarget(1, height);
      const caps = makeCaps('webgl2', false);
      const bottomUp = new Uint8Array([3, 0, 0, 255, 2, 0, 0, 255, 1, 0, 0, 255, 0, 0, 0, 255]);
      const renderer = {
        readRenderTargetPixelsAsync: vi
          .fn()
          .mockImplementation(
            (
              _t: THREE.WebGLRenderTarget,
              _x: number,
              _y: number,
              _w: number,
              _h: number,
              dst: Uint8Array
            ) => {
              dst.set(bottomUp);
              return Promise.resolve();
            }
          ),
      } as unknown as Renderer;

      const result = await readPixelsCompactAsync(renderer, caps, {
        target,
        kind: 'rgba8',
        flipY: true,
      });

      // No flip applied: rows arrive in raw GL bottom-up order.
      expect(extractRowIndices(result.pixels)).toEqual([3, 2, 1, 0]);
    });

    it('allocates the destination buffer and forwards it to the WebGL2 signature', async () => {
      const target = makeTarget(2, 2);
      const caps = makeCaps('webgl2', false);
      const readPixels = vi.fn().mockResolvedValue(undefined);
      const renderer = { readRenderTargetPixelsAsync: readPixels } as unknown as Renderer;

      await readPixelsCompactAsync(renderer, caps, { target, kind: 'rgba8' });

      expect(readPixels).toHaveBeenCalledTimes(1);
      const args = readPixels.mock.calls[0];
      expect(args[0]).toBe(target);
      expect(args[1]).toBe(0); // x
      expect(args[2]).toBe(0); // y
      expect(args[3]).toBe(2); // width
      expect(args[4]).toBe(2); // height
      expect(args[5]).toBeInstanceOf(Uint8Array); // destination buffer
      expect((args[5] as Uint8Array).length).toBe(2 * 2 * 4);
    });

    it('returns Float32Array buffer for kind=rgba32f', async () => {
      const target = makeTarget(1, 1);
      const caps = makeCaps('webgl2', false);
      const renderer = {
        readRenderTargetPixelsAsync: vi.fn().mockResolvedValue(undefined),
      } as unknown as Renderer;

      const result = await readPixelsCompactAsync(renderer, caps, {
        target,
        kind: 'rgba32f',
      });

      expect(result.pixels).toBeInstanceOf(Float32Array);
      expect(result.pixels.length).toBe(4);
    });
  });

  describe('WebGPU (sampling: top-down; readback: bottom-up)', () => {
    // Under WebGPURenderer, `readRenderTargetPixelsAsync` returns
    // **bottom-up** rows on both its real-WebGPU and WebGL2 backends —
    // verified empirically by the y-orientation E2E spec. (Sampling
    // via TSL `texture(...).sample(uv)` uses top-down UVs, which
    // `caps.framebufferYDown` describes — but the readback memory
    // convention is independent of that and matches the GL contract.)
    // The primitive flips bottom-up readback to top-down by default.

    it('returns top-down rows by default (flips bottom-up readback like WebGL2)', async () => {
      const height = 4;
      const target = makeTarget(1, height);
      const caps = makeCaps('webgpu', true);
      const bottomUp = makeRowProbeBuffer(height); // R: [0, 1, 2, 3] = bottom-up
      const renderer = {
        readRenderTargetPixelsAsync: vi.fn().mockResolvedValue(bottomUp),
      } as unknown as Renderer;

      const result = await readPixelsCompactAsync(renderer, caps, {
        target,
        kind: 'rgba8',
      });

      // Bottom-up [0,1,2,3] → flipped to top-down [3,2,1,0].
      expect(extractRowIndices(result.pixels)).toEqual([3, 2, 1, 0]);
    });

    it('passes through unflipped when flipY=true (raw is already bottom-up)', async () => {
      const height = 4;
      const target = makeTarget(1, height);
      const caps = makeCaps('webgpu', true);
      const bottomUp = makeRowProbeBuffer(height);
      const renderer = {
        readRenderTargetPixelsAsync: vi.fn().mockResolvedValue(bottomUp),
      } as unknown as Renderer;

      const result = await readPixelsCompactAsync(renderer, caps, {
        target,
        kind: 'rgba8',
        flipY: true,
      });

      // flipY=true requests bottom-up output; raw is already bottom-up,
      // so it passes through unflipped.
      expect(extractRowIndices(result.pixels)).toEqual([0, 1, 2, 3]);
    });

    it('calls the WebGPU signature (no destination buffer arg)', async () => {
      const target = makeTarget(2, 2);
      const caps = makeCaps('webgpu', true);
      const readPixels = vi.fn().mockResolvedValue(new Uint8Array(16));
      const renderer = { readRenderTargetPixelsAsync: readPixels } as unknown as Renderer;

      await readPixelsCompactAsync(renderer, caps, { target, kind: 'rgba8' });

      expect(readPixels).toHaveBeenCalledTimes(1);
      const args = readPixels.mock.calls[0];
      // WebGPU signature: 5 args (target + x + y + w + h). No destination.
      expect(args.length).toBe(5);
    });
  });

  describe('mixed caps (api=webgpu, framebufferYDown=false)', () => {
    // Defensive: production no longer produces this combination
    // because Three.js's WebGPURenderer normalises Y on both its real
    // and compat backends (`detectFramebufferYDown` returns `true` for
    // any WebGPURenderer). The primitive must still handle the
    // combination correctly so a future Three.js change that breaks
    // the assumption surfaces as a localised regression rather than a
    // visual bug.

    it('flips to top-down by default', async () => {
      const height = 4;
      const target = makeTarget(1, height);
      const caps = makeCaps('webgpu', false);
      const bottomUp = new Uint8Array([3, 0, 0, 255, 2, 0, 0, 255, 1, 0, 0, 255, 0, 0, 0, 255]);
      const renderer = {
        readRenderTargetPixelsAsync: vi.fn().mockResolvedValue(bottomUp),
      } as unknown as Renderer;

      const result = await readPixelsCompactAsync(renderer, caps, {
        target,
        kind: 'rgba8',
      });

      expect(extractRowIndices(result.pixels)).toEqual([0, 1, 2, 3]);
    });
  });

  describe('top-down sub-region input coordinates', () => {
    // The primitive treats `(x, y)` input as canonical top-down. Under
    // WebGL2 (bottom-up framebuffer) it must translate to the
    // gl.readPixels bottom-up origin so a top-down sub-region read
    // hits the intended pixels. The picking 5×5 voter relies on this.

    it('translates top-down y to bottom-up framebuffer y for WebGL2 sub-region reads', async () => {
      // Target is 10 tall; request a 3-row read starting at top-down y=2.
      // Expected GL `y` (bottom-up) = targetHeight - yTopDown - height =
      // 10 - 2 - 3 = 5.
      const target = makeTarget(1, 10);
      const caps = makeCaps('webgl2', false);
      const readPixels = vi.fn().mockResolvedValue(undefined);
      const renderer = { readRenderTargetPixelsAsync: readPixels } as unknown as Renderer;

      await readPixelsCompactAsync(renderer, caps, {
        target,
        kind: 'rgba8',
        x: 0,
        y: 2,
        width: 1,
        height: 3,
      });

      const args = readPixels.mock.calls[0];
      expect(args[1]).toBe(0); // x — top-down x equals bottom-up x
      expect(args[2]).toBe(5); // y — flipped to bottom-up
      expect(args[3]).toBe(1); // width
      expect(args[4]).toBe(3); // height
    });

    it('translates top-down y to bottom-up on WebGPU readback (same as WebGL2)', async () => {
      // WebGPURenderer's readRenderTargetPixelsAsync uses the same
      // bottom-up addressing as gl.readPixels on both its real-WebGPU
      // and WebGL2 backends. caps.framebufferYDown=true describes the
      // shader-sampling convention, not the readback memory convention.
      const target = makeTarget(1, 10);
      const caps = makeCaps('webgpu', true);
      const readPixels = vi.fn().mockResolvedValue(new Uint8Array(12));
      const renderer = { readRenderTargetPixelsAsync: readPixels } as unknown as Renderer;

      await readPixelsCompactAsync(renderer, caps, {
        target,
        kind: 'rgba8',
        x: 0,
        y: 2,
        width: 1,
        height: 3,
      });

      const args = readPixels.mock.calls[0];
      // y = target.height - yTopDown - height = 10 - 2 - 3 = 5.
      expect(args[2]).toBe(5);
    });

    it('translates y to bottom-up when caps.framebufferYDown=false regardless of api', async () => {
      // Defensive: production no longer pairs `api='webgpu'` with
      // `framebufferYDown=false` (see the matching describe block
      // above), but the primitive must still honour the cap if it ever
      // appears.
      const target = makeTarget(1, 10);
      const caps = makeCaps('webgpu', false);
      const readPixels = vi.fn().mockResolvedValue(new Uint8Array(12));
      const renderer = { readRenderTargetPixelsAsync: readPixels } as unknown as Renderer;

      await readPixelsCompactAsync(renderer, caps, {
        target,
        kind: 'rgba8',
        x: 0,
        y: 2,
        width: 1,
        height: 3,
      });

      const args = readPixels.mock.calls[0];
      expect(args[2]).toBe(5);
    });

    it('keeps the previous behaviour for full-target reads (y=0, h=H)', async () => {
      // Default args: y=0, height=target.height → bottom-up y becomes
      // H - 0 - H = 0. Identical to the pre-translation behaviour.
      const target = makeTarget(1, 10);
      const caps = makeCaps('webgl2', false);
      const readPixels = vi.fn().mockResolvedValue(undefined);
      const renderer = { readRenderTargetPixelsAsync: readPixels } as unknown as Renderer;

      await readPixelsCompactAsync(renderer, caps, { target, kind: 'rgba8' });

      const args = readPixels.mock.calls[0];
      expect(args[2]).toBe(0);
      expect(args[4]).toBe(10);
    });
  });

  describe('WebGPU row-padding (compactWebGPUReadbackRows integration)', () => {
    it('compacts a padded WebGPU readback before flipping', async () => {
      // 1 row, 53-pixel-wide RGBA8 → 212-byte rows, padded to 256.
      const width = 53;
      const height = 1;
      const target = makeTarget(width, height);
      const caps = makeCaps('webgpu', true);

      // Build padded buffer: row data fills the first 212 bytes; the
      // remaining 44 bytes are padding (sentinel 0xff). After
      // compaction, the result should be exactly 212 bytes long with
      // none of the sentinel bytes leaking through.
      const padded = new Uint8Array(256);
      for (let i = 0; i < 212; i++) padded[i] = i % 256;
      for (let i = 212; i < 256; i++) padded[i] = 0xff; // sentinel
      const renderer = {
        readRenderTargetPixelsAsync: vi.fn().mockResolvedValue(padded),
      } as unknown as Renderer;

      const result = await readPixelsCompactAsync(renderer, caps, {
        target,
        kind: 'rgba8',
      });

      expect(result.pixels.length).toBe(width * height * 4);
      // No sentinel bytes — only the first 212 bytes of `padded`.
      for (let i = 0; i < 212; i++) expect(result.pixels[i]).toBe(i % 256);
    });
  });
});
