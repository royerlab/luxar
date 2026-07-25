/**
 * Native-WebGPU partial texture uploads (depth-sorting spec §7 Stage 3).
 *
 * The wrapper consumes the element textures' per-row `updateRanges`
 * into per-range `queue.writeTexture` calls and CLEARS them (nothing
 * else does on this backend — without the clear the fold's union grows
 * monotonically and partial degrades back to full). These tests drive
 * the wrapper with a recorded mock backend against REAL DataTextures
 * whose ranges come from the REAL `registerElementTexelDirtyRange`.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import * as THREE from 'three';
import {
  installWebGPUPartialTextureUploads,
  isRangedUploadEligible,
} from '../../../rendering/webgpu-partial-texture-upload';
import {
  configureElementTextureFullUploadKnee,
  markElementTextureFullDirty,
  registerElementTexelDirtyRange,
} from '../../../rendering/element-storage';

afterEach(() => {
  configureElementTextureFullUploadKnee(0.75);
});

const WIDTH = 12; // texels per row → rowFloats = 48

function makeTexture(rows: number): THREE.DataTexture {
  const tex = new THREE.DataTexture(
    new Float32Array(WIDTH * rows * 4),
    WIDTH,
    rows,
    THREE.RGBAFormat,
    THREE.FloatType
  );
  tex.flipY = false;
  return tex;
}

interface WriteCall {
  origin: { x: number; y: number };
  offsetBytes: number;
  bytesPerRow: number;
  size: { width: number; height: number };
}

function makeBackend(opts: { knownTexture?: boolean } = {}) {
  const writes: WriteCall[] = [];
  const originalCalls: THREE.Texture[] = [];
  const gpuTexture = { label: 'mock-gpu-texture' };
  const backend = {
    isWebGLBackend: false,
    device: {
      queue: {
        writeTexture: (
          dst: { texture: unknown; origin: { x: number; y: number } },
          _data: ArrayBufferView,
          layout: { offset: number; bytesPerRow: number },
          size: { width: number; height: number }
        ) => {
          expect(dst.texture).toBe(gpuTexture);
          writes.push({
            origin: { ...dst.origin },
            offsetBytes: layout.offset,
            bytesPerRow: layout.bytesPerRow,
            size: { ...size },
          });
        },
      },
    },
    get: (_t: THREE.Texture) => (opts.knownTexture === false ? {} : { texture: gpuTexture }),
    updateTexture: (t: THREE.Texture) => {
      originalCalls.push(t);
    },
  };
  return { backend, writes, originalCalls };
}

describe('installWebGPUPartialTextureUploads', () => {
  it('consumes per-row ranges into per-range writeTexture calls and CLEARS them', () => {
    const { backend, writes, originalCalls } = makeBackend();
    expect(installWebGPUPartialTextureUploads({ backend } as never)).toBe(true);

    const tex = makeTexture(8);
    // Real range registration: elements of 4 floats each (1 texel), the
    // span [6, 30) elements = floats [24, 120) = rows 0..2 with partial
    // head (x=6..11 of row 0) and partial tail (x=0..5 of row 2).
    registerElementTexelDirtyRange(tex, 4, 6, 30);
    expect(tex.updateRanges.length).toBeGreaterThan(0);

    backend.updateTexture(tex, undefined);

    // Head partial row: origin x=6 row 0, 6 texels; middle full row 1;
    // tail partial row 2 with 6 texels.
    expect(writes).toEqual([
      { origin: { x: 6, y: 0 }, offsetBytes: 24 * 4, bytesPerRow: WIDTH * 16, size: { width: 6, height: 1 } },
      { origin: { x: 0, y: 1 }, offsetBytes: 48 * 4, bytesPerRow: WIDTH * 16, size: { width: 12, height: 1 } },
      { origin: { x: 0, y: 2 }, offsetBytes: 96 * 4, bytesPerRow: WIDTH * 16, size: { width: 6, height: 1 } },
    ]);
    // Consume-and-clear — the monotonic-union trap's fix.
    expect(tex.updateRanges.length).toBe(0);
    expect(originalCalls.length).toBe(0); // the stock path never ran
  });

  it('append suffix uploads ONLY the suffix rows (the Stage-2 fast path on WebGPU)', () => {
    const { backend, writes } = makeBackend();
    installWebGPUPartialTextureUploads({ backend } as never);
    const tex = makeTexture(8);
    registerElementTexelDirtyRange(tex, 4, 24, 36); // rows 2..2 (elements 24..36 = floats 96..144 = rows 2..2)
    backend.updateTexture(tex, undefined);
    expect(writes.every((w) => w.origin.y === 2)).toBe(true);
    const texels = writes.reduce((s, w) => s + w.size.width, 0);
    expect(texels).toBe(12); // exactly the 12-texel suffix, nothing else
  });

  it('delegates the pendingFullUpload full-dirty encoding (EMPTY ranges) to the stock path', () => {
    const { backend, writes, originalCalls } = makeBackend();
    installWebGPUPartialTextureUploads({ backend } as never);
    const tex = makeTexture(8);
    markElementTextureFullDirty(tex); // needsUpdate + empty ranges
    backend.updateTexture(tex, undefined);
    expect(writes.length).toBe(0);
    expect(originalCalls).toEqual([tex]);
  });

  it('delegates non-DataTexture and ranged-ineligible textures untouched', () => {
    const { backend, originalCalls } = makeBackend();
    installWebGPUPartialTextureUploads({ backend } as never);
    const plain = new THREE.Texture();
    backend.updateTexture(plain, undefined);
    const noRanges = makeTexture(4);
    backend.updateTexture(noRanges, undefined);
    const flipped = makeTexture(4);
    flipped.flipY = true;
    registerElementTexelDirtyRange(flipped, 4, 0, 2);
    backend.updateTexture(flipped, undefined);
    expect(originalCalls).toEqual([plain, noRanges, flipped]);
  });

  it('falls back to the stock path on an unknown GPU texture or a malformed straddling range', () => {
    const unknown = makeBackend({ knownTexture: false });
    installWebGPUPartialTextureUploads({ backend: unknown.backend } as never);
    const tex = makeTexture(4);
    registerElementTexelDirtyRange(tex, 4, 0, 2);
    unknown.backend.updateTexture(tex, undefined);
    expect(unknown.originalCalls).toEqual([tex]);

    const { backend, writes, originalCalls } = makeBackend();
    installWebGPUPartialTextureUploads({ backend } as never);
    const tex2 = makeTexture(4);
    // A foreign producer's straddling range (crosses the row boundary) —
    // wrong bytes if uploaded height-1; must fall back to full.
    tex2.addUpdateRange(40, 16); // floats 40..56 straddle rows 0/1
    tex2.needsUpdate = true;
    backend.updateTexture(tex2, undefined);
    expect(writes.length).toBe(0);
    expect(originalCalls).toEqual([tex2]);
  });

  it('refuses installation (returns false) when the seam shape drifted', () => {
    const spy = vi.fn();
    expect(
      installWebGPUPartialTextureUploads({ backend: { updateTexture: spy } } as never)
    ).toBe(false);
    expect(
      installWebGPUPartialTextureUploads({ backend: undefined } as never)
    ).toBe(false);
  });
});

describe('configureElementTextureFullUploadKnee', () => {
  it('Infinity disables the ranged→full collapse (the WebGPU setting)', () => {
    configureElementTextureFullUploadKnee(Number.POSITIVE_INFINITY);
    const tex = makeTexture(8);
    // Dirty 100% of rows — under the 0.75 default this collapses to a
    // pending full upload (empty ranges); at Infinity it must stay ranged.
    registerElementTexelDirtyRange(tex, 4, 0, (WIDTH * 8) / 1); // all elements (element = 1 texel here → floatsPerElement 4)
    expect(tex.updateRanges.length).toBe(8); // one range per row
  });

  it('the default 0.75 knee still collapses to a full upload on classic WebGL', () => {
    configureElementTextureFullUploadKnee(0.75);
    const tex = makeTexture(8);
    registerElementTexelDirtyRange(tex, 4, 0, WIDTH * 8);
    expect(tex.updateRanges.length).toBe(0); // pending full (empty ranges + needsUpdate)
    expect(tex.needsUpdate || tex.version > 0).toBe(true);
  });

  it('eligibility helper matches the element-texture shape only', () => {
    const tex = makeTexture(2);
    expect(isRangedUploadEligible(tex)).toBe(false); // no ranges yet
    registerElementTexelDirtyRange(tex, 4, 0, 2);
    expect(isRangedUploadEligible(tex)).toBe(true);
  });
});
