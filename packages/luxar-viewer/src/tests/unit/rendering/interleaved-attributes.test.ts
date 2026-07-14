import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  packInterleavedAttributes,
  writeInterleavedAttribute,
  widenToFloat32,
} from '../../../rendering/interleaved-attributes';

describe('packInterleavedAttributes', () => {
  it('produces a single InstancedInterleavedBuffer with correct stride', () => {
    const { buffer, stride, views, offsets } = packInterleavedAttributes(
      [
        { name: 'pos', data: new Float32Array([1, 2, 3, 4, 5, 6]), itemSize: 3 },
        { name: 'scalar', data: new Float32Array([7, 8]), itemSize: 1 },
      ],
      2
    );

    expect(buffer).toBeInstanceOf(THREE.InstancedInterleavedBuffer);
    expect(stride).toBe(4);
    expect(offsets).toEqual({ pos: 0, scalar: 3 });
    // Two instances × stride 4 = 8 floats.
    expect(buffer.array.length).toBe(8);
    // Interleaved layout: [pos_0.xyz, scalar_0, pos_1.xyz, scalar_1]
    expect(Array.from(buffer.array as Float32Array)).toEqual([1, 2, 3, 7, 4, 5, 6, 8]);
    expect(views.pos).toBeInstanceOf(THREE.InterleavedBufferAttribute);
    expect(views.pos.itemSize).toBe(3);
    expect(views.pos.offset).toBe(0);
    expect(views.scalar.itemSize).toBe(1);
    expect(views.scalar.offset).toBe(3);
  });

  it('lays out three vec3 attributes with stride 9', () => {
    const { stride, offsets, buffer } = packInterleavedAttributes(
      [
        { name: 'a', data: new Float32Array([1, 1, 1]), itemSize: 3 },
        { name: 'b', data: new Float32Array([2, 2, 2]), itemSize: 3 },
        { name: 'c', data: new Float32Array([3, 3, 3]), itemSize: 3 },
      ],
      1
    );
    expect(stride).toBe(9);
    expect(offsets).toEqual({ a: 0, b: 3, c: 6 });
    expect(Array.from(buffer.array as Float32Array)).toEqual([1, 1, 1, 2, 2, 2, 3, 3, 3]);
  });

  it('preserves the per-view normalized flag', () => {
    const { views } = packInterleavedAttributes(
      [
        { name: 'pos', data: new Float32Array([0, 0, 0]), itemSize: 3, normalized: false },
        { name: 'col', data: new Float32Array([1]), itemSize: 1, normalized: true },
      ],
      1
    );
    expect(views.pos.normalized).toBe(false);
    expect(views.col.normalized).toBe(true);
  });

  it('handles instanceCount=0 (empty geometry)', () => {
    const { buffer, stride } = packInterleavedAttributes(
      [{ name: 'a', data: new Float32Array(0), itemSize: 3 }],
      0
    );
    expect(stride).toBe(3);
    expect(buffer.array.length).toBe(0);
  });

  it('rejects negative instanceCount', () => {
    expect(() =>
      packInterleavedAttributes([{ name: 'a', data: new Float32Array(0), itemSize: 1 }], -1)
    ).toThrow(/instanceCount must be >= 0/);
  });

  it('rejects an empty spec list', () => {
    expect(() => packInterleavedAttributes([], 10)).toThrow(/at least one attribute spec/);
  });

  it('rejects mismatched data length', () => {
    expect(() =>
      packInterleavedAttributes(
        // 5 floats but instanceCount=2 * itemSize=3 expects 6.
        [{ name: 'bad', data: new Float32Array(5), itemSize: 3 }],
        2
      )
    ).toThrow(/spec 'bad' has data.length=5, expected 6/);
  });

  it('marks itemSize=4 attributes via the dedicated copy branch', () => {
    const { buffer, stride } = packInterleavedAttributes(
      [{ name: 'rgba', data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), itemSize: 4 }],
      2
    );
    expect(stride).toBe(4);
    expect(Array.from(buffer.array as Float32Array)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });
});

describe('writeInterleavedAttribute', () => {
  it('updates one attribute column without disturbing the others', () => {
    const { buffer, offsets } = packInterleavedAttributes(
      [
        { name: 'pos', data: new Float32Array([1, 2, 3, 4, 5, 6]), itemSize: 3 },
        { name: 's', data: new Float32Array([10, 20]), itemSize: 1 },
      ],
      2
    );
    // Initial: [1, 2, 3, 10, 4, 5, 6, 20]
    expect(Array.from(buffer.array as Float32Array)).toEqual([1, 2, 3, 10, 4, 5, 6, 20]);
    // Three.js InterleavedBuffer exposes `needsUpdate` as a setter
    // that bumps the integer `version` field — there's no getter.
    // Snapshot the version before, expect it to advance after.
    const versionBefore = buffer.version;

    writeInterleavedAttribute(buffer, offsets.s, 1, new Float32Array([99, 88]), 2);

    expect(Array.from(buffer.array as Float32Array)).toEqual([1, 2, 3, 99, 4, 5, 6, 88]);
    expect(buffer.version).toBeGreaterThan(versionBefore);
  });

  it('writes a vec3 column', () => {
    const { buffer, offsets } = packInterleavedAttributes(
      [
        { name: 'a', data: new Float32Array([0, 0, 0, 0, 0, 0]), itemSize: 3 },
        { name: 'b', data: new Float32Array([0, 0, 0, 0, 0, 0]), itemSize: 3 },
      ],
      2
    );
    writeInterleavedAttribute(buffer, offsets.b, 3, new Float32Array([1, 2, 3, 4, 5, 6]), 2);
    // Layout: [a_0.xyz, b_0.xyz, a_1.xyz, b_1.xyz]
    expect(Array.from(buffer.array as Float32Array)).toEqual([0, 0, 0, 1, 2, 3, 0, 0, 0, 4, 5, 6]);
  });

  it('collapses per-attribute and cross-commit update ranges to a single prefix union', () => {
    // The WebGPU backends replay `updateRanges` verbatim (no flush-time
    // merge), and nothing clears ranges while a mesh is not drawn — so
    // repeated writes must leave exactly ONE range covering the widest
    // prefix, or a hidden layer scrubbed through k timepoints uploads
    // k× duplicate full prefixes on its first visible frame.
    const { buffer, offsets } = packInterleavedAttributes(
      [
        { name: 'a', data: new Float32Array(9), itemSize: 3 },
        { name: 'b', data: new Float32Array(3), itemSize: 1 },
      ],
      3 // stride 4, capacity 3 instances
    );
    buffer.clearUpdateRanges(); // start from a clean slate

    // Same-commit pattern: one write per attribute, same count.
    writeInterleavedAttribute(buffer, offsets.a, 3, new Float32Array(9), 3);
    writeInterleavedAttribute(buffer, offsets.b, 1, new Float32Array(3), 3);
    expect(buffer.updateRanges).toEqual([{ start: 0, count: 12 }]);

    // Cross-commit shrink (fewer instances next timepoint): the union
    // must KEEP the wider unflushed prefix, not clobber it.
    writeInterleavedAttribute(buffer, offsets.a, 3, new Float32Array(3), 1);
    expect(buffer.updateRanges).toEqual([{ start: 0, count: 12 }]);
  });

  it('rejects an over-sized source', () => {
    const { buffer, offsets } = packInterleavedAttributes(
      [{ name: 's', data: new Float32Array([0, 0]), itemSize: 1 }],
      2
    );
    expect(() =>
      writeInterleavedAttribute(buffer, offsets.s, 1, new Float32Array([1, 2, 3]), 2)
    ).toThrow(/src.length=3/);
  });

  it('rejects an offset that overflows the stride', () => {
    const { buffer } = packInterleavedAttributes(
      [{ name: 's', data: new Float32Array([0]), itemSize: 1 }],
      1
    );
    expect(() => writeInterleavedAttribute(buffer, 1, 1, new Float32Array([0]), 1)).toThrow(
      /exceeds buffer stride/
    );
  });
});

describe('widenToFloat32', () => {
  it('returns the same reference when already Float32 and no divisor', () => {
    const src = new Float32Array([1, 2, 3]);
    expect(widenToFloat32(src)).toBe(src);
  });

  it('widens a Uint8Array to Float32 verbatim', () => {
    const out = widenToFloat32(new Uint8Array([0, 128, 255]));
    expect(out).toBeInstanceOf(Float32Array);
    expect(Array.from(out)).toEqual([0, 128, 255]);
  });

  it('divides by `divisor` to preserve the normalized [0,1] range', () => {
    const out = widenToFloat32(new Uint8Array([0, 128, 255]), 255);
    expect(out[0]).toBeCloseTo(0.0, 5);
    expect(out[1]).toBeCloseTo(128 / 255, 5);
    expect(out[2]).toBeCloseTo(1.0, 5);
  });

  it('widens Uint16 sources as well', () => {
    const out = widenToFloat32(new Uint16Array([0, 32768, 65535]));
    expect(Array.from(out)).toEqual([0, 32768, 65535]);
  });
});
