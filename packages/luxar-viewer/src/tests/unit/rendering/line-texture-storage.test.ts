/**
 * Texture-backed line storage — unit coverage for the line-bound
 * wrappers and the fused texel writer in `line-geometry.ts` (6
 * texels/segment; layout authority in `element-texture-layout.ts`).
 * Mirrors the point/gsplat texel-writer suites in
 * `point-texture-storage.test.ts` / `splat-texture-storage.test.ts`.
 */

import { describe, it, expect, afterEach } from 'vitest';
import * as THREE from 'three';
import {
  LINE_FLOATS_PER_SEGMENT,
  configureElementTextureLayout,
  resetElementTextureLayoutForTests,
  getLineTextureWidth,
  getMaxLineCapacityPerNode,
  clampLineCapacity,
  lineTextureHeightForCapacity,
} from '../../../rendering/element-texture-layout';
import { elementTexelCapacity } from '../../../rendering/element-storage';
import {
  attachLineStorage,
  getLineTexture,
  writeLineTexels,
  type LineTexelSource,
} from '../../../rendering/line-geometry';

function makeSource(count: number, withScalars = false): LineTexelSource {
  const startPositions = new Float32Array(count * 3);
  const endPositions = new Float32Array(count * 3);
  const startColors = new Float32Array(count * 3);
  const endColors = new Float32Array(count * 3);
  const startWidths = new Float32Array(count);
  const endWidths = new Float32Array(count);
  const startSharpness = new Float32Array(count);
  const endSharpness = new Float32Array(count);
  const segmentLengths = new Float32Array(count);
  const startClipped = new Uint8Array(count);
  const endClipped = new Uint8Array(count);
  const startScalars = withScalars ? new Float32Array(count) : undefined;
  const endScalars = withScalars ? new Float32Array(count) : undefined;
  for (let i = 0; i < count; i++) {
    startPositions.set([i, i + 0.25, i + 0.5], i * 3);
    endPositions.set([i + 1, i + 1.25, i + 1.5], i * 3);
    startColors.set([i * 0.01, i * 0.02, i * 0.03], i * 3);
    endColors.set([i * 0.04, i * 0.05, i * 0.06], i * 3);
    startWidths[i] = 0.5 + i;
    endWidths[i] = 0.25 + i;
    startSharpness[i] = 0.1 * i;
    endSharpness[i] = 0.05 * i;
    segmentLengths[i] = 1.5 + i;
    startClipped[i] = i % 2;
    endClipped[i] = (i + 1) % 2;
    if (startScalars && endScalars) {
      startScalars[i] = 0.05 * i;
      endScalars[i] = 0.07 * i;
    }
  }
  return {
    startPositions,
    endPositions,
    startColors,
    endColors,
    startWidths,
    endWidths,
    startSharpness,
    endSharpness,
    segmentLengths,
    startClipped,
    endClipped,
    startScalars,
    endScalars,
  };
}

afterEach(() => {
  resetElementTextureLayoutForTests();
});

describe('element-texture-layout — line bindings (6 texels/segment)', () => {
  it('defaults to a multiple-of-6 width with a 4096² capacity bound', () => {
    // Width is forced to a multiple of 6 so a segment's texels never
    // straddle a row: floor(4096 / 6) * 6 = 4092.
    expect(getLineTextureWidth()).toBe(4092);
    expect(getLineTextureWidth() % 6).toBe(0);
    expect(getMaxLineCapacityPerNode()).toBe(Math.floor((4092 * 4096) / 6));
  });

  it('computes row-padded texture heights and clamps capacities', () => {
    configureElementTextureLayout(12); // width 12 → 2 segments/row, bound 12*12/6 = 24
    expect(getLineTextureWidth()).toBe(12);
    expect(lineTextureHeightForCapacity(0)).toBe(1);
    expect(lineTextureHeightForCapacity(2)).toBe(1);
    expect(lineTextureHeightForCapacity(3)).toBe(2);
    const max = getMaxLineCapacityPerNode();
    expect(clampLineCapacity(max)).toBe(max);
    expect(clampLineCapacity(max + 1)).toBe(max);
  });
});

describe('attachLineStorage / writeLineTexels — fused writer round-trip', () => {
  it('writes the documented 6-texel layout and reads back exactly', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachLineStorage(geometry, 8);
    expect(getLineTexture(geometry)).toBe(texture);
    const src = makeSource(8, /*withScalars=*/ true);
    const written = writeLineTexels(texture, src, 8);
    expect(written).toBe(8);
    expect(texture.needsUpdate || texture.version > 0).toBe(true);

    const arr = texture.image.data as Float32Array;
    for (let i = 0; i < 8; i++) {
      const o = i * LINE_FLOATS_PER_SEGMENT;
      const p3 = i * 3;
      // texel 0: startPos.xyz, startWidth
      expect(arr[o]).toBe(src.startPositions[p3]);
      expect(arr[o + 1]).toBe(src.startPositions[p3 + 1]);
      expect(arr[o + 2]).toBe(src.startPositions[p3 + 2]);
      expect(arr[o + 3]).toBe(src.startWidths[i]);
      // texel 1: endPos.xyz, endWidth
      expect(arr[o + 4]).toBe(src.endPositions[p3]);
      expect(arr[o + 5]).toBe(src.endPositions[p3 + 1]);
      expect(arr[o + 6]).toBe(src.endPositions[p3 + 2]);
      expect(arr[o + 7]).toBe(src.endWidths[i]);
      // texel 2: startColor.rgb, startSharpness
      expect(arr[o + 8]).toBe(src.startColors[p3]);
      expect(arr[o + 9]).toBe(src.startColors[p3 + 1]);
      expect(arr[o + 10]).toBe(src.startColors[p3 + 2]);
      expect(arr[o + 11]).toBe(src.startSharpness[i]);
      // texel 3: endColor.rgb, endSharpness
      expect(arr[o + 12]).toBe(src.endColors[p3]);
      expect(arr[o + 13]).toBe(src.endColors[p3 + 1]);
      expect(arr[o + 14]).toBe(src.endColors[p3 + 2]);
      expect(arr[o + 15]).toBe(src.endSharpness[i]);
      // texel 4: segmentLength, startClipped, endClipped, 0 (the Uint8
      // clipped flags are read element-wise — 0/1 exact in Float32).
      expect(arr[o + 16]).toBe(src.segmentLengths[i]);
      expect(arr[o + 17]).toBe(src.startClipped[i]);
      expect(arr[o + 18]).toBe(src.endClipped[i]);
      expect(arr[o + 19]).toBe(0.0);
      // texel 5: startScalar, endScalar, per-endpoint alphas
      expect(arr[o + 20]).toBe(src.startScalars![i]);
      expect(arr[o + 21]).toBe(src.endScalars![i]);
      expect(arr[o + 22]).toBe(1.0);
      expect(arr[o + 23]).toBe(1.0);
    }
  });

  it('scalar-absent writes the 0.0 identity and alphas write 1.0 UNCONDITIONALLY', () => {
    // Pool textures are reused — a previous tenant's scalars/alphas must
    // never leak through, so all four texel5 slots are written on every
    // pass (texel4.w is zero-filled for the same determinism).
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachLineStorage(geometry, 4);
    const arr = texture.image.data as Float32Array;
    // Poison the slots as a previous tenant would have left them.
    for (let i = 0; i < 4; i++) {
      const o = i * LINE_FLOATS_PER_SEGMENT;
      arr[o + 19] = 33;
      arr[o + 20] = 77;
      arr[o + 21] = 88;
      arr[o + 22] = 0.25;
      arr[o + 23] = 0.5;
    }
    writeLineTexels(texture, makeSource(4 /* no scalars */), 4);
    for (let i = 0; i < 4; i++) {
      const o = i * LINE_FLOATS_PER_SEGMENT;
      expect(arr[o + 19]).toBe(0.0);
      expect(arr[o + 20]).toBe(0.0);
      expect(arr[o + 21]).toBe(0.0);
      expect(arr[o + 22]).toBe(1.0);
      expect(arr[o + 23]).toBe(1.0);
    }
  });

  it('registers per-row dirty ranges over [0, n), leaving slack rows clean', () => {
    // width 12 → rowFloats = 48, 2 segments/row.
    configureElementTextureLayout(12);
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachLineStorage(geometry, 24); // bound = 12×12/6 = 24 → 12 rows
    expect(texture.image.height).toBe(12);
    const rowFloats = getLineTextureWidth() * 4;
    expect(rowFloats).toBe(48);

    writeLineTexels(texture, makeSource(4), 4); // 4 segments = floats [0, 96) = rows 0,1
    const ranges = texture.updateRanges;
    // Only the 2 written rows are dirty — the 10 slack rows never upload.
    expect(ranges.length).toBe(2);
    const covered = ranges.reduce((s, r) => s + r.count, 0);
    expect(covered).toBe(4 * LINE_FLOATS_PER_SEGMENT); // 96 floats
    // Every range stays within a single texture row (the WebGL path uploads
    // each with height=1 and rejects a row straddle).
    for (const r of ranges) {
      expect(Math.floor(r.start / rowFloats)).toBe(Math.floor((r.start + r.count - 1) / rowFloats));
    }
    // Union spans exactly [0, 96).
    expect(Math.min(...ranges.map((r) => r.start))).toBe(0);
    expect(Math.max(...ranges.map((r) => r.start + r.count))).toBe(96);
  });

  it('writeLineTexels({fromSegment}) writes ONLY the suffix, leaving prefix texels untouched (Stage 2 append)', () => {
    // Narrow width → multiple rows so the appended suffix lands on its own
    // row rather than hitting the single-row full-upload fallback.
    configureElementTextureLayout(12); // rowFloats 48, 2 segments/row
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachLineStorage(geometry, 8); // 8 segments → 4 rows
    const arr = texture.image.data as Float32Array;
    // Full write establishes the prefix, then mark a sentinel on segment
    // 2's startPos.x so we can prove the append pass never touches it.
    writeLineTexels(texture, makeSource(6), 6);
    const sentinel = -12345;
    arr[2 * LINE_FLOATS_PER_SEGMENT] = sentinel;
    texture.clearUpdateRanges();
    // Simulate the renderer flush (three invokes onUpdate after consuming
    // the upload) — the initial full write crossed the >=75% knee into
    // full-upload mode; this test exercises the ranged append path that
    // runs once the upload has flushed.
    texture.onUpdate?.(texture);

    // Append: source is FULL-LENGTH (6), write only segments [4, 6) = row 2.
    const src = makeSource(6);
    const written = writeLineTexels(texture, src, 6, { fromSegment: 4 });
    expect(written).toBe(6);
    // Prefix sentinel survived — texels [0,4) were not rewritten.
    expect(arr[2 * LINE_FLOATS_PER_SEGMENT]).toBe(sentinel);
    // Suffix texels [4,6) hold the new values.
    for (let i = 4; i < 6; i++) {
      expect(arr[i * LINE_FLOATS_PER_SEGMENT]).toBe(src.startPositions[i * 3]);
    }
    // The dirty range starts at segment 4, not 0 — prefix rows never upload.
    const union = texture.updateRanges;
    expect(union.length).toBeGreaterThan(0);
    const minStart = Math.min(...union.map((r) => r.start));
    expect(minStart).toBe(4 * LINE_FLOATS_PER_SEGMENT);
  });

  it('writeLineTexels({fromSegment: 0}) and omitted opts are byte-identical (regression)', () => {
    const g1 = new THREE.InstancedBufferGeometry();
    const g2 = new THREE.InstancedBufferGeometry();
    const t1 = attachLineStorage(g1, 8);
    const t2 = attachLineStorage(g2, 8);
    writeLineTexels(t1, makeSource(6, true), 6);
    writeLineTexels(t2, makeSource(6, true), 6, { fromSegment: 0 });
    expect(Array.from(t2.image.data as Float32Array)).toEqual(
      Array.from(t1.image.data as Float32Array)
    );
  });

  it('throws on source arrays shorter than the requested count BEFORE any store (fail-loud, torn-write-proof)', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachLineStorage(geometry, 8);
    const arr = texture.image.data as Float32Array;
    const before = Array.from(arr);
    // endColors sized for 4 in the MIDDLE of the field list — the guard
    // must reject the whole write before the first store (the
    // interleaved era needed a separate pre-flight sweep for this).
    const torn = { ...makeSource(8), endColors: new Float32Array(4 * 3) };
    expect(() => writeLineTexels(texture, torn, 8)).toThrow(/shorter than count/);
    expect(Array.from(arr)).toEqual(before);
    // A short OPTIONAL scalars array fails loud too.
    const shortScalars = {
      ...makeSource(8, true),
      endScalars: new Float32Array(4),
    };
    expect(() => writeLineTexels(texture, shortScalars, 8)).toThrow(/shorter than count/);
  });

  it('clamps the written count to the texture capacity (memory safety)', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachLineStorage(geometry, 4);
    const cap = elementTexelCapacity(texture, LINE_FLOATS_PER_SEGMENT);
    const src = makeSource(cap + 5);
    expect(writeLineTexels(texture, src, cap + 5)).toBe(cap);
  });

  it('clamps the attach capacity to the per-node bound (structural safety net)', () => {
    // maxTextureSize 12 → width 12, bound = 12×12/6 = 24 segments — an
    // over-bound request must yield a texture no taller than 12 rows and
    // a matching aSortedIndex length, whatever the caller asked for.
    configureElementTextureLayout(12);
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachLineStorage(geometry, 100);
    expect(texture.image.height).toBeLessThanOrEqual(12);
    expect((geometry.getAttribute('aSortedIndex').array as Uint32Array).length).toBe(24);
    expect(elementTexelCapacity(texture, LINE_FLOATS_PER_SEGMENT)).toBe(24);
  });

  it('disposes the texture WITH the geometry (structural lifetime pin)', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachLineStorage(geometry, 4);
    let disposed = false;
    texture.addEventListener('dispose', () => {
      disposed = true;
    });
    geometry.dispose();
    expect(disposed).toBe(true);
  });
});
