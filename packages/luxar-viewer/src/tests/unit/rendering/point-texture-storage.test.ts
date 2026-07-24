/**
 * Stage-2 texture-backed point storage — unit coverage for the
 * point-bound wrappers and the fused texel writer in
 * `point-geometry.ts` (3 texels/point; layout authority in
 * `element-texture-layout.ts`). Mirrors the gsplat texel-writer suite
 * in `splat-texture-storage.test.ts`.
 */

import { describe, it, expect, afterEach } from 'vitest';
import * as THREE from 'three';
import {
  POINT_FLOATS_PER_POINT,
  configureElementTextureLayout,
  resetElementTextureLayoutForTests,
  getPointTextureWidth,
  getMaxPointCapacityPerNode,
  clampPointCapacity,
  pointTextureHeightForCapacity,
} from '../../../rendering/element-texture-layout';
import {
  elementTexelCapacity,
  registerElementTexelDirtyRange,
} from '../../../rendering/element-storage';
import {
  attachPointStorage,
  getPointTexture,
  stampPointPresenceFlags,
  writePointTexels,
  type PointTexelSource,
} from '../../../rendering/point-geometry';

function makeSource(count: number, withScalars = false): PointTexelSource {
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const radii = new Float32Array(count);
  const sharpness = new Float32Array(count);
  const scalars = withScalars ? new Float32Array(count) : undefined;
  for (let i = 0; i < count; i++) {
    positions.set([i, i + 0.25, i + 0.5], i * 3);
    colors.set([i * 0.01, i * 0.02, i * 0.03], i * 3);
    radii[i] = 0.5 + i;
    sharpness[i] = 0.1 * i;
    if (scalars) scalars[i] = 0.05 * i;
  }
  return { positions, colors, radii, sharpness, scalars };
}

afterEach(() => {
  resetElementTextureLayoutForTests();
});

describe('element-texture-layout — point bindings (3 texels/point)', () => {
  it('defaults to a multiple-of-3 width with a 4096² capacity bound', () => {
    // Width is forced to a multiple of 3 so a point's texels never
    // straddle a row: floor(4096 / 3) * 3 = 4095.
    expect(getPointTextureWidth()).toBe(4095);
    expect(getPointTextureWidth() % 3).toBe(0);
    expect(getMaxPointCapacityPerNode()).toBe(Math.floor((4095 * 4096) / 3));
  });

  it('computes row-padded texture heights and clamps capacities', () => {
    configureElementTextureLayout(9); // width 9 → 3 points/row, bound 9*9/3 = 27
    expect(getPointTextureWidth()).toBe(9);
    expect(pointTextureHeightForCapacity(0)).toBe(1);
    expect(pointTextureHeightForCapacity(3)).toBe(1);
    expect(pointTextureHeightForCapacity(4)).toBe(2);
    const max = getMaxPointCapacityPerNode();
    expect(clampPointCapacity(max)).toBe(max);
    expect(clampPointCapacity(max + 1)).toBe(max);
  });
});

describe('attachPointStorage / writePointTexels — fused writer round-trip', () => {
  it('writes the documented 3-texel layout and reads back exactly', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachPointStorage(geometry, 8);
    expect(getPointTexture(geometry)).toBe(texture);
    const src = makeSource(8, /*withScalars=*/ true);
    const written = writePointTexels(texture, src, 8);
    expect(written).toBe(8);
    expect(texture.needsUpdate || texture.version > 0).toBe(true);

    const arr = texture.image.data as Float32Array;
    for (let i = 0; i < 8; i++) {
      const o = i * POINT_FLOATS_PER_POINT;
      // texel 0: center.xyz, radius
      expect(arr[o]).toBe(src.positions[i * 3]);
      expect(arr[o + 1]).toBe(src.positions[i * 3 + 1]);
      expect(arr[o + 2]).toBe(src.positions[i * 3 + 2]);
      expect(arr[o + 3]).toBe(src.radii[i]);
      // texel 1: color.rgb, sharpness
      expect(arr[o + 4]).toBe(src.colors[i * 3]);
      expect(arr[o + 5]).toBe(src.colors[i * 3 + 1]);
      expect(arr[o + 6]).toBe(src.colors[i * 3 + 2]);
      expect(arr[o + 7]).toBe(src.sharpness[i]);
      // texel 2: scalar, alpha (per-point opacity)
      expect(arr[o + 8]).toBe(src.scalars![i]);
      expect(arr[o + 9]).toBe(1.0);
    }
  });

  it('scalar-absent writes the 0.0 identity and alpha writes 1.0 UNCONDITIONALLY', () => {
    // Pool textures are reused — a previous tenant's scalar/alpha must
    // never leak through, so both slots are written on every pass.
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachPointStorage(geometry, 4);
    const arr = texture.image.data as Float32Array;
    // Poison the slots as a previous tenant would have left them.
    for (let i = 0; i < 4; i++) {
      arr[i * POINT_FLOATS_PER_POINT + 8] = 77;
      arr[i * POINT_FLOATS_PER_POINT + 9] = 0.25;
    }
    writePointTexels(texture, makeSource(4 /* no scalars */), 4);
    for (let i = 0; i < 4; i++) {
      expect(arr[i * POINT_FLOATS_PER_POINT + 8]).toBe(0.0);
      expect(arr[i * POINT_FLOATS_PER_POINT + 9]).toBe(1.0);
    }
  });

  it('registers per-row dirty ranges over [0, n), leaving slack rows clean', () => {
    // width 6 → rowFloats = 24, 2 points/row.
    configureElementTextureLayout(6);
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachPointStorage(geometry, 12); // bound = 6×6/3 = 12 → 6 rows
    expect(texture.image.height).toBe(6);
    const rowFloats = getPointTextureWidth() * 4;
    expect(rowFloats).toBe(24);

    writePointTexels(texture, makeSource(4), 4); // 4 points = floats [0, 48) = rows 0,1
    const ranges = texture.updateRanges;
    // Only the 2 written rows are dirty — the 4 slack rows never upload.
    expect(ranges.length).toBe(2);
    const covered = ranges.reduce((s, r) => s + r.count, 0);
    expect(covered).toBe(4 * POINT_FLOATS_PER_POINT); // 48 floats
    // Every range stays within a single texture row (the WebGL path uploads
    // each with height=1 and rejects a row straddle).
    for (const r of ranges) {
      expect(Math.floor(r.start / rowFloats)).toBe(Math.floor((r.start + r.count - 1) / rowFloats));
    }
    // Union spans exactly [0, 48).
    expect(Math.min(...ranges.map((r) => r.start))).toBe(0);
    expect(Math.max(...ranges.map((r) => r.start + r.count))).toBe(48);
  });

  it('writePointTexels({fromPoint}) writes ONLY the suffix, leaving prefix texels untouched (Stage 2 append)', () => {
    // Narrow width → multiple rows so the appended suffix lands on its own
    // row rather than hitting the single-row full-upload fallback.
    configureElementTextureLayout(6); // rowFloats 24, 2 points/row
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachPointStorage(geometry, 8); // 8 points → 4 rows
    const arr = texture.image.data as Float32Array;
    // Full write establishes the prefix, then mark a sentinel on point 2's
    // center.x so we can prove the append pass never touches it.
    writePointTexels(texture, makeSource(6), 6);
    const sentinel = -12345;
    arr[2 * POINT_FLOATS_PER_POINT] = sentinel;
    texture.clearUpdateRanges();
    // Simulate the renderer flush (three invokes onUpdate after consuming
    // the upload) — the initial full write crossed the >=75% knee into
    // full-upload mode, and a pre-flush append correctly STAYS full
    // (pinned by the points pending-full test below); this test exercises
    // the ranged append path that runs once the upload has flushed.
    texture.onUpdate?.(texture);

    // Append: source is FULL-LENGTH (6), write only points [4, 6) = row 2.
    const src = makeSource(6);
    const written = writePointTexels(texture, src, 6, { fromPoint: 4 });
    expect(written).toBe(6);
    // Prefix sentinel survived — texels [0,4) were not rewritten.
    expect(arr[2 * POINT_FLOATS_PER_POINT]).toBe(sentinel);
    // Suffix texels [4,6) hold the new values.
    for (let i = 4; i < 6; i++) {
      expect(arr[i * POINT_FLOATS_PER_POINT]).toBe(src.positions[i * 3]);
    }
    // The dirty range starts at point 4, not 0 — prefix rows never upload.
    const union = texture.updateRanges;
    expect(union.length).toBeGreaterThan(0);
    const minStart = Math.min(...union.map((r) => r.start));
    expect(minStart).toBe(4 * POINT_FLOATS_PER_POINT);
  });

  it('a pending FULL upload is NOT downgraded by a later append (three-geometry twin)', () => {
    // Points twin of the splat suite's pending-full test: full-upload mode
    // is encoded as needsUpdate + EMPTY updateRanges — invisible to the
    // range fold. A full write dirtying >= 75% of rows enters full mode;
    // an append BEFORE any flush must NOT register partial ranges (that
    // would downgrade the full upload and leave the prefix rendering the
    // previous commit's texels on classic WebGL).
    configureElementTextureLayout(6); // 2 points/row
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachPointStorage(geometry, 4); // 4 points → 2 rows
    texture.clearUpdateRanges();
    texture.onUpdate?.(texture); // simulate the attach upload flush
    const v0 = texture.version;

    // Full write of all 4 points → 2/2 rows dirty → full-upload mode.
    writePointTexels(texture, makeSource(4), 4);
    expect(texture.updateRanges.length).toBe(0);
    expect(texture.version).toBeGreaterThan(v0);

    // A suffix register before the flush must keep full mode (no ranges).
    registerElementTexelDirtyRange(texture, POINT_FLOATS_PER_POINT, 3, 4);
    expect(texture.updateRanges.length).toBe(0);

    // A simulated flush ends the pending-full state; ranged uploads resume.
    texture.onUpdate?.(texture);
    registerElementTexelDirtyRange(texture, POINT_FLOATS_PER_POINT, 3, 4);
    expect(texture.updateRanges.length).toBeGreaterThan(0);
  });

  it('writePointTexels({fromPoint: 0}) and omitted opts are byte-identical (regression)', () => {
    const g1 = new THREE.InstancedBufferGeometry();
    const g2 = new THREE.InstancedBufferGeometry();
    const t1 = attachPointStorage(g1, 8);
    const t2 = attachPointStorage(g2, 8);
    writePointTexels(t1, makeSource(6, true), 6);
    writePointTexels(t2, makeSource(6, true), 6, { fromPoint: 0 });
    expect(Array.from(t2.image.data as Float32Array)).toEqual(
      Array.from(t1.image.data as Float32Array)
    );
  });

  it('throws on source arrays shorter than the requested count (fail-loud contract)', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachPointStorage(geometry, 8);
    const src = makeSource(4); // arrays sized for 4, count says 8
    expect(() => writePointTexels(texture, src, 8)).toThrow(/shorter than count/);
    // A short OPTIONAL scalars array fails loud too.
    const shortScalars = { ...makeSource(8), scalars: new Float32Array(4) };
    expect(() => writePointTexels(texture, shortScalars, 8)).toThrow(/shorter than count/);
  });

  it('clamps the written count to the texture capacity (memory safety)', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachPointStorage(geometry, 4);
    const cap = elementTexelCapacity(texture, POINT_FLOATS_PER_POINT);
    const src = makeSource(cap + 5);
    expect(writePointTexels(texture, src, cap + 5)).toBe(cap);
  });

  it('clamps the attach capacity to the per-node bound (structural safety net)', () => {
    // maxTextureSize 6 → width 6, bound = 6×6/3 = 12 points — an
    // over-bound request must yield a texture no taller than 6 rows and
    // a matching aSortedIndex length, whatever the caller asked for.
    configureElementTextureLayout(6);
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachPointStorage(geometry, 100);
    expect(texture.image.height).toBeLessThanOrEqual(6);
    expect((geometry.getAttribute('aSortedIndex').array as Uint32Array).length).toBe(12);
    expect(elementTexelCapacity(texture, POINT_FLOATS_PER_POINT)).toBe(12);
  });

  it('disposes the texture WITH the geometry (structural lifetime pin)', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachPointStorage(geometry, 4);
    let disposed = false;
    texture.addEventListener('dispose', () => {
      disposed = true;
    });
    geometry.dispose();
    expect(disposed).toBe(true);
  });
});

describe('writePointTexels — RGBA color layout (per-point opacity, volumetric Phase 3)', () => {
  // The colors array is strided by `colorComponents` (3 = RGB, 4 = RGBA):
  // point i's color lives at colors[i*colorK .. i*colorK+2] and — for RGBA —
  // its alpha at colors[i*colorK+3], packed into texel2.y. Mirrors the gsplat
  // writer's SplatTexelSource.colorComponents.
  function makeRgbaSource(count: number): PointTexelSource {
    const src = makeSource(count);
    const colors = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      colors[i * 4] = 0.1 + i;
      colors[i * 4 + 1] = 0.2 + i;
      colors[i * 4 + 2] = 0.3 + i;
      colors[i * 4 + 3] = 0.5 / (i + 1); // distinct per-point alpha
    }
    return { ...src, colors, colorComponents: 4 };
  }

  it('writes each point’s alpha into texel2.y and RGB at STRIDE 4 (two-point stride proof)', () => {
    // Two points with distinct RGBA tuples: a stride-3 read would smear
    // point 1's RGB (reading [a0, r1, g1]) and drop both alphas — asserting
    // BOTH points' full tuples pins the stride, not just the alpha copy.
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachPointStorage(geometry, 4);
    const src = makeRgbaSource(2);
    writePointTexels(texture, src, 2);
    const arr = texture.image.data as Float32Array;
    for (let i = 0; i < 2; i++) {
      const o = i * POINT_FLOATS_PER_POINT;
      // texel 1: color.rgb from the point's OWN stride-4 tuple
      expect(arr[o + 4]).toBeCloseTo(0.1 + i, 6);
      expect(arr[o + 5]).toBeCloseTo(0.2 + i, 6);
      expect(arr[o + 6]).toBeCloseTo(0.3 + i, 6);
      // texel 2.y: the point's own alpha column, NOT the 1.0 identity
      expect(arr[o + 9]).toBeCloseTo(0.5 / (i + 1), 6);
    }
  });

  it('an RGB source resets alpha to the opaque 1.0 identity (pool-reuse overwrite)', () => {
    // An RGBA tenant followed by an RGB tenant on the SAME pool texture:
    // the RGB pass must overwrite the previous alphas with 1.0.
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachPointStorage(geometry, 4);
    writePointTexels(texture, makeRgbaSource(2), 2);
    const arr = texture.image.data as Float32Array;
    expect(arr[9]).not.toBe(1.0); // RGBA alpha landed first
    writePointTexels(texture, makeSource(2), 2);
    expect(arr[9]).toBe(1.0);
    expect(arr[POINT_FLOATS_PER_POINT + 9]).toBe(1.0);
  });

  it('throws when RGBA colors are shorter than count × 4 (length guard uses n×colorK)', () => {
    // colors sized count*3 would satisfy an RGB-stride guard — with
    // colorComponents=4 the guard must demand count*4 and fail loud
    // instead of writing NaN alphas from out-of-bounds reads.
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachPointStorage(geometry, 8);
    const src = { ...makeRgbaSource(8), colors: new Float32Array(8 * 3) };
    expect(() => writePointTexels(texture, src, 8)).toThrow(/shorter than count/);
  });
});

describe('stampPointPresenceFlags — the shared presence-stamp chokepoint', () => {
  // Mutation-found on the LINES twin (volumetric phase-4 double-check):
  // hardwiring the alpha stamp to false survived the entire unit suite —
  // the stamp is the sole source the commit sync reads for the
  // material's uHasElementAlpha gate. This is the points twin of that
  // pin, added when the two historical inline stamp blocks (node
  // factory + pool adapter) were extracted into one helper.
  it('stamps all five flags from source presence + colorK, and REFRESHES across writes', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    stampPointPresenceFlags(
      geometry,
      {
        scalars: new Float32Array(0), // zero-length declared field COUNTS as present
        colors: new Float32Array(12),
        radii: new Float32Array(4),
        sharpness: new Float32Array(4),
      },
      4
    );
    expect(geometry.userData.hasScalars).toBe(true);
    expect(geometry.userData.hasColors).toBe(true);
    expect(geometry.userData.hasRadii).toBe(true);
    expect(geometry.userData.hasSharpness).toBe(true);
    expect(geometry.userData.hasElementAlpha).toBe(true);

    // Pool-tenant flip: an RGB tenant without optional fields must reset
    // every flag — a stale true would leak the previous tenant's stamp.
    stampPointPresenceFlags(geometry, { colors: null }, 3);
    expect(geometry.userData.hasScalars).toBe(false);
    expect(geometry.userData.hasColors).toBe(false);
    expect(geometry.userData.hasRadii).toBe(false);
    expect(geometry.userData.hasSharpness).toBe(false);
    expect(geometry.userData.hasElementAlpha).toBe(false);
  });

  it('hasElementAlpha derives from the SOURCE layout (colorK), not buffer contents', () => {
    // RGBA layout with all-1.0 alphas is still hasElementAlpha=true (the
    // gate is about layout, not values); RGB layout is false even with
    // colors present.
    const geometry = new THREE.InstancedBufferGeometry();
    stampPointPresenceFlags(geometry, { colors: new Float32Array(8) }, 4);
    expect(geometry.userData.hasElementAlpha).toBe(true);
    stampPointPresenceFlags(geometry, { colors: new Float32Array(6) }, 3);
    expect(geometry.userData.hasElementAlpha).toBe(false);
  });
});
