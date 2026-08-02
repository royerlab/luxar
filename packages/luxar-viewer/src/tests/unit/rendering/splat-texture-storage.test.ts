/**
 * Phase 1 texture-backed splat storage — unit coverage for the layout
 * authority (`element-texture-layout.ts`), the storage helpers
 * (`element-storage.ts` + the gsplat-bound wrappers in
 * `gsplat-geometry.ts`), the pool adapter's growth/dispose behavior,
 * byte accounting, and the commit material sync. See
 * `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md` §4 for the design.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';
import {
  SPLAT_FLOATS_PER_SPLAT,
  SPLAT_TEXELS_PER_SPLAT,
  configureElementTextureLayout,
  resetElementTextureLayoutForTests,
  getSplatTextureWidth,
  getMaxSplatCapacityPerNode,
  clampSplatCapacity,
  splatTextureHeightForCapacity,
  getPlaceholderElementTexture,
} from '../../../rendering/element-texture-layout';
import {
  SORTED_INDEX_CHUNK_ELEMENTS,
  acknowledgeSortedIndexOrderingDraw,
  activeSortedIndexSlot,
  getActiveSortedIndexAttribute,
  cancelAllSortedIndexOrderingApplies,
  cancelSortedIndexOrderingApply,
  configureSortedIndexChunkedApply,
  elementTexelCapacity,
  hasPendingSortedIndexOrderingApply,
  pumpSortedIndexOrderingApply,
  registerElementTexelDirtyRange,
  setSortedIndexChunkElementsForTests,
  writeSortedIndexIdentity,
  writeSortedIndexIdentityRange,
  writeSortedIndexOrdering,
  markElementTextureFullDirty,
} from '../../../rendering/element-storage';
import {
  attachSplatStorage,
  getSplatTexture,
  writeSplatTexels,
  type SplatTexelSource,
} from '../../../rendering/gsplat-geometry';
import { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import { estimateGeometryBytes } from '../../../rendering/gpu-buffer-pool/geometry-bytes';
import { syncGSplatMaterialWithGeometry } from '../../../rendering/material-sync-helpers';
import { GSplatMaterial } from '../../../rendering/materials/gsplat/material-glsl';
import { materialManager } from '../../../rendering/material-manager';
import { GSplatPickingMaterial } from '../../../rendering/picking/gsplat/material';

function makeSource(count: number): SplatTexelSource {
  const centers = new Float32Array(count * 3);
  const choleskyFactors = new Float32Array(count * 6);
  const amplitudes = new Float32Array(count);
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    centers.set([i, i + 0.25, i + 0.5], i * 3);
    // [L00, L10, L11, L20, L21, L22]
    choleskyFactors.set([1 + i, 0.1 * i, 2 + i, 0.2 * i, 0.3 * i, 3 + i], i * 6);
    amplitudes[i] = 0.5 + i;
    colors.set([i * 0.01, i * 0.02, i * 0.03], i * 3);
  }
  return { centers, choleskyFactors, amplitudes, colors };
}

afterEach(() => {
  resetElementTextureLayoutForTests();
});

describe('element-texture-layout — texel address math', () => {
  it('defaults to a 4096-wide texture with a 4096² capacity bound', () => {
    expect(getSplatTextureWidth()).toBe(4096);
    expect(getMaxSplatCapacityPerNode()).toBe((4096 * 4096) / SPLAT_TEXELS_PER_SPLAT);
  });

  it('caps the width at min(4096, maxTextureSize) and forces a multiple of 4', () => {
    configureElementTextureLayout(16384);
    expect(getSplatTextureWidth()).toBe(4096); // never wider than 4096

    // Non-4096 width: a 2048-class device.
    configureElementTextureLayout(2048);
    expect(getSplatTextureWidth()).toBe(2048);
    expect(getMaxSplatCapacityPerNode()).toBe((2048 * 2048) / SPLAT_TEXELS_PER_SPLAT);

    // A pathological non-multiple-of-4 limit is rounded DOWN so a
    // splat's 4 texels can never straddle a row boundary.
    configureElementTextureLayout(2050);
    expect(getSplatTextureWidth()).toBe(2048);

    // Sub-4 limits floor at 4 (would otherwise round to width 0 and
    // divide-by-zero the height math).
    configureElementTextureLayout(3);
    expect(getSplatTextureWidth()).toBe(4);
  });

  it('computes row-padded texture heights', () => {
    configureElementTextureLayout(4096);
    // 1024 splats/row at width 4096.
    expect(splatTextureHeightForCapacity(0)).toBe(1);
    expect(splatTextureHeightForCapacity(1)).toBe(1);
    expect(splatTextureHeightForCapacity(1024)).toBe(1);
    expect(splatTextureHeightForCapacity(1025)).toBe(2);
  });

  it('clamps requested capacities to the per-node texture bound', () => {
    configureElementTextureLayout(2048);
    const max = getMaxSplatCapacityPerNode();
    expect(clampSplatCapacity(max - 1)).toBe(max - 1);
    expect(clampSplatCapacity(max)).toBe(max);
    expect(clampSplatCapacity(max + 1)).toBe(max);
  });

  it('exposes a single shared placeholder texture', () => {
    expect(getPlaceholderElementTexture()).toBe(getPlaceholderElementTexture());
  });
});

describe('attachSplatStorage / writeSplatTexels — fused writer round-trip', () => {
  it('writes the documented 4-texel layout and reads back exactly', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 8);
    const src = makeSource(8);
    const written = writeSplatTexels(texture, src, 8);
    expect(written).toBe(8);
    expect(texture.needsUpdate || texture.version > 0).toBe(true);

    const arr = texture.image.data as Float32Array;
    for (let i = 0; i < 8; i++) {
      const o = i * SPLAT_FLOATS_PER_SPLAT;
      // texel 0: center.xyz, amplitude
      expect(arr[o]).toBe(src.centers[i * 3]);
      expect(arr[o + 1]).toBe(src.centers[i * 3 + 1]);
      expect(arr[o + 2]).toBe(src.centers[i * 3 + 2]);
      expect(arr[o + 3]).toBe(src.amplitudes[i]);
      // texel 1: [L00, L10, L11, L20]
      expect(arr[o + 4]).toBe(src.choleskyFactors[i * 6]);
      expect(arr[o + 5]).toBe(src.choleskyFactors[i * 6 + 1]);
      expect(arr[o + 6]).toBe(src.choleskyFactors[i * 6 + 2]);
      expect(arr[o + 7]).toBe(src.choleskyFactors[i * 6 + 3]);
      // texel 2: [L21, L22], color.rg
      expect(arr[o + 8]).toBe(src.choleskyFactors[i * 6 + 4]);
      expect(arr[o + 9]).toBe(src.choleskyFactors[i * 6 + 5]);
      expect(arr[o + 10]).toBe(src.colors[i * 3]);
      expect(arr[o + 11]).toBe(src.colors[i * 3 + 1]);
      // texel 3: color.b, alpha (per-splat opacity). RGB source ⇒ alpha
      // defaults to 1.0 (opaque), written UNCONDITIONALLY so a reused pool
      // texel never leaks a previous tenant's alpha.
      expect(arr[o + 12]).toBe(src.colors[i * 3 + 2]);
      expect(arr[o + 13]).toBe(1.0);
    }
  });

  it('packs RGBA colors: alpha (per-splat opacity) lands in texel3.y', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 8);
    const base = makeSource(8);
    // Widen colors to RGBA with a distinct per-splat alpha ramp.
    const rgba = new Float32Array(8 * 4);
    for (let i = 0; i < 8; i++) {
      rgba.set(
        [base.colors[i * 3], base.colors[i * 3 + 1], base.colors[i * 3 + 2], i * 0.1],
        i * 4
      );
    }
    const src: SplatTexelSource = { ...base, colors: rgba, colorComponents: 4 };
    const written = writeSplatTexels(texture, src, 8);
    expect(written).toBe(8);

    const arr = texture.image.data as Float32Array;
    for (let i = 0; i < 8; i++) {
      const o = i * SPLAT_FLOATS_PER_SPLAT;
      expect(arr[o + 10]).toBe(rgba[i * 4]); // color.r
      expect(arr[o + 11]).toBe(rgba[i * 4 + 1]); // color.g
      expect(arr[o + 12]).toBe(rgba[i * 4 + 2]); // color.b
      expect(arr[o + 13]).toBeCloseTo(i * 0.1, 6); // alpha
    }
  });

  it('registers per-row dirty ranges over [0, n), leaving slack rows clean', () => {
    // width 8 → 2 texels-per-splat-row math: rowFloats = 32, 2 splats/row.
    configureElementTextureLayout(8);
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 16); // bound = 8×8/4 = 16 → 8 rows
    expect(texture.image.height).toBe(8);
    const rowFloats = getSplatTextureWidth() * 4;
    expect(rowFloats).toBe(32);

    writeSplatTexels(texture, makeSource(4), 4); // 4 splats = floats [0, 64) = rows 0,1
    const ranges = texture.updateRanges;
    // Only the 2 written rows are dirty — the 6 slack rows never upload.
    expect(ranges.length).toBe(2);
    const covered = ranges.reduce((s, r) => s + r.count, 0);
    expect(covered).toBe(4 * SPLAT_FLOATS_PER_SPLAT); // 64 floats
    // Every range stays within a single texture row (the WebGL path uploads
    // each with height=1 and rejects a row straddle).
    for (const r of ranges) {
      expect(Math.floor(r.start / rowFloats)).toBe(Math.floor((r.start + r.count - 1) / rowFloats));
    }
    // Union spans exactly [0, 64).
    expect(Math.min(...ranges.map((r) => r.start))).toBe(0);
    expect(Math.max(...ranges.map((r) => r.start + r.count))).toBe(64);
  });

  it('falls back to a full-image upload (empty ranges) when most rows are dirty', () => {
    configureElementTextureLayout(8);
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 16); // 8 rows
    // 15 splats = floats [0, 240) = rows 0..7 = all 8 rows ≥ 0.75×8 → full upload.
    writeSplatTexels(texture, makeSource(15), 15);
    expect(texture.updateRanges.length).toBe(0);
    expect(texture.version > 0 || texture.needsUpdate).toBe(true);
  });

  it('collapses pending ranges across hidden commits into one contiguous per-row set', () => {
    // WebGPU backends replay texture ranges verbatim and never clear them,
    // so successive writes while hidden must union, not accumulate.
    configureElementTextureLayout(8);
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 16);
    writeSplatTexels(texture, makeSource(2), 2); // floats [0, 32) = row 0
    writeSplatTexels(texture, makeSource(4), 4); // floats [0, 64) = rows 0,1
    const ranges = texture.updateRanges;
    // One range per dirty row (2), never four accumulated fragments.
    expect(ranges.length).toBe(2);
    expect(Math.max(...ranges.map((r) => r.start + r.count))).toBe(64);
  });

  it('registers an append-only span [firstSplat, endSplat) (Stage-2 contract)', () => {
    configureElementTextureLayout(8);
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 16);
    texture.clearUpdateRanges();
    // Splats [2, 4) = floats [32, 64) = row 1 only — no prefix re-upload.
    registerElementTexelDirtyRange(texture, SPLAT_FLOATS_PER_SPLAT, 2, 4);
    const ranges = texture.updateRanges;
    expect(ranges.length).toBe(1);
    expect(ranges[0].start).toBe(32);
    expect(ranges[0].count).toBe(32);
  });

  it('an EMPTY span with no pending ranges triggers NO upload (reused-pool zero-count commit)', () => {
    // Empty updateRanges + needsUpdate would take three's FULL-image
    // texSubImage2D path — re-uploading a reused pool texture's whole
    // capacity-sized backing store on every empty commit (e.g. nD
    // navigation into an empty slice). Nothing was written, so nothing
    // may upload; fresh textures are covered by attachElementStorage's
    // own needsUpdate.
    configureElementTextureLayout(8);
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 16);
    texture.clearUpdateRanges();
    texture.needsUpdate = false; // simulate an already-uploaded pool texture
    registerElementTexelDirtyRange(texture, SPLAT_FLOATS_PER_SPLAT, 0, 0);
    expect(texture.updateRanges.length).toBe(0);
    // needsUpdate is a setter incrementing texture.version; an untouched
    // version means no upload was scheduled.
    expect(texture.version).toBe(1); // 1 = the attach-time needsUpdate only
  });

  it('a pending FULL upload is NOT downgraded by a later append (fuzz-found stale-prefix bug)', () => {
    // Full-upload mode is encoded as needsUpdate + EMPTY updateRanges —
    // invisible to the range fold. Sequence: full write dirtying >= 75%
    // of rows (full mode), then an append BEFORE any flush. The append
    // must NOT register partial ranges (that would downgrade the full
    // upload and leave the prefix rendering the previous commit's texels
    // on classic WebGL).
    configureElementTextureLayout(8);
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 4); // 2 splats/row → 2 rows
    texture.clearUpdateRanges();
    texture.onUpdate?.(texture); // simulate the attach upload flush
    const v0 = texture.version;

    // Full write of all 4 splats → 2/2 rows dirty → full-upload mode.
    writeSplatTexels(texture, makeSource(4), 4);
    expect(texture.updateRanges.length).toBe(0);
    expect(texture.version).toBeGreaterThan(v0);

    // Append splat [4..4) — capacity-clamped no-op is separate; use a
    // REAL suffix register instead (splats 3→4 shape via the helper).
    registerElementTexelDirtyRange(texture, SPLAT_FLOATS_PER_SPLAT, 3, 4);
    // Full mode retained: still no partial ranges.
    expect(texture.updateRanges.length).toBe(0);

    // Simulated flush (three calls onUpdate after consuming the upload)
    // ends the pending-full state; ranged uploads resume.
    texture.onUpdate?.(texture);
    registerElementTexelDirtyRange(texture, SPLAT_FLOATS_PER_SPLAT, 3, 4);
    expect(texture.updateRanges.length).toBeGreaterThan(0);
  });

  it('markElementTextureFullDirty keeps full mode across later ranged writes (context restore)', () => {
    configureElementTextureLayout(8);
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 16);
    texture.clearUpdateRanges();
    texture.onUpdate?.(texture);

    markElementTextureFullDirty(texture);
    registerElementTexelDirtyRange(texture, SPLAT_FLOATS_PER_SPLAT, 0, 2);
    expect(texture.updateRanges.length).toBe(0); // full upload still pending
    texture.onUpdate?.(texture);
    registerElementTexelDirtyRange(texture, SPLAT_FLOATS_PER_SPLAT, 0, 2);
    expect(texture.updateRanges.length).toBeGreaterThan(0);
  });

  it('an EMPTY span does NOT inflate pending ranges up to its position (fold non-inflation)', () => {
    // Seeding the collapse fold with an empty span's position would stretch
    // a pending [100, 300) up to the span's floats — uploading a huge run of
    // clean data (and here also tripping the ≥75%-dirty-rows full-image
    // fallback, wiping the ranges entirely). Default width 4096 → rowFloats
    // 16384; capacity 2048 splats → 2 rows, so [100, 300) stays 1 dirty row.
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 2048);
    texture.clearUpdateRanges();
    texture.addUpdateRange(100, 200); // pending floats [100, 300), row 0
    // At-capacity no-op append at splat 2000 (float 32000): an empty span.
    registerElementTexelDirtyRange(texture, SPLAT_FLOATS_PER_SPLAT, 2000, 2000);
    const ranges = texture.updateRanges;
    expect(ranges.length).toBe(1);
    expect(ranges[0].start).toBe(100);
    expect(ranges[0].start + ranges[0].count).toBe(300); // no inflation to 2000×FLOATS
  });

  it('splits by the TEXTURE width, not the reconfigured global width (renderer-swap safety)', () => {
    // Allocate at width 8, then reconfigure the session width (as a
    // backend/renderer swap does). The dirty-range split must follow the
    // texture's OWN width (8 → rowFloats 32), or a range would straddle rows
    // and the WebGL upload would fail with INVALID_VALUE.
    configureElementTextureLayout(8);
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 16); // width 8, 8 rows
    expect(texture.image.width).toBe(8);
    configureElementTextureLayout(4096); // global width now diverges from the texture
    writeSplatTexels(texture, makeSource(4), 4); // 4 splats = floats [0, 64)
    const rowFloats = texture.image.width * 4; // 32 — the TEXTURE's stride
    const ranges = texture.updateRanges;
    for (const r of ranges) {
      expect(r.count).toBeLessThanOrEqual(rowFloats);
      expect(Math.floor(r.start / rowFloats)).toBe(Math.floor((r.start + r.count - 1) / rowFloats));
    }
    // With the texture width (8) it's 2 rows; the buggy global-width (4096)
    // path would emit a single 64-float range straddling both rows.
    expect(ranges.length).toBe(2);
  });

  it('writeSplatTexels({fromSplat}) writes ONLY the suffix, leaving prefix texels untouched (Stage 2 append)', () => {
    // Narrow width → multiple rows so the appended suffix lands on its own
    // row rather than hitting the single-row full-upload fallback.
    configureElementTextureLayout(8); // rowFloats 32, 2 splats/row
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 8); // 8 splats → 4 rows
    const arr = texture.image.data as Float32Array;
    // Full write establishes the prefix, then mark a sentinel on splat 2's
    // center.x so we can prove the append pass never touches it.
    writeSplatTexels(texture, makeSource(6), 6);
    const sentinel = -12345;
    arr[2 * SPLAT_FLOATS_PER_SPLAT] = sentinel;
    texture.clearUpdateRanges();
    // Simulate the renderer flush (three invokes onUpdate after consuming
    // the upload) — the initial full write crossed the >=75% knee into
    // full-upload mode, and a pre-flush append correctly STAYS full
    // (pinned by the pending-full test); this test exercises the ranged
    // append path that runs once the upload has flushed.
    texture.onUpdate?.(texture);

    // Append: source is FULL-LENGTH (6), write only splats [4, 6) = row 2.
    const src = makeSource(6);
    const written = writeSplatTexels(texture, src, 6, { fromSplat: 4 });
    expect(written).toBe(6);
    // Prefix sentinel survived — texels [0,4) were not rewritten.
    expect(arr[2 * SPLAT_FLOATS_PER_SPLAT]).toBe(sentinel);
    // Suffix texels [4,6) hold the new values.
    for (let i = 4; i < 6; i++) {
      expect(arr[i * SPLAT_FLOATS_PER_SPLAT]).toBe(src.centers[i * 3]);
    }
    // The dirty range starts at splat 4, not 0 — prefix rows never upload.
    const union = texture.updateRanges;
    expect(union.length).toBeGreaterThan(0);
    const minStart = Math.min(...union.map((r) => r.start));
    expect(minStart).toBe(4 * SPLAT_FLOATS_PER_SPLAT);
  });

  it('writeSplatTexels({fromSplat: 0}) and omitted opts are byte-identical (regression)', () => {
    const g1 = new THREE.InstancedBufferGeometry();
    const g2 = new THREE.InstancedBufferGeometry();
    const t1 = attachSplatStorage(g1, 8);
    const t2 = attachSplatStorage(g2, 8);
    writeSplatTexels(t1, makeSource(6), 6);
    writeSplatTexels(t2, makeSource(6), 6, { fromSplat: 0 });
    expect(Array.from(t2.image.data as Float32Array)).toEqual(
      Array.from(t1.image.data as Float32Array)
    );
  });

  it('writeSortedIndexIdentityRange appends identity for the suffix, preserving the prefix permutation', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    attachSplatStorage(geometry, 16);
    // Prefix carries a real depth-sort permutation over [0,4). An
    // ordering stages into the back buffer, so drain the pump to make it
    // the live one before appending onto it.
    writeSortedIndexOrdering(geometry, new Uint32Array([3, 2, 1, 0]), 4);
    while (pumpSortedIndexOrderingApply(geometry).more) {
      /* drain */
    }
    const attr = getActiveSortedIndexAttribute(geometry) as THREE.InstancedBufferAttribute;
    const arr = attr.array as Uint32Array;
    // Append identity for [4, 8): prefix permutation stays, suffix = identity.
    writeSortedIndexIdentityRange(geometry, 4, 8);
    expect(Array.from(arr.subarray(0, 8))).toEqual([3, 2, 1, 0, 4, 5, 6, 7]);
    // Collapsed to a single [0, count) range (index buffer is tiny).
    expect(attr.updateRanges.length).toBe(1);
    expect(attr.updateRanges[0].start).toBe(0);
    expect(attr.updateRanges[0].count).toBe(8);
  });

  it('throws on source arrays shorter than the requested count (fail-loud contract)', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 8);
    const src = makeSource(4); // arrays sized for 4, count says 8
    expect(() => writeSplatTexels(texture, src, 8)).toThrow(/shorter than count/);
  });

  it('clamps the written count to the texture capacity (memory safety)', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 4);
    const cap = elementTexelCapacity(texture, SPLAT_FLOATS_PER_SPLAT);
    const src = makeSource(cap + 5);
    expect(writeSplatTexels(texture, src, cap + 5)).toBe(cap);
  });

  it('fills identity ordering with ONE collapsed prefix update range', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    attachSplatStorage(geometry, 16);
    // Two writes while "hidden" (no flush clears ranges in a unit test):
    // the pending set must stay a single [0, max-end) range, not
    // accumulate (WebGPU backends replay ranges verbatim).
    writeSortedIndexIdentity(geometry, 10);
    writeSortedIndexIdentity(geometry, 6);
    const attr = geometry.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
    expect(attr.updateRanges.length).toBe(1);
    expect(attr.updateRanges[0].start).toBe(0);
    expect(attr.updateRanges[0].count).toBe(10); // union of both writes
    const arr = attr.array as Uint32Array;
    for (let i = 0; i < 10; i++) expect(arr[i]).toBe(i);
  });

  it('writeSortedIndexOrdering REJECTS a truncated or oversized ordering', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    attachSplatStorage(geometry, 8);

    // Fewer indices than drawn elements is not a permutation of the drawn
    // population. Staging it would flip the slot with the tail holding
    // whatever the inactive buffer contained — zeros here, so [3,1] over
    // count 5 would draw element 0 three times and elements 2 and 4 never.
    // Exactly the corruption double-buffering exists to prevent: drop it.
    expect(writeSortedIndexOrdering(geometry, new Uint32Array([3, 1]), 5)).toBe(0);
    expect(hasPendingSortedIndexOrderingApply(geometry)).toBe(false);
    expect(activeSortedIndexSlot(geometry)).toBe(0);

    // An ordering LARGER than the buffers is rejected the same way, never
    // clamped: the retained prefix of a larger permutation is not a
    // permutation of anything — its entries index past the kept range.
    const long = new Uint32Array(32).fill(7);
    expect(writeSortedIndexOrdering(geometry, long, 32)).toBe(0);
    expect(hasPendingSortedIndexOrderingApply(geometry)).toBe(false);
  });

  it('writeSortedIndexOrdering REJECTS a malformed ordering pair instead of repairing it', () => {
    // Repairing here — materialising a missing back buffer, or splitting
    // an aliased one — is the native-WebGPU black-screen bug: it grows
    // the attribute set behind a vertex layout three cached at first draw
    // and never rebuilds. A hand-built geometry must simply never sort.
    const aliased = new THREE.InstancedBufferGeometry();
    const shared = new THREE.InstancedBufferAttribute(new Uint32Array(8), 1);
    aliased.setAttribute('aSortedIndex', shared);
    aliased.setAttribute('aSortedIndexB', shared);
    expect(writeSortedIndexOrdering(aliased, new Uint32Array([7, 6, 5, 4, 3, 2, 1, 0]), 8)).toBe(0);
    expect(aliased.getAttribute('aSortedIndexB')).toBe(shared);

    const missing = new THREE.InstancedBufferGeometry();
    missing.setAttribute('aSortedIndex', new THREE.InstancedBufferAttribute(new Uint32Array(8), 1));
    expect(writeSortedIndexOrdering(missing, new Uint32Array([7, 6, 5, 4, 3, 2, 1, 0]), 8)).toBe(0);
    expect(missing.getAttribute('aSortedIndexB')).toBeUndefined();

    // A SHORTER back buffer would also stall the pump forever: the staged
    // count is clamped against the active buffer but each slice against
    // the inactive one, so `cursor` could never reach `count`.
    const short = new THREE.InstancedBufferGeometry();
    short.setAttribute('aSortedIndex', new THREE.InstancedBufferAttribute(new Uint32Array(8), 1));
    short.setAttribute('aSortedIndexB', new THREE.InstancedBufferAttribute(new Uint32Array(4), 1));
    expect(writeSortedIndexOrdering(short, new Uint32Array([7, 6, 5, 4, 3, 2, 1, 0]), 8)).toBe(0);
    expect(hasPendingSortedIndexOrderingApply(short)).toBe(false);
  });

  it('clamps the attach capacity to the per-node bound (structural safety net)', () => {
    // maxTextureSize 8 → width 8, bound = 8×8/4 = 16 splats — so an
    // over-bound request must yield a texture no taller than 8 rows
    // (height ≤ maxTextureSize by construction) and a matching
    // aSortedIndex length, whatever the caller asked for.
    configureElementTextureLayout(8);
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 100);
    expect(texture.image.height).toBeLessThanOrEqual(8);
    expect((geometry.getAttribute('aSortedIndex').array as Uint32Array).length).toBe(16);
    expect(elementTexelCapacity(texture, SPLAT_FLOATS_PER_SPLAT)).toBe(16);
  });

  it('disposes the texture WITH the geometry (structural lifetime pin)', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 4);
    let disposed = false;
    texture.addEventListener('dispose', () => {
      disposed = true;
    });
    geometry.dispose();
    expect(disposed).toBe(true);
  });
});

describe('pool adapter — growth, dispose, byte accounting', () => {
  let pool: GPUBufferPool;

  beforeEach(() => {
    pool = new GPUBufferPool(20, 300, 5, () => Infinity);
  });

  afterEach(() => {
    pool.dispose();
  });

  it('growth = release + reacquire: FRESH geometry+texture pair, old pair pooled intact', () => {
    const geom1 = pool.acquireGSplatsGeometry('node', 100);
    const tex1 = getSplatTexture(geom1)!;
    // Force growth beyond capacity (100 * 1.5 = 150 < 1000).
    const geom2 = pool.acquireGSplatsGeometry('node', 1000);
    const tex2 = getSplatTexture(geom2)!;

    expect(geom2).not.toBe(geom1);
    expect(tex2).not.toBe(tex1);
    expect(pool.didLastAcquireRebuildAttributes()).toBe(true);
    expect(pool.getStats().capacityGrowths).toBe(1);

    // The OLD pair went back to the free pool INTACT (no dispose, no
    // in-place realloc — the forbidden mechanism): another node can
    // adopt it, texture still attached.
    const geom3 = pool.acquireGSplatsGeometry('other', 100);
    expect(geom3).toBe(geom1);
    expect(getSplatTexture(geom3)).toBe(tex1);
  });

  it('byte-budget eviction frees a pooled gsplat texture (pressure path end-to-end)', () => {
    // Tiny budget: the pooled buffer's texture bytes alone exceed it,
    // so the acquire-triggered sweep must evict and dispose the texture.
    let budget = Infinity;
    const tight = new GPUBufferPool(20, 0, 5, () => budget);
    try {
      const geom = tight.acquireGSplatsGeometry('a', 1000);
      const tex = getSplatTexture(geom)!;
      let disposed = false;
      tex.addEventListener('dispose', () => {
        disposed = true;
      });
      tight.releaseGSplatsGeometry('a');
      budget = 1; // now over budget
      // Advance past the eviction grace and trigger a sweep.
      tight.beginFrame();
      tight.beginFrame();
      tight.evictUnused();
      expect(disposed).toBe(true);
      expect(tight.getStats().pooledBuffers).toBe(0);
    } finally {
      tight.dispose();
    }
  });

  it('pool.dispose() disposes pooled textures through the geometry dispose event', () => {
    const geom = pool.acquireGSplatsGeometry('node', 100);
    const tex = getSplatTexture(geom)!;
    let disposed = false;
    tex.addEventListener('dispose', () => {
      disposed = true;
    });
    pool.releaseGSplatsGeometry('node');
    pool.dispose();
    expect(disposed).toBe(true);
  });

  it('estimateGeometryBytes counts texture + BOTH ordering buffers (≈72 B/splat envelope)', () => {
    const geom = pool.acquireGSplatsGeometry('node', 1000);
    const capacity = (geom.getAttribute('aSortedIndex').array as Uint32Array).length;
    const bytes = estimateGeometryBytes(geom);

    // Envelope, for the headline number: 64 B texture (4 texels × 16 B)
    // + 2 × 4 B for the ordering PAIR.
    const rowBytes = getSplatTextureWidth() * 16;
    expect(bytes).toBeGreaterThanOrEqual(capacity * 72);
    expect(bytes).toBeLessThanOrEqual(capacity * 72 + rowBytes + 256);

    // The envelope ALONE cannot see the ordering buffers: texture rows
    // are padded by up to `rowBytes` (64 KB), which dwarfs 4 B/element,
    // so dropping a whole ordering buffer still lands inside it. Net out
    // the texture's exact byte length and the remainder is pinned.
    const textureBytes = (getSplatTexture(geom)!.image.data as Float32Array).byteLength;
    const nonTexture = bytes - textureBytes;
    expect(nonTexture).toBeGreaterThanOrEqual(capacity * 8);
    // …and nothing beyond the pair except the shared quad + index.
    expect(nonTexture).toBeLessThanOrEqual(capacity * 8 + 256);
  });

  it('releasing a geometry to the pool cancels its in-flight ordering apply', () => {
    // `chunkedApplies` keys its state by GEOMETRY in a strong Map, so an
    // apply left running on a released buffer pins both the geometry and
    // its ordering (4 B/element — ~32 MB for an 8M node) on the free list
    // until it is re-acquired or evicted. Dispose already cancels through
    // the geometry's own listener; release is the other exit from "in
    // use" and needs the same treatment.
    setSortedIndexChunkElementsForTests(2);
    configureSortedIndexChunkedApply(true);
    const geom = pool.acquireGSplatsGeometry('tenant', 32);
    writeSortedIndexIdentity(geom, 16);
    const rev16 = Uint32Array.from({ length: 16 }, (_, i) => 15 - i);
    writeSortedIndexOrdering(geom, rev16, 16);
    pumpSortedIndexOrderingApply(geom); // mid-stream, several slices to go
    expect(hasPendingSortedIndexOrderingApply(geom)).toBe(true);

    pool.releaseGSplatsGeometry('tenant');
    expect(hasPendingSortedIndexOrderingApply(geom)).toBe(false);
    // The DRAWN buffer is untouched by the cancel — an abandoned stream
    // only ever wrote into the inactive one.
    expect(activeSortedIndexSlot(geom)).toBe(0);
    const drawn = getActiveSortedIndexAttribute(geom)!.array as Uint32Array;
    expect(Array.from(drawn.subarray(0, 16))).toEqual([...Array(16).keys()]);
  });

  it('charges BOTH ordering buffers from attach, and sorting adds nothing', () => {
    const geom = pool.acquireGSplatsGeometry('node', 1000);
    const capacity = (geom.getAttribute('aSortedIndex').array as Uint32Array).length;
    const before = estimateGeometryBytes(geom);

    // Both buffers are real and distinct from attach, so the budget must
    // already carry both — under-reporting here would let the pool
    // over-admit nodes and blow the GPU byte budget.
    expect(geom.getAttribute('aSortedIndexB')).not.toBe(geom.getAttribute('aSortedIndex'));
    // Net out the texture: `before` is dominated by it (64 B/element plus
    // up to a 64 KB row pad), so a bare `before >= capacity * 8` would
    // hold even with the back buffer uncounted.
    const textureBytes = (getSplatTexture(geom)!.image.data as Float32Array).byteLength;
    expect(before - textureBytes).toBeGreaterThanOrEqual(capacity * 8);

    // A node's first sort no longer allocates anything, so the estimate
    // must not move (the alias-splitting era grew it by one buffer here).
    writeSortedIndexOrdering(geom, new Uint32Array([3, 2, 1, 0]), 4);
    expect(estimateGeometryBytes(geom)).toBe(before);
  });

  it('clamps acquire capacity AND written count to the per-node texture bound', () => {
    // Shrink the bound so the clamp is testable at unit scale:
    // maxTextureSize 16 -> width 16, bound = 16*16/4 = 64 splats.
    configureElementTextureLayout(16);
    const small = new GPUBufferPool(20, 300, 5, () => Infinity);
    try {
      const geom = small.acquireGSplatsGeometry('big', 1000);
      const texture = getSplatTexture(geom)!;
      expect(elementTexelCapacity(texture, SPLAT_FLOATS_PER_SPLAT)).toBe(64);
      const src = makeSource(64); // writer clamps count to capacity first
      small.updateGSplatsGeometry(
        geom,
        {
          centers3D: src.centers,
          amplitudes: src.amplitudes,
          choleskyFactors: src.choleskyFactors,
          colors: src.colors,
        },
        1000
      );
      // instanceCount mirrors the clamped written count — a
      // bound-clamped node never draws instances without texels.
      expect(geom.instanceCount).toBe(64);
    } finally {
      small.dispose();
    }
  });

  it('updateGeometry writes texels + identity and sets instanceCount', () => {
    const geom = pool.acquireGSplatsGeometry('node', 4);
    const src = makeSource(4);
    pool.updateGSplatsGeometry(
      geom,
      {
        centers3D: src.centers,
        amplitudes: src.amplitudes,
        choleskyFactors: src.choleskyFactors,
        colors: src.colors,
      },
      4
    );
    expect(geom.instanceCount).toBe(4);
    const arr = getSplatTexture(geom)!.image.data as Float32Array;
    expect(arr[SPLAT_FLOATS_PER_SPLAT * 3]).toBe(src.centers[9]); // splat 3 center.x
    expect(geom.boundingBox).not.toBeNull();
  });

  it('a full identity write returns the geometry to slot 0 (pool reuse across tenants)', () => {
    // The slot lives on the GEOMETRY (so it survives release/re-acquire with
    // the buffers it describes) while the selector uniform lives on the
    // MATERIAL. The pool re-pairs them freely: a geometry left on slot 1 by a
    // sorted tenant can be handed to a node whose fresh material defaults to
    // slot 0 — and an order-INDEPENDENT tenant is never tracked by the
    // coordinator, so nothing would ever push the slot to its uniform. It
    // would then draw buffer A: the PREVIOUS tenant's permutation, over a
    // different element count.
    //
    // A full identity write is exactly the "fresh start" signal, so it must
    // re-home the geometry on slot 0. Then a default uniform is always right
    // and no sync is required for untracked nodes.
    const geom = pool.acquireGSplatsGeometry('tenantA', 4);
    const src = makeSource(4);
    pool.updateGSplatsGeometry(
      geom,
      {
        centers3D: src.centers,
        amplitudes: src.amplitudes,
        choleskyFactors: src.choleskyFactors,
        colors: src.colors,
      },
      4
    );
    writeSortedIndexOrdering(geom, new Uint32Array([3, 2, 1, 0]), 4);
    while (pumpSortedIndexOrderingApply(geom).more) {
      /* drain: flips to slot 1 */
    }
    expect(activeSortedIndexSlot(geom)).toBe(1);

    // A fresh full commit (the !preserveOrdering path every new tenant takes).
    pool.updateGSplatsGeometry(
      geom,
      {
        centers3D: src.centers,
        amplitudes: src.amplitudes,
        choleskyFactors: src.choleskyFactors,
        colors: src.colors,
      },
      4
    );
    expect(activeSortedIndexSlot(geom)).toBe(0);
    // ...and slot 0 — what a default-uniform material reads — holds identity.
    const front = geom.getAttribute('aSortedIndex').array as Uint32Array;
    expect(Array.from(front.subarray(0, 4))).toEqual([0, 1, 2, 3]);
  });

  it('preserveOrdering keeps the sort permutation while still rewriting texels', () => {
    const geom = pool.acquireGSplatsGeometry('node', 4);
    const src = makeSource(4);
    const packed = (amplitudes: Float32Array) => ({
      centers3D: src.centers,
      amplitudes,
      choleskyFactors: src.choleskyFactors,
      colors: src.colors,
    });
    pool.updateGSplatsGeometry(geom, packed(src.amplitudes), 4);
    // The SortWorker landed a depth-sort permutation between commits.
    writeSortedIndexOrdering(geom, new Uint32Array([3, 2, 1, 0]), 4);
    while (pumpSortedIndexOrderingApply(geom).more) {
      /* drain: the ordering swaps in on completion */
    }

    // Same-count recommit with preserveOrdering: permutation intact,
    // texels + instanceCount + bounds refreshed as usual.
    const newAmplitudes = new Float32Array([9, 8, 7, 6]);
    pool.updateGSplatsGeometry(geom, packed(newAmplitudes), 4, 3.0, { preserveOrdering: true });
    const ordering = getActiveSortedIndexAttribute(geom)!.array as Uint32Array;
    expect(Array.from(ordering.subarray(0, 4))).toEqual([3, 2, 1, 0]);
    const texels = getSplatTexture(geom)!.image.data as Float32Array;
    expect(texels[3]).toBe(9); // splat 0 amplitude — texels WERE rewritten
    expect(geom.instanceCount).toBe(4);

    // Without the flag the identity reset is restored (default behavior).
    // Re-resolve the active attribute: a full identity write also re-homes
    // the geometry on slot 0, so the pre-reset reference is stale by design.
    pool.updateGSplatsGeometry(geom, packed(src.amplitudes), 4);
    expect(activeSortedIndexSlot(geom)).toBe(0);
    const reset = getActiveSortedIndexAttribute(geom)!.array as Uint32Array;
    expect(Array.from(reset.subarray(0, 4))).toEqual([0, 1, 2, 3]);
  });

  it('fromInstance append: writes only the suffix texels, extends aSortedIndex, keeps the prefix permutation', () => {
    const geom = pool.acquireGSplatsGeometry('node', 16);
    const src6 = makeSource(6);
    const packed = (s: SplatTexelSource, count: number) => ({
      centers3D: s.centers.subarray(0, count * 3),
      amplitudes: s.amplitudes.subarray(0, count),
      choleskyFactors: s.choleskyFactors.subarray(0, count * 6),
      colors: s.colors.subarray(0, count * 3),
    });
    // Prefix commit of 4 splats, then a real permutation lands on it.
    pool.updateGSplatsGeometry(geom, packed(src6, 4), 4);
    writeSortedIndexOrdering(geom, new Uint32Array([3, 2, 1, 0]), 4);
    while (pumpSortedIndexOrderingApply(geom).more) {
      /* drain: the ordering swaps in on completion */
    }
    const texels = getSplatTexture(geom)!.image.data as Float32Array;
    const sentinel = -999;
    texels[0] = sentinel; // splat 0 center.x — must survive the append

    // Append to 6 splats: fromInstance = 4 → only [4,6) rewritten.
    pool.updateGSplatsGeometry(geom, packed(src6, 6), 6, 3.0, { fromInstance: 4 });
    expect(geom.instanceCount).toBe(6);
    expect(texels[0]).toBe(sentinel); // prefix texels untouched
    // Suffix splat 5 center.x written.
    expect(texels[5 * SPLAT_FLOATS_PER_SPLAT]).toBe(src6.centers[15]);
    // Prefix permutation preserved; suffix gets identity.
    const ordering = getActiveSortedIndexAttribute(geom)!.array as Uint32Array;
    expect(Array.from(ordering.subarray(0, 6))).toEqual([3, 2, 1, 0, 4, 5]);
  });
});

describe('per-node gsplat materials — manager registration lifecycle', () => {
  it('creates DISTINCT materials per call (LRU bypass) and unregisters on dispose', () => {
    // Spec §4 exit criterion: material-disposal leak check. Per-node
    // materials are only safe if dispose() removes them from the
    // manager's registered set (camera-broadcast registry) — otherwise
    // every node teardown leaks a strongly-held material.
    const props = {
      blendingMode: 'additive',
      opacity: 1.0,
      gamma: 1.0,
      intensity: 1.0,
      offset: 0.0,
    } as Parameters<typeof materialManager.getGSplatMaterial>[0];
    const before = materialManager.getCacheStats().totalRegistered;
    const a = materialManager.getGSplatMaterial(props);
    const b = materialManager.getGSplatMaterial(props);
    expect(a).not.toBe(b); // same props, still per-node
    expect(materialManager.getCacheStats().totalRegistered).toBe(before + 2);
    a.dispose();
    expect(materialManager.getCacheStats().totalRegistered).toBe(before + 1);
    b.dispose();
    expect(materialManager.getCacheStats().totalRegistered).toBe(before);
  });
});

describe('syncGSplatMaterialWithGeometry — commit material rebind', () => {
  it('rebinds uSplatTex on the render AND pick materials', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 4);
    const renderMat = new GSplatMaterial();
    const pickMat = new GSplatPickingMaterial({ nodeId: 7 });
    const mesh = new THREE.Mesh(geometry, renderMat);
    const pickNode = new THREE.Mesh(geometry, pickMat);
    mesh.userData.pickNode = pickNode;

    syncGSplatMaterialWithGeometry(mesh);
    expect(renderMat.uniforms.uSplatTex.value).toBe(texture);
    expect(pickMat.uniforms.uSplatTex.value).toBe(texture);
  });

  it('no-ops on a geometry without splat storage (points/lines meshes)', () => {
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new GSplatMaterial());
    expect(() => syncGSplatMaterialWithGeometry(mesh)).not.toThrow();
  });
});

/**
 * Deterministic PRNG for randomized-but-reproducible fixtures below
 * (mulberry32, same generator the synthetic-scene module uses).
 */
function prng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Random splat set with NEGATIVE coordinates and ANISOTROPIC Cholesky
 * factors (large axis spreads + signed off-diagonals) — the stress case
 * for both the texel-byte identity and the fused-bounds tests.
 */
function makeRandomSource(count: number, seed = 42) {
  const rand = prng(seed);
  const centers = new Float32Array(count * 3);
  const choleskyFactors = new Float32Array(count * 6);
  const amplitudes = new Float32Array(count);
  const colors = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    centers.set([rand() * 200 - 100, rand() * 200 - 100, rand() * 200 - 100], i * 3);
    // Anisotropic: diagonals spread over ~3 decades, signed off-diagonals.
    const d0 = Math.pow(10, rand() * 3 - 1.5);
    const d1 = Math.pow(10, rand() * 3 - 1.5);
    const d2 = Math.pow(10, rand() * 3 - 1.5);
    choleskyFactors.set(
      [d0, (rand() * 2 - 1) * d1, d1, (rand() * 2 - 1) * d2, (rand() * 2 - 1) * d2, d2],
      i * 6
    );
    amplitudes[i] = rand();
    colors.set([rand(), rand(), rand(), rand()], i * 4);
  }
  return { centers, choleskyFactors, amplitudes, colors, colorComponents: 4 as const };
}

describe('writeSplatTexels — texel-byte identity vs the retired packed-triple path', () => {
  /**
   * Test-local reference implementation of the RETIRED pipeline:
   * `packCholeskyForShader` split the 6-stride factors into three
   * [x, y] attribute pairs, and the texel writer re-interleaved them.
   * Kept verbatim here so the direct 6-stride path can be byte-compared
   * against exactly what the old code produced.
   */
  function writeSplatTexelsPackedReference(
    arr: Float32Array,
    src: ReturnType<typeof makeRandomSource>,
    count: number
  ): void {
    // packCholeskyForShader (retired):
    const cholesky01 = new Float32Array(count * 2);
    const cholesky23 = new Float32Array(count * 2);
    const cholesky45 = new Float32Array(count * 2);
    for (let i = 0; i < count; i++) {
      const srcOffset = i * 6;
      const dstOffset = i * 2;
      cholesky01[dstOffset] = src.choleskyFactors[srcOffset];
      cholesky01[dstOffset + 1] = src.choleskyFactors[srcOffset + 1];
      cholesky23[dstOffset] = src.choleskyFactors[srcOffset + 2];
      cholesky23[dstOffset + 1] = src.choleskyFactors[srcOffset + 3];
      cholesky45[dstOffset] = src.choleskyFactors[srcOffset + 4];
      cholesky45[dstOffset + 1] = src.choleskyFactors[srcOffset + 5];
    }
    // Re-interleaving texel writer (retired shape):
    const colorK = src.colorComponents ?? 3;
    for (let i = 0; i < count; i++) {
      const o = i * SPLAT_FLOATS_PER_SPLAT;
      const p3 = i * 3;
      const ck = i * colorK;
      const c2 = i * 2;
      arr[o] = src.centers[p3];
      arr[o + 1] = src.centers[p3 + 1];
      arr[o + 2] = src.centers[p3 + 2];
      arr[o + 3] = src.amplitudes[i];
      arr[o + 4] = cholesky01[c2];
      arr[o + 5] = cholesky01[c2 + 1];
      arr[o + 6] = cholesky23[c2];
      arr[o + 7] = cholesky23[c2 + 1];
      arr[o + 8] = cholesky45[c2];
      arr[o + 9] = cholesky45[c2 + 1];
      arr[o + 10] = src.colors[ck];
      arr[o + 11] = src.colors[ck + 1];
      arr[o + 12] = src.colors[ck + 2];
      arr[o + 13] = colorK === 4 ? src.colors[ck + 3] : 1.0;
    }
  }

  it('produces byte-identical texture content on a randomized anisotropic splat set', () => {
    const count = 64;
    const src = makeRandomSource(count, 1234);

    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, count);
    const written = writeSplatTexels(texture, src, count);
    expect(written).toBe(count);
    const direct = texture.image.data as Float32Array;

    const reference = new Float32Array(count * SPLAT_FLOATS_PER_SPLAT);
    writeSplatTexelsPackedReference(reference, src, count);

    // BYTE identity over every written float (bit-exact, not toBeCloseTo).
    const directBytes = new Uint8Array(
      direct.buffer,
      direct.byteOffset,
      count * SPLAT_FLOATS_PER_SPLAT * 4
    );
    const referenceBytes = new Uint8Array(reference.buffer, 0, reference.byteLength);
    expect(directBytes).toEqual(referenceBytes);
  });
});

describe('pool adapter — precomputed projection bounds fast path', () => {
  it('bounds-present and scan-fallback commits produce identical cull bounds', async () => {
    const { computeGSplatsProjectionBounds } =
      await import('../../../workers/data-worker/projection/gsplats');
    const count = 128;
    const src = makeRandomSource(count, 777);
    const pool = new GPUBufferPool(20, 300, 5, () => Infinity);
    try {
      const payload = {
        centers3D: src.centers,
        amplitudes: src.amplitudes,
        choleskyFactors: src.choleskyFactors,
        colors: src.colors,
        colorComponents: src.colorComponents,
      };

      // Fallback: no bounds metadata → the adapter scans.
      const scanGeom = pool.acquireGSplatsGeometry('scan', count);
      pool.updateGSplatsGeometry(scanGeom, payload, count, 2.5);
      expect(scanGeom.boundingBox).not.toBeNull();

      // Fast path: fused-scan metadata supplied → scans skipped.
      const fastGeom = pool.acquireGSplatsGeometry('fast', count);
      pool.updateGSplatsGeometry(
        fastGeom,
        {
          ...payload,
          bounds: computeGSplatsProjectionBounds(src.centers, src.choleskyFactors, count),
        },
        count,
        2.5
      );

      // Bit-exact equality (same float ops in the fused scan).
      expect(fastGeom.boundingBox!.min.toArray()).toEqual(scanGeom.boundingBox!.min.toArray());
      expect(fastGeom.boundingBox!.max.toArray()).toEqual(scanGeom.boundingBox!.max.toArray());
      expect(fastGeom.boundingSphere!.center.toArray()).toEqual(
        scanGeom.boundingSphere!.center.toArray()
      );
      expect(fastGeom.boundingSphere!.radius).toBe(scanGeom.boundingSphere!.radius);
    } finally {
      pool.dispose();
    }
  });
});

describe('double-buffered ordering apply (atomic swap)', () => {
  // Tiny slice (4 indices) so tests stay readable; the production
  // constant is 1M (4 MB/frame — see the element-storage module note).
  const CHUNK = 4;

  beforeEach(() => {
    setSortedIndexChunkElementsForTests(CHUNK);
    configureSortedIndexChunkedApply(true);
  });

  afterEach(() => {
    cancelAllSortedIndexOrderingApplies();
    setSortedIndexChunkElementsForTests(null);
    configureSortedIndexChunkedApply(true);
  });

  function makeGeometry(capacity: number): { geometry: THREE.InstancedBufferGeometry } {
    const geometry = new THREE.InstancedBufferGeometry();
    attachSplatStorage(geometry, capacity);
    return { geometry };
  }

  /** The attribute the shaders are currently reading. */
  function activeAttr(geometry: THREE.InstancedBufferGeometry): THREE.InstancedBufferAttribute {
    return getActiveSortedIndexAttribute(geometry) as THREE.InstancedBufferAttribute;
  }
  function activeArr(geometry: THREE.InstancedBufferGeometry): Uint32Array {
    return activeAttr(geometry).array as Uint32Array;
  }

  /** Reversed permutation over [0, n) — every entry differs from identity (n ≥ 2). */
  function reversed(n: number): Uint32Array {
    const out = new Uint32Array(n);
    for (let i = 0; i < n; i++) out[i] = n - 1 - i;
    return out;
  }

  /** Simulate the classic WebGLRenderer consuming the pending upload. */
  function flushAttr(attr: THREE.InstancedBufferAttribute): void {
    attr.clearUpdateRanges();
  }

  /**
   * THE invariant this whole design exists for: what is on screen is a
   * bijection of [0, n) onto itself. A duplicate means one element draws
   * twice and the element it displaced draws not at all.
   */
  function expectPermutation(geometry: THREE.InstancedBufferGeometry, n: number): void {
    const arr = activeArr(geometry);
    const seen = new Set<number>();
    for (let i = 0; i < n; i++) {
      const v = arr[i];
      expect(v, `index ${i} out of range`).toBeLessThan(n);
      expect(seen.has(v), `index ${v} drawn twice (slot ${i})`).toBe(false);
      seen.add(v);
    }
    expect(seen.size).toBe(n);
  }

  it('exposes a 1M-index (4 MB) production slice size', () => {
    expect(SORTED_INDEX_CHUNK_ELEMENTS).toBe(1_000_000);
  });

  it('keeps the ordering attribute OBJECTS fixed for the geometry lifetime', () => {
    // REGRESSION GUARD, native WebGPU. Three keys a pipeline's vertex
    // buffer layout by BufferAttribute IDENTITY but hashes only attribute
    // NAMES into the geometry cache key, and answers `needsGeometryUpdate`
    // with a bare `setGeometry()` that never rebuilds the pipeline. So
    // swapping in a new attribute object after first render — as the
    // alias-splitting revision of this module did on a node's first sort —
    // leaves the draw binding N+1 vertex buffers into an N-buffer layout:
    // every later attribute shifts a slot, the quad-corner attribute reads
    // the ordering buffer's u32s as vec2<f32>, and the scene goes BLACK
    // with no validation error. WebGL binds by program location and is
    // immune, as is the `forceWebGL` TSL-parity harness, so this unit
    // invariant is the only cheap guard we have.
    const { geometry } = makeGeometry(16);
    const a = geometry.getAttribute('aSortedIndex');
    const b = geometry.getAttribute('aSortedIndexB');
    expect(b).not.toBe(a);
    // Equal length is load-bearing: three derives _maxInstanceCount from
    // the SMALLEST instanced attribute, so a short back buffer would
    // silently clamp the draw.
    expect((b.array as Uint32Array).length).toBe((a.array as Uint32Array).length);

    // Drive a full cycle through BOTH slots — identity write, a sort that
    // flips to B, and a second sort that flips back to A.
    writeSortedIndexIdentity(geometry, 12);
    for (const ordering of [reversed(12), reversed(12)]) {
      writeSortedIndexOrdering(geometry, ordering, 12);
      while (pumpSortedIndexOrderingApply(geometry).more) {
        /* drain */
      }
    }
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    expect(geometry.getAttribute('aSortedIndex')).toBe(a);
    expect(geometry.getAttribute('aSortedIndexB')).toBe(b);
  });

  it('staging leaves the DRAWN buffer untouched (no ranges, no version bump)', () => {
    const { geometry } = makeGeometry(16);
    writeSortedIndexIdentity(geometry, 12);
    const attr = activeAttr(geometry);
    flushAttr(attr);
    const versionBefore = attr.version;

    const n = writeSortedIndexOrdering(geometry, reversed(12), 12);
    expect(n).toBe(12);
    expect(hasPendingSortedIndexOrderingApply(geometry)).toBe(true);
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    expect(Array.from(activeArr(geometry).subarray(0, 12))).toEqual([
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    ]);
    expect(attr.updateRanges.length).toBe(0);
    expect(attr.version).toBe(versionBefore);
  });

  it('the DRAWN buffer is a whole permutation after EVERY pump, and flips only on the last slice', () => {
    // The regression guard. Pre-double-buffering this failed on the very
    // first pump: the live attribute held new[0,4) ∪ old[4,12), which is
    // not a permutation (index 8 sat at slots 3 and 8 — one splat drawn
    // twice, indices 0-3 omitted).
    const { geometry } = makeGeometry(16);
    writeSortedIndexIdentity(geometry, 12);
    const ordering = reversed(12); // 3 slices of 4
    writeSortedIndexOrdering(geometry, ordering, 12);

    // Slices 1 and 2: still streaming, slot unchanged, screen unchanged.
    for (let slice = 1; slice <= 2; slice++) {
      const result = pumpSortedIndexOrderingApply(geometry);
      expect(result).toEqual({ more: true, flipped: false });
      expect(activeSortedIndexSlot(geometry)).toBe(0);
      expectPermutation(geometry, 12);
      expect(Array.from(activeArr(geometry).subarray(0, 12))).toEqual([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
      ]);
    }

    // Slice 3 completes the back buffer → the swap.
    expect(pumpSortedIndexOrderingApply(geometry)).toEqual({ more: false, flipped: true });
    expect(activeSortedIndexSlot(geometry)).toBe(1);
    expectPermutation(geometry, 12);
    expect(Array.from(activeArr(geometry).subarray(0, 12))).toEqual(Array.from(ordering));
    expect(hasPendingSortedIndexOrderingApply(geometry)).toBe(false);
    // Idempotent past completion.
    expect(pumpSortedIndexOrderingApply(geometry)).toEqual({ more: false, flipped: false });
  });

  it('registers per-slice ranges on the buffer being written, not the whole prefix', () => {
    const { geometry } = makeGeometry(16);
    writeSortedIndexOrdering(geometry, reversed(12), 12);
    // The target is the INACTIVE attribute — re-registering [0, k·CHUNK)
    // every frame would upload 4+8+12 MB instead of 4 MB/frame.
    const back = geometry.getAttribute('aSortedIndexB') as THREE.InstancedBufferAttribute;
    pumpSortedIndexOrderingApply(geometry);
    expect(back.updateRanges[0]).toMatchObject({ start: 0, count: CHUNK });
    flushAttr(back);
    pumpSortedIndexOrderingApply(geometry);
    expect(back.updateRanges[0]).toMatchObject({ start: CHUNK, count: CHUNK });
  });

  it('unflushed slices collapse into ONE contiguous range (WebGPU never-clears discipline)', () => {
    const { geometry } = makeGeometry(16);
    writeSortedIndexOrdering(geometry, reversed(12), 12);
    const back = geometry.getAttribute('aSortedIndexB') as THREE.InstancedBufferAttribute;
    // No flush between slices (hidden mesh / coalesced frames): ranges
    // must fold, never accumulate.
    pumpSortedIndexOrderingApply(geometry);
    expect(back.updateRanges.length).toBe(1);
    expect(back.updateRanges[0]).toMatchObject({ start: 0, count: CHUNK });
    pumpSortedIndexOrderingApply(geometry);
    expect(back.updateRanges.length).toBe(1);
    expect(back.updateRanges[0]).toMatchObject({ start: 0, count: 2 * CHUNK });
    pumpSortedIndexOrderingApply(geometry);
    expect(back.updateRanges.length).toBe(1);
    expect(back.updateRanges[0]).toMatchObject({ start: 0, count: 3 * CHUNK });
  });

  it('successive orderings ping-pong the slot back to 0', () => {
    const { geometry } = makeGeometry(16);
    writeSortedIndexIdentity(geometry, 12);
    const a = reversed(12);
    writeSortedIndexOrdering(geometry, a, 12);
    while (pumpSortedIndexOrderingApply(geometry).more) {
      /* drain */
    }
    expect(activeSortedIndexSlot(geometry)).toBe(1);

    const b = new Uint32Array([5, 4, 7, 6, 1, 0, 3, 2, 9, 8, 11, 10]);
    writeSortedIndexOrdering(geometry, b, 12);
    while (pumpSortedIndexOrderingApply(geometry).more) {
      /* drain */
    }
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    expect(Array.from(activeArr(geometry).subarray(0, 12))).toEqual(Array.from(b));
    expectPermutation(geometry, 12);
  });

  it('a NEW ordering mid-apply is HELD and starts only after the current one completes', () => {
    // Never restart a streaming apply: under a continuous orbit new
    // orderings arrive every sort round-trip, and restart-from-slice-0
    // meant the stream never converged (measured 8× frame-median
    // regression). Finishing, flipping, then starting the newest always
    // converges — and every intermediate frame shows a whole permutation.
    const { geometry } = makeGeometry(16);
    const orderingA = reversed(12);
    writeSortedIndexOrdering(geometry, orderingA, 12);
    pumpSortedIndexOrderingApply(geometry);
    pumpSortedIndexOrderingApply(geometry); // A written through [0, 8)

    const orderingB = new Uint32Array([5, 4, 7, 6, 1, 0, 3, 2, 9, 8, 11, 10]);
    writeSortedIndexOrdering(geometry, orderingB, 12);
    expect(hasPendingSortedIndexOrderingApply(geometry)).toBe(true);

    // A completes and swaps in; B's stream starts into the old front.
    expect(pumpSortedIndexOrderingApply(geometry)).toEqual({ more: true, flipped: true });
    expect(Array.from(activeArr(geometry).subarray(0, 12))).toEqual(Array.from(orderingA));
    expectPermutation(geometry, 12);

    expect(pumpSortedIndexOrderingApply(geometry)).toEqual({ more: true, flipped: false });
    expectPermutation(geometry, 12); // still showing A while B streams
    pumpSortedIndexOrderingApply(geometry);
    expect(pumpSortedIndexOrderingApply(geometry)).toEqual({ more: false, flipped: true });
    expect(Array.from(activeArr(geometry).subarray(0, 12))).toEqual(Array.from(orderingB));
    expectPermutation(geometry, 12);
  });

  it('held ordering: LATEST wins — an even newer arrival replaces the held one', () => {
    const { geometry } = makeGeometry(16);
    writeSortedIndexOrdering(geometry, reversed(12), 12);
    pumpSortedIndexOrderingApply(geometry); // A streaming

    const orderingB = new Uint32Array(12).fill(1);
    const orderingC = new Uint32Array([5, 4, 7, 6, 1, 0, 3, 2, 9, 8, 11, 10]);
    writeSortedIndexOrdering(geometry, orderingB, 12); // held...
    writeSortedIndexOrdering(geometry, orderingC, 12); // ...replaced (B dropped)

    for (let i = 0; i < 8 && pumpSortedIndexOrderingApply(geometry).more; i++) {
      /* drain A then C */
    }
    expect(Array.from(activeArr(geometry).subarray(0, 12))).toEqual(Array.from(orderingC));
    expectPermutation(geometry, 12);
  });

  it('an ordering that fits ONE slice completes and flips on the first pump', () => {
    const { geometry } = makeGeometry(16);
    const n = writeSortedIndexOrdering(geometry, reversed(CHUNK), CHUNK);
    expect(n).toBe(CHUNK);
    expect(pumpSortedIndexOrderingApply(geometry)).toEqual({ more: false, flipped: true });
    expect(Array.from(activeArr(geometry).subarray(0, CHUNK))).toEqual([3, 2, 1, 0]);
  });

  it('fires onApplied only after the selected ordering completes a draw, not at the flip', () => {
    const { geometry } = makeGeometry(16);
    let applied = 0;
    let abandoned = 0;
    writeSortedIndexOrdering(geometry, reversed(CHUNK), CHUNK, {
      onApplied: () => applied++,
      onAbandoned: () => abandoned++,
    });

    // The pump copies the final slice and selects the buffer for the next
    // render, but THREE has not consumed the update range or drawn it yet.
    expect(pumpSortedIndexOrderingApply(geometry)).toEqual({ more: false, flipped: true });
    expect(applied).toBe(0);
    expect(abandoned).toBe(0);

    acknowledgeSortedIndexOrderingDraw(geometry);
    expect(applied).toBe(1);
    expect(abandoned).toBe(0);
    // Exactly once even if a backend/render path invokes the hook again.
    acknowledgeSortedIndexOrderingDraw(geometry);
    expect(applied).toBe(1);
  });

  it('abandons a selected ordering if a newer flip supersedes it before any draw', () => {
    const { geometry } = makeGeometry(16);
    const events: string[] = [];
    writeSortedIndexOrdering(geometry, reversed(CHUNK), CHUNK, {
      onApplied: () => events.push('A applied'),
      onAbandoned: () => events.push('A abandoned'),
    });
    pumpSortedIndexOrderingApply(geometry); // A selected, not rendered

    writeSortedIndexOrdering(geometry, new Uint32Array([1, 0, 3, 2]), CHUNK, {
      onApplied: () => events.push('B applied'),
      onAbandoned: () => events.push('B abandoned'),
    });
    pumpSortedIndexOrderingApply(geometry); // B supersedes un-rendered A
    expect(events).toEqual(['A abandoned']);

    acknowledgeSortedIndexOrderingDraw(geometry);
    expect(events).toEqual(['A abandoned', 'B applied']);
  });

  it('cancelling after a flip abandons the selected-but-unrendered ordering', () => {
    const { geometry } = makeGeometry(16);
    const events: string[] = [];
    writeSortedIndexOrdering(geometry, reversed(CHUNK), CHUNK, {
      onApplied: () => events.push('applied'),
      onAbandoned: () => events.push('abandoned'),
    });
    pumpSortedIndexOrderingApply(geometry);

    cancelSortedIndexOrderingApply(geometry);
    expect(events).toEqual(['abandoned']);
    acknowledgeSortedIndexOrderingDraw(geometry);
    expect(events).toEqual(['abandoned']);
  });

  it('an EMPTY ordering stages nothing — it must not flip away from a good order', () => {
    // A zero-length ordering would "complete" on its first pump and FLIP,
    // swapping the newest ordering out for the older buffer behind it.
    const { geometry } = makeGeometry(16);
    writeSortedIndexIdentity(geometry, 8);
    const good = reversed(8);
    writeSortedIndexOrdering(geometry, good, 8);
    while (pumpSortedIndexOrderingApply(geometry).more) {
      /* drain */
    }
    const slot = activeSortedIndexSlot(geometry);
    const shown = Array.from(activeArr(geometry).subarray(0, 8));
    expect(shown).toEqual(Array.from(good));

    expect(writeSortedIndexOrdering(geometry, new Uint32Array(0), 0)).toBe(0);
    expect(hasPendingSortedIndexOrderingApply(geometry)).toBe(false);
    expect(pumpSortedIndexOrderingApply(geometry)).toEqual({ more: false, flipped: false });
    expect(activeSortedIndexSlot(geometry)).toBe(slot);
    expect(Array.from(activeArr(geometry).subarray(0, 8))).toEqual(shown);
  });

  it('a NEGATIVE, NaN, or FRACTIONAL count is rejected like an empty one', () => {
    // Same failure mode as the empty ordering above, and why the guard
    // demands a positive INTEGER: a negative or NaN count writes nothing
    // on the first pump while `cursor >= count` is already satisfied — so
    // it FLIPS, publishing whatever stale content the inactive buffer
    // held as the new ordering — and a fractional count flips with only
    // `floor(count)` entries written (`subarray` truncates while the
    // cursor reaches `count` exactly). Unreachable from the coordinator
    // (it only ever passes `ordering.length`), but this writer's whole
    // contract is that a half- or un-written buffer is never swapped in.
    for (const bad of [-5, Number.NaN, 6.5]) {
      const { geometry } = makeGeometry(16);
      writeSortedIndexIdentity(geometry, 8);
      const good = reversed(8);
      writeSortedIndexOrdering(geometry, good, 8);
      while (pumpSortedIndexOrderingApply(geometry).more) {
        /* drain */
      }
      const slot = activeSortedIndexSlot(geometry);
      const shown = Array.from(activeArr(geometry).subarray(0, 8));

      expect(writeSortedIndexOrdering(geometry, reversed(8), bad), `count=${bad}`).toBe(0);
      expect(hasPendingSortedIndexOrderingApply(geometry)).toBe(false);
      expect(pumpSortedIndexOrderingApply(geometry)).toEqual({ more: false, flipped: false });
      expect(activeSortedIndexSlot(geometry), `count=${bad} flipped the slot`).toBe(slot);
      expect(Array.from(activeArr(geometry).subarray(0, 8))).toEqual(shown);
    }
  });

  it('writeSortedIndexIdentity cancels an in-flight apply AND its held ordering (commit supersedes)', () => {
    const { geometry } = makeGeometry(16);
    writeSortedIndexOrdering(geometry, reversed(12), 12);
    pumpSortedIndexOrderingApply(geometry); // streaming
    writeSortedIndexOrdering(geometry, new Uint32Array(12).fill(2), 12); // held
    expect(hasPendingSortedIndexOrderingApply(geometry)).toBe(true);
    writeSortedIndexIdentity(geometry, 12);
    expect(hasPendingSortedIndexOrderingApply(geometry)).toBe(false);
    expect(pumpSortedIndexOrderingApply(geometry)).toEqual({ more: false, flipped: false });
    // Neither the stale stream nor the held ordering scribbles over the
    // fresh identity, and the slot never moved.
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    expect(Array.from(activeArr(geometry).subarray(0, 12))).toEqual([
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    ]);
  });

  it('writeSortedIndexIdentityRange (append commit) cancels an in-flight apply', () => {
    const { geometry } = makeGeometry(16);
    writeSortedIndexOrdering(geometry, reversed(12), 12);
    writeSortedIndexIdentityRange(geometry, 12, 16);
    expect(hasPendingSortedIndexOrderingApply(geometry)).toBe(false);
    expect(pumpSortedIndexOrderingApply(geometry)).toEqual({ more: false, flipped: false });
  });

  it('identity writers target whichever buffer is live after a flip', () => {
    const { geometry } = makeGeometry(16);
    writeSortedIndexOrdering(geometry, reversed(12), 12);
    while (pumpSortedIndexOrderingApply(geometry).more) {
      /* drain */
    }
    expect(activeSortedIndexSlot(geometry)).toBe(1);
    // A commit landing while slot 1 is live must reset THAT buffer.
    writeSortedIndexIdentity(geometry, 12);
    expect(Array.from(activeArr(geometry).subarray(0, 12))).toEqual([
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    ]);
  });

  it('geometry dispose cancels the apply (structural lifetime pin)', () => {
    const { geometry } = makeGeometry(16);
    writeSortedIndexOrdering(geometry, reversed(12), 12);
    expect(hasPendingSortedIndexOrderingApply(geometry)).toBe(true);
    geometry.dispose();
    expect(hasPendingSortedIndexOrderingApply(geometry)).toBe(false);
  });

  it('explicit cancel + cancel-all clear pending applies, leaving the drawn order valid', () => {
    const a = makeGeometry(16);
    const b = makeGeometry(16);
    writeSortedIndexIdentity(a.geometry, 12);
    writeSortedIndexOrdering(a.geometry, reversed(12), 12);
    writeSortedIndexOrdering(b.geometry, reversed(12), 12);
    pumpSortedIndexOrderingApply(a.geometry); // mid-stream
    cancelSortedIndexOrderingApply(a.geometry);
    expect(hasPendingSortedIndexOrderingApply(a.geometry)).toBe(false);
    // Abandoning mid-stream discards an un-drawn buffer — what is on
    // screen is untouched and still whole.
    expectPermutation(a.geometry, 12);
    expect(hasPendingSortedIndexOrderingApply(b.geometry)).toBe(true);
    cancelAllSortedIndexOrderingApplies();
    expect(hasPendingSortedIndexOrderingApply(b.geometry)).toBe(false);
  });

  it('slicing disabled (WebGPU backends): one slice, then the same atomic flip', () => {
    // The WebGPU backends ignore attribute ranges and re-upload the whole
    // buffer per flush, so slicing there would multiply uploads. They
    // still double-buffer — the swap is correctness, not an optimisation.
    configureSortedIndexChunkedApply(false);
    const { geometry } = makeGeometry(16);
    const ordering = reversed(12);
    const n = writeSortedIndexOrdering(geometry, ordering, 12);
    expect(n).toBe(12);
    expect(pumpSortedIndexOrderingApply(geometry)).toEqual({ more: false, flipped: true });
    expect(activeSortedIndexSlot(geometry)).toBe(1);
    expect(Array.from(activeArr(geometry).subarray(0, 12))).toEqual(Array.from(ordering));
    const back = activeAttr(geometry);
    expect(back.updateRanges.length).toBe(1);
    expect(back.updateRanges[0]).toMatchObject({ start: 0, count: 12 });
  });

  it('rejects an ordering larger than the attribute capacity (never clamps to a prefix)', () => {
    const { geometry } = makeGeometry(8); // attr length 8
    writeSortedIndexIdentity(geometry, 8);
    const shown = Array.from(activeArr(geometry).subarray(0, 8));
    // reversed(12) clamped to 8 entries would publish [11,10,9,8,7,6,5,4]
    // — every entry indexing past the drawn population. Such an ordering
    // belongs to a population this geometry cannot hold (capacity >=
    // instanceCount), so it is dropped whole, keeping the shown order.
    expect(writeSortedIndexOrdering(geometry, reversed(12), 12)).toBe(0);
    expect(hasPendingSortedIndexOrderingApply(geometry)).toBe(false);
    expect(pumpSortedIndexOrderingApply(geometry)).toEqual({ more: false, flipped: false });
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    expect(Array.from(activeArr(geometry).subarray(0, 8))).toEqual(shown);
  });
});

/**
 * The whole point of double-buffering, stated as ONE property and tested
 * against randomized interleavings rather than scripted ones: whatever
 * the shaders would read at any instant is a COMPLETE permutation of the
 * live population — every storage slot drawn exactly once, none out of
 * range.
 *
 * The scripted tests above each pin one transition. This pins the
 * property across the whole reachable state space, including the
 * combinations nobody thought to script (a resize landing between two
 * slices, an append while an ordering is held, a cancel one pump before
 * completion). That mix is what the shipped-then-reverted chunked apply
 * got wrong: it satisfied every scripted case and still drew a corrupt
 * permutation 70-80% of frames under a continuous orbit.
 *
 * Deterministic (seeded LCG), so a failure reproduces exactly from the
 * printed seed/step/op.
 */
describe('ordering invariant — randomized interleavings', () => {
  function prng(seed: number): () => number {
    let s = seed >>> 0;
    return () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32;
  }
  function shuffled(n: number, r: () => number): Uint32Array {
    const a = Uint32Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
  /** null when `a[0, n)` is a permutation of `[0, n)`, else why not. */
  function permError(a: Uint32Array, n: number): string | null {
    const seen = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const v = a[i];
      if (v >= n) return `slot ${i} -> ${v} out of range (n=${n})`;
      if (seen[v]) return `storage ${v} drawn twice (slot ${i})`;
      seen[v] = 1;
    }
    return null;
  }

  it('the DRAWN ordering is a whole permutation after every operation', () => {
    configureSortedIndexChunkedApply(true);
    const opTally: Record<string, number> = {};
    let checks = 0;

    for (let seed = 1; seed <= 400; seed++) {
      const r = prng(seed);
      // Small chunks so a stream spans many frames — the window the
      // single-buffer apply corrupted.
      setSortedIndexChunkElementsForTests(1 + Math.floor(r() * 5));
      const capacityRequest = 64;
      const geometry = new THREE.InstancedBufferGeometry();
      attachSplatStorage(geometry, capacityRequest);
      let n = 1 + Math.floor(r() * 24);
      writeSortedIndexIdentity(geometry, n);

      for (let step = 0; step < 40; step++) {
        const roll = r();
        let op: string;
        if (roll < 0.3) {
          op = 'sort';
          writeSortedIndexOrdering(geometry, shuffled(n, r), n);
        } else if (roll < 0.7) {
          op = 'pump';
          pumpSortedIndexOrderingApply(geometry);
        } else if (roll < 0.8) {
          op = 'cancel';
          cancelSortedIndexOrderingApply(geometry);
        } else if (roll < 0.9) {
          op = 'commit-resize';
          n = 1 + Math.floor(r() * 24);
          writeSortedIndexIdentity(geometry, n);
        } else {
          op = 'append';
          const from = n;
          const grown = Math.min(capacityRequest, n + 1 + Math.floor(r() * 8));
          writeSortedIndexIdentityRange(geometry, from, grown);
          n = grown;
        }
        opTally[op] = (opTally[op] ?? 0) + 1;

        const drawn = getActiveSortedIndexAttribute(geometry)!.array as Uint32Array;
        const err = permError(drawn, n);
        checks++;
        expect(
          err,
          `seed=${seed} step=${step} op=${op} n=${n} slot=${activeSortedIndexSlot(geometry)}`
        ).toBeNull();
      }
    }

    // Non-vacuity: the run really visited every op class, many times.
    expect(checks).toBe(400 * 40);
    for (const op of ['sort', 'pump', 'cancel', 'commit-resize', 'append']) {
      expect(opTally[op] ?? 0, `op '${op}' never exercised`).toBeGreaterThan(50);
    }
  });
});
