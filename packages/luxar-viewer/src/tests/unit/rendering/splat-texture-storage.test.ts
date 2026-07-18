/**
 * Phase 1 texture-backed splat storage — unit coverage for the layout
 * authority (`splat-texture-layout.ts`), the storage helpers
 * (`gsplat-geometry.ts::attachSplatStorage` and friends), the pool
 * adapter's growth/dispose behavior, byte accounting, and the commit
 * material sync. See `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md`
 * §4 for the design.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';
import {
  SPLAT_FLOATS_PER_SPLAT,
  SPLAT_TEXELS_PER_SPLAT,
  configureSplatTextureLayout,
  resetSplatTextureLayoutForTests,
  getSplatTextureWidth,
  getMaxSplatCapacityPerNode,
  clampSplatCapacity,
  splatTextureHeightForCapacity,
  getPlaceholderSplatTexture,
} from '../../../rendering/splat-texture-layout';
import {
  attachSplatStorage,
  getSplatTexture,
  splatTexelCapacity,
  writeSplatTexels,
  writeSortedIndexIdentity,
  writeSortedIndexOrdering,
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
  const cholesky01 = new Float32Array(count * 2);
  const cholesky23 = new Float32Array(count * 2);
  const cholesky45 = new Float32Array(count * 2);
  const amplitudes = new Float32Array(count);
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    centers.set([i, i + 0.25, i + 0.5], i * 3);
    cholesky01.set([1 + i, 0.1 * i], i * 2);
    cholesky23.set([2 + i, 0.2 * i], i * 2);
    cholesky45.set([0.3 * i, 3 + i], i * 2);
    amplitudes[i] = 0.5 + i;
    colors.set([i * 0.01, i * 0.02, i * 0.03], i * 3);
  }
  return { centers, cholesky01, cholesky23, cholesky45, amplitudes, colors };
}

afterEach(() => {
  resetSplatTextureLayoutForTests();
});

describe('splat-texture-layout — texel address math', () => {
  it('defaults to a 4096-wide texture with a 4096² capacity bound', () => {
    expect(getSplatTextureWidth()).toBe(4096);
    expect(getMaxSplatCapacityPerNode()).toBe((4096 * 4096) / SPLAT_TEXELS_PER_SPLAT);
  });

  it('caps the width at min(4096, maxTextureSize) and forces a multiple of 4', () => {
    configureSplatTextureLayout(16384);
    expect(getSplatTextureWidth()).toBe(4096); // never wider than 4096

    // Non-4096 width: a 2048-class device.
    configureSplatTextureLayout(2048);
    expect(getSplatTextureWidth()).toBe(2048);
    expect(getMaxSplatCapacityPerNode()).toBe((2048 * 2048) / SPLAT_TEXELS_PER_SPLAT);

    // A pathological non-multiple-of-4 limit is rounded DOWN so a
    // splat's 4 texels can never straddle a row boundary.
    configureSplatTextureLayout(2050);
    expect(getSplatTextureWidth()).toBe(2048);

    // Sub-4 limits floor at 4 (would otherwise round to width 0 and
    // divide-by-zero the height math).
    configureSplatTextureLayout(3);
    expect(getSplatTextureWidth()).toBe(4);
  });

  it('computes row-padded texture heights', () => {
    configureSplatTextureLayout(4096);
    // 1024 splats/row at width 4096.
    expect(splatTextureHeightForCapacity(0)).toBe(1);
    expect(splatTextureHeightForCapacity(1)).toBe(1);
    expect(splatTextureHeightForCapacity(1024)).toBe(1);
    expect(splatTextureHeightForCapacity(1025)).toBe(2);
  });

  it('clamps requested capacities to the per-node texture bound', () => {
    configureSplatTextureLayout(2048);
    const max = getMaxSplatCapacityPerNode();
    expect(clampSplatCapacity(max - 1)).toBe(max - 1);
    expect(clampSplatCapacity(max)).toBe(max);
    expect(clampSplatCapacity(max + 1)).toBe(max);
  });

  it('exposes a single shared placeholder texture', () => {
    expect(getPlaceholderSplatTexture()).toBe(getPlaceholderSplatTexture());
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
      // texel 1: cholesky01.xy, cholesky23.xy
      expect(arr[o + 4]).toBe(src.cholesky01[i * 2]);
      expect(arr[o + 5]).toBe(src.cholesky01[i * 2 + 1]);
      expect(arr[o + 6]).toBe(src.cholesky23[i * 2]);
      expect(arr[o + 7]).toBe(src.cholesky23[i * 2 + 1]);
      // texel 2: cholesky45.xy, color.rg
      expect(arr[o + 8]).toBe(src.cholesky45[i * 2]);
      expect(arr[o + 9]).toBe(src.cholesky45[i * 2 + 1]);
      expect(arr[o + 10]).toBe(src.colors[i * 3]);
      expect(arr[o + 11]).toBe(src.colors[i * 3 + 1]);
      // texel 3: color.b, padding zeros
      expect(arr[o + 12]).toBe(src.colors[i * 3 + 2]);
      expect(arr[o + 13]).toBe(0);
    }
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
    const cap = splatTexelCapacity(texture);
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

  it('writeSortedIndexOrdering clamps to ordering AND attribute lengths', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    attachSplatStorage(geometry, 8);
    // ordering shorter than count: writes only ordering.length entries.
    let n = writeSortedIndexOrdering(geometry, new Uint32Array([3, 1]), 5);
    expect(n).toBe(2);
    const arr = geometry.getAttribute('aSortedIndex').array as Uint32Array;
    expect(arr[0]).toBe(3);
    expect(arr[1]).toBe(1);
    // ordering longer than the attribute: clamps to the attribute.
    const long = new Uint32Array(32).fill(7);
    n = writeSortedIndexOrdering(geometry, long, 32);
    expect(n).toBe(arr.length);
    // Update ranges stay a single collapsed prefix across mixed writes.
    const attr = geometry.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
    expect(attr.updateRanges.length).toBe(1);
    expect(attr.updateRanges[0].start).toBe(0);
  });

  it('clamps the attach capacity to the per-node bound (structural safety net)', () => {
    // maxTextureSize 8 → width 8, bound = 8×8/4 = 16 splats — so an
    // over-bound request must yield a texture no taller than 8 rows
    // (height ≤ maxTextureSize by construction) and a matching
    // aSortedIndex length, whatever the caller asked for.
    configureSplatTextureLayout(8);
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachSplatStorage(geometry, 100);
    expect(texture.image.height).toBeLessThanOrEqual(8);
    expect((geometry.getAttribute('aSortedIndex').array as Uint32Array).length).toBe(16);
    expect(splatTexelCapacity(texture)).toBe(16);
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

  it('estimateGeometryBytes counts texture + ordering (≈68 B/splat envelope)', () => {
    const geom = pool.acquireGSplatsGeometry('node', 1000);
    const capacity = (geom.getAttribute('aSortedIndex').array as Uint32Array).length;
    const bytes = estimateGeometryBytes(geom);
    // Texture is row-padded, so expect at least capacity × (64 + 4) B
    // and no more than one extra texture row + static quad overhead.
    const rowBytes = getSplatTextureWidth() * 16;
    expect(bytes).toBeGreaterThanOrEqual(capacity * 68);
    expect(bytes).toBeLessThanOrEqual(capacity * 68 + rowBytes + 256);
  });

  it('clamps acquire capacity AND written count to the per-node texture bound', () => {
    // Shrink the bound so the clamp is testable at unit scale:
    // maxTextureSize 16 -> width 16, bound = 16*16/4 = 64 splats.
    configureSplatTextureLayout(16);
    const small = new GPUBufferPool(20, 300, 5, () => Infinity);
    try {
      const geom = small.acquireGSplatsGeometry('big', 1000);
      const texture = getSplatTexture(geom)!;
      expect(splatTexelCapacity(texture)).toBe(64);
      const src = makeSource(64); // writer clamps count to capacity first
      small.updateGSplatsGeometry(
        geom,
        {
          centers3D: src.centers,
          amplitudes: src.amplitudes,
          cholesky01: src.cholesky01,
          cholesky23: src.cholesky23,
          cholesky45: src.cholesky45,
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
        cholesky01: src.cholesky01,
        cholesky23: src.cholesky23,
        cholesky45: src.cholesky45,
        colors: src.colors,
      },
      4
    );
    expect(geom.instanceCount).toBe(4);
    const arr = getSplatTexture(geom)!.image.data as Float32Array;
    expect(arr[SPLAT_FLOATS_PER_SPLAT * 3]).toBe(src.centers[9]); // splat 3 center.x
    expect(geom.boundingBox).not.toBeNull();
  });

  it('preserveOrdering keeps the sort permutation while still rewriting texels', () => {
    const geom = pool.acquireGSplatsGeometry('node', 4);
    const src = makeSource(4);
    const packed = (amplitudes: Float32Array) => ({
      centers3D: src.centers,
      amplitudes,
      cholesky01: src.cholesky01,
      cholesky23: src.cholesky23,
      cholesky45: src.cholesky45,
      colors: src.colors,
    });
    pool.updateGSplatsGeometry(geom, packed(src.amplitudes), 4);
    // The SortWorker landed a depth-sort permutation between commits.
    writeSortedIndexOrdering(geom, new Uint32Array([3, 2, 1, 0]), 4);

    // Same-count recommit with preserveOrdering: permutation intact,
    // texels + instanceCount + bounds refreshed as usual.
    const newAmplitudes = new Float32Array([9, 8, 7, 6]);
    pool.updateGSplatsGeometry(geom, packed(newAmplitudes), 4, 3.0, { preserveOrdering: true });
    const ordering = geom.getAttribute('aSortedIndex').array as Uint32Array;
    expect(Array.from(ordering.subarray(0, 4))).toEqual([3, 2, 1, 0]);
    const texels = getSplatTexture(geom)!.image.data as Float32Array;
    expect(texels[3]).toBe(9); // splat 0 amplitude — texels WERE rewritten
    expect(geom.instanceCount).toBe(4);

    // Without the flag the identity reset is restored (default behavior).
    pool.updateGSplatsGeometry(geom, packed(src.amplitudes), 4);
    expect(Array.from(ordering.subarray(0, 4))).toEqual([0, 1, 2, 3]);
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
