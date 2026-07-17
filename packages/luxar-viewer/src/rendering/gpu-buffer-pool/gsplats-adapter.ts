/**
 * GSplats buffer-pool adapter.
 *
 * Owns the gsplats-specific pool state (per-capacity buckets) and
 * implements acquire/release/update for Gaussian-splat geometries.
 * Shared state (activeBuffers, stats, frame counter) is read from the
 * GPUBufferPool reference passed at construction.
 *
 * The top-level GPUBufferPool owns only shared coordination logic
 * (eviction, frame counter, dispose); this adapter owns GSplats-specific
 * buffer layout and update behavior.
 */

import * as THREE from 'three';
import {
  attachSplatStorage,
  getSplatTexture,
  writeSplatTexels,
  writeSortedIndexIdentity,
} from '../gsplat-geometry';
import { clampSplatCapacity } from '../splat-texture-layout';
import type { PooledBuffer } from './pool-stats';
import { chooseCapacity } from './capacity';

/** Packed GSplats data ready for GPU upload (from gsplats/projection.ts). */
export interface PackedGSplatsData {
  centers3D: Float32Array; // M * 3
  amplitudes: Float32Array; // M
  cholesky01: Float32Array; // M * 2 [L00, L10]
  cholesky23: Float32Array; // M * 2 [L11, L20]
  cholesky45: Float32Array; // M * 2 [L21, L22]
  colors: Float32Array; // M * 3 (RGB)
  splatCount: number;
}

function createGSplatsGeometry(splatCapacity: number): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();

  const quadPositions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  geometry.setAttribute('aQuadCorner', new THREE.Float32BufferAttribute(quadPositions, 2));
  geometry.setIndex([0, 1, 2, 2, 1, 3]);

  // Splat data lives in the RGBA32F texture attached here (disposed BY
  // the geometry's dispose event, so every pool dispose site frees it);
  // `aSortedIndex` is the only per-instance attribute.
  attachSplatStorage(geometry, splatCapacity);
  return geometry;
}

export interface GSplatsAdapterHost {
  activeBuffers: Map<string, PooledBuffer>;
  readonly frameCount: number;
  stats: {
    allocations: number;
    reuses: number;
    evictions: number;
    capacityGrowths: number;
    deferredEvictions: number;
  };
  typeStats: {
    points: { allocations: number; reuses: number; evictions: number };
    lines: { allocations: number; reuses: number; evictions: number };
    gsplats: { allocations: number; reuses: number; evictions: number };
  };
  _lastAcquireRebuilt: boolean;
  getBucket(count: number): number;
  evictUnused(fromAcquire?: boolean): number;
}

export class GSplatsBufferAdapter {
  /** @internal — pool-bucketed reusable gsplat geometries. */
  readonly gsplatBuffers = new Map<number, PooledBuffer[]>();

  constructor(private readonly host: GSplatsAdapterHost) {}

  acquireGeometry(nodeId: string, splatCount: number): THREE.InstancedBufferGeometry {
    const host = this.host;
    host._lastAcquireRebuilt = false;

    // Per-node texture bound: width × maxTextureSize / 4 texels (4.19M
    // splats on a 4096-class device). Warns once; the update path
    // clamps its written count to the texture capacity to match.
    splatCount = clampSplatCapacity(splatCount);

    const active = host.activeBuffers.get(nodeId);
    if (active && active.type === 'gsplats') {
      if (active.capacity >= splatCount) {
        active.lastUsedFrame = host.frameCount;
        host.stats.reuses++;
        host.typeStats.gsplats.reuses++;
        return active.geometry as THREE.InstancedBufferGeometry;
      } else {
        // Grow = RELEASE + REACQUIRE — an in-place rebuild strands the
        // old GL/GPU buffer in the renderer caches (hard leak under the
        // WebGPU renderer via the strong Info.memoryMap). See the
        // points adapter for the full rationale. Fall-through best-fit/
        // fresh-alloc sets _lastAcquireRebuilt + allocation counters.
        host.stats.capacityGrowths++;
        this.releaseGeometry(nodeId);
      }
    }

    // BEST-fit, not first-fit: scan every pooled candidate and claim the
    // smallest adequate one. Map iteration order is bucket-insertion
    // order, so first-fit could pin an arbitrarily oversized buffer
    // (e.g. a 52 MB 1M-capacity buffer) to a small node until release.
    let bestList: PooledBuffer[] | null = null;
    let bestIndex = -1;
    let bestCapacity = Infinity;
    for (const pooled of this.gsplatBuffers.values()) {
      for (let i = pooled.length - 1; i >= 0; i--) {
        const candidate = pooled[i];
        if (candidate.capacity >= splatCount && candidate.capacity < bestCapacity) {
          bestList = pooled;
          bestIndex = i;
          bestCapacity = candidate.capacity;
        }
      }
    }
    if (bestList) {
      const candidate = bestList[bestIndex];
      bestList.splice(bestIndex, 1);
      candidate.inUse = true;
      candidate.lastUsedFrame = host.frameCount;
      host.activeBuffers.set(nodeId, candidate);
      host.stats.reuses++;
      host.typeStats.gsplats.reuses++;
      host._lastAcquireRebuilt = true;
      return candidate.geometry as THREE.InstancedBufferGeometry;
    }

    host._lastAcquireRebuilt = true;
    // Growth headroom (1.5×) can itself cross the texture bound; clamp
    // the chosen capacity too (still >= splatCount, which was clamped).
    const capacity = clampSplatCapacity(chooseCapacity(splatCount));
    const geometry = createGSplatsGeometry(capacity);

    const newBuffer: PooledBuffer = {
      geometry,
      capacity,
      type: 'gsplats',
      inUse: true,
      lastUsedFrame: host.frameCount,
    };

    host.activeBuffers.set(nodeId, newBuffer);
    host.stats.allocations++;
    // Fresh allocations count against the byte budget too — sweep idle
    // pooled buffers (see growth-path note above).
    host.evictUnused(true);
    host.typeStats.gsplats.allocations++;
    return geometry;
  }

  releaseGeometry(nodeId: string): void {
    const host = this.host;
    const buffer = host.activeBuffers.get(nodeId);
    if (!buffer || buffer.type !== 'gsplats') return;

    host.activeBuffers.delete(nodeId);
    buffer.inUse = false;
    // Stamp the release frame so acquire-triggered byte sweeps later in
    // this same frame grace the buffer (see EvictorCtx.graceFrame) — a
    // released buffer otherwise carries the frame of its last ACQUIRE
    // and the dataset-switch grace never matches. Also makes the
    // just-released buffer the freshest LRU reuse candidate.
    buffer.lastUsedFrame = host.frameCount;

    const bucket = host.getBucket(buffer.capacity);
    if (!this.gsplatBuffers.has(bucket)) {
      this.gsplatBuffers.set(bucket, []);
    }
    this.gsplatBuffers.get(bucket)!.push(buffer);

    host.evictUnused();
  }

  updateGeometry(
    geometry: THREE.InstancedBufferGeometry,
    data: PackedGSplatsData,
    count: number,
    truncationRadius: number = 3.0,
    options?: { preserveOrdering?: boolean }
  ): void {
    const texture = getSplatTexture(geometry);
    if (!texture) {
      throw new Error(
        'GSplatsBufferAdapter.updateGeometry: geometry has no splat texture — ' +
          'was it acquired from the pool?'
      );
    }
    // One fused pass over the staged arrays into the texel layout
    // (replaces the six per-attribute strided writes), then identity
    // ordering. The writer clamps to the texture capacity; mirror that
    // clamp in instanceCount so a bound-clamped node never draws
    // instances whose texels were not written.
    count = writeSplatTexels(
      texture,
      {
        centers: data.centers3D,
        cholesky01: data.cholesky01,
        cholesky23: data.cholesky23,
        cholesky45: data.cholesky45,
        amplitudes: data.amplitudes,
        colors: data.colors,
      },
      count
    );
    // `preserveOrdering` (commit path decides — see
    // commit-gsplats-geometry.ts): keep the node's existing depth-sort
    // permutation instead of resetting to identity, so a same-count
    // recommit doesn't flash storage order while the re-sort lands.
    // Skipping the write also skips registering an update range — correct
    // because the attribute content didn't change (the already-uploaded
    // permutation stays valid; any still-pending ranges from earlier
    // writes remain registered on the attribute and flush as usual).
    if (!options?.preserveOrdering) {
      writeSortedIndexIdentity(geometry, count);
    }

    geometry.instanceCount = count;

    // CRITICAL: Force THREE.js to recalculate _maxInstanceCount.
    delete (geometry as unknown as { _maxInstanceCount?: number })._maxInstanceCount;

    // CRITICAL: Recompute bounding box from updated center positions
    // and expand by max splat extent for correct frustum culling.
    // For each splat, the per-axis extent is truncationRadius × σ_d, where
    // σ_d = ||L[d,:]|| (the row norm of the Cholesky factor). We use the
    // max row norm across all splats and axes as a conservative expansion.
    //
    // Cholesky layout (packed as attribute pairs):
    //   cholesky01 = [L00, L10], cholesky23 = [L11, L20], cholesky45 = [L21, L22]
    // Row norms: ||row0|| = |L00|, ||row1|| = sqrt(L10² + L11²),
    //            ||row2|| = sqrt(L20² + L21² + L22²)
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      v.set(data.centers3D[i * 3], data.centers3D[i * 3 + 1], data.centers3D[i * 3 + 2]);
      box.expandByPoint(v);
    }

    let maxRowNorm = 0;
    for (let i = 0; i < count; i++) {
      const L00 = data.cholesky01[i * 2];
      const L10 = data.cholesky01[i * 2 + 1];
      const L11 = data.cholesky23[i * 2];
      const L20 = data.cholesky23[i * 2 + 1];
      const L21 = data.cholesky45[i * 2];
      const L22 = data.cholesky45[i * 2 + 1];

      const row0 = Math.abs(L00);
      const row1 = Math.sqrt(L10 * L10 + L11 * L11);
      const row2 = Math.sqrt(L20 * L20 + L21 * L21 + L22 * L22);
      maxRowNorm = Math.max(maxRowNorm, row0, row1, row2);
    }
    const expansion = maxRowNorm * truncationRadius;
    box.expandByScalar(expansion);

    geometry.boundingBox = box;
    const sphere = new THREE.Sphere();
    box.getBoundingSphere(sphere);
    geometry.boundingSphere = sphere;
  }

  dispose(): void {
    for (const buffers of this.gsplatBuffers.values()) {
      for (const buffer of buffers) {
        buffer.geometry.dispose();
      }
    }
    this.gsplatBuffers.clear();
  }
}
