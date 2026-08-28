/**
 * GPU-resident geometry byte accounting. Used by the GPU buffer pool
 * (and adapters) to track memory pressure across point / line / gsplat
 * layers without re-iterating attribute byteLengths on every stats poll.
 *
 * @module rendering/gpu-buffer-pool/geometry-bytes
 */

import * as THREE from 'three';

/**
 * Estimate the GPU-resident byte footprint of a geometry by summing
 * the underlying typed-array byte lengths of every attribute (and
 * the index, if present). Mirrors what THREE.js will actually upload
 * — slightly overstates because we count the full backing array even
 * if `count < array.length / itemSize`, but that's the footprint that
 * matters for pool memory pressure.
 *
 * The result is cached on `geometry.userData.cachedByteSize` so repeated
 * stats polls don't re-iterate attribute byteLengths. Grow paths
 * invalidate the cache via `invalidateCachedByteSize()`.
 */
export function estimateGeometryBytes(geometry: THREE.BufferGeometry): number {
  const userData = geometry.userData as { cachedByteSize?: number };
  if (typeof userData.cachedByteSize === 'number') {
    return userData.cachedByteSize;
  }
  let total = 0;
  // Multiple `InterleavedBufferAttribute` views can share one
  // underlying `InterleavedBuffer`. Counting `attr.array.byteLength`
  // for each view would multi-count the shared backing array; dedupe
  // by visited-buffer identity.
  const seenInterleavedBuffers = new WeakSet<THREE.InterleavedBuffer>();
  // Two NAMES can also resolve to one attribute object — a hand-built
  // geometry may alias the ordering pair `aSortedIndex`/`aSortedIndexB`
  // onto one buffer (`attachElementStorage` allocates two distinct ones,
  // so production never does). Counting the shared array twice would
  // charge such a node for a back buffer it does not own.
  const seenAttributes = new WeakSet<THREE.BufferAttribute | THREE.InterleavedBufferAttribute>();
  for (const name in geometry.attributes) {
    const attr = geometry.attributes[name];
    if (seenAttributes.has(attr)) continue;
    seenAttributes.add(attr);
    const interleaved = (attr as THREE.InterleavedBufferAttribute).data;
    if (interleaved !== undefined) {
      if (seenInterleavedBuffers.has(interleaved)) continue;
      seenInterleavedBuffers.add(interleaved);
      const arr = interleaved.array as ArrayBufferView | undefined;
      if (arr && typeof arr.byteLength === 'number') {
        total += arr.byteLength;
      }
      continue;
    }
    const arr = (attr as THREE.BufferAttribute).array as ArrayBufferView | undefined;
    if (arr && typeof arr.byteLength === 'number') {
      total += arr.byteLength;
    }
  }
  // InstancedBufferGeometry indices are shared with the base geometry
  // (a single quad), so they're a fixed overhead — small, but include
  // them for correctness.
  if (geometry.index) {
    const idxArr = geometry.index.array as ArrayBufferView | undefined;
    if (idxArr && typeof idxArr.byteLength === 'number') {
      total += idxArr.byteLength;
    }
  }
  // GSplat, Points and Lines geometries all carry their element data in an
  // RGBA32F texture (gsplats: 64 B/splat, 4 texels; points: 48 B/point, 3
  // texels; lines: 96 B/segment, 6 texels) riding `userData.elementTexture`
  // (see
  // `element-storage.ts::attachElementStorage`); it shares the
  // geometry's lifetime, so its footprint belongs to the geometry. The
  // capacity-padded texture rows plus the `aSortedIndex`/`aSortedIndexB`
  // ordering PAIR (4 B/element each = 8 B/element, both counted in the
  // attribute loop above — `attachElementStorage` allocates two distinct
  // buffers) and the static quad make up the whole estimate for
  // texture-backed geometries.
  const elementTexture = (geometry.userData as { elementTexture?: THREE.DataTexture })
    .elementTexture;
  if (elementTexture) {
    const texArr = elementTexture.image.data as ArrayBufferView | undefined;
    if (texArr && typeof texArr.byteLength === 'number') {
      total += texArr.byteLength;
    }
  }
  userData.cachedByteSize = total;
  return total;
}

/**
 * Invalidate the cached byte estimate on a geometry. Call after
 * resizing/replacing any attribute or index so the next
 * `estimateGeometryBytes` call recomputes.
 */
export function invalidateCachedByteSize(geometry: THREE.BufferGeometry): void {
  const userData = geometry.userData as { cachedByteSize?: number };
  delete userData.cachedByteSize;
}
