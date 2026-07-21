/**
 * Geometry-agnostic helpers for the GPU buffer pool's interleaved
 * attribute layout.
 *
 * Consumed by the Lines adapter (the one remaining interleaved type)
 * so the attribute write logic stays in one implementation rather than
 * duplicating strided-copy bookkeeping. There is deliberately NO
 * in-place buffer rebuild here: growth and spec-set changes go through
 * release + reacquire (a rebuild on a rendered geometry strands the
 * old GPU buffer in the renderer caches — a permanent leak under the
 * WebGPU renderer).
 *
 * The spec arrays (LINES_BASE_ATTRIBUTE_SPECS etc.) stay with the
 * adapter — only the generic codec functions live here. (Points and
 * GSplats no longer use this path: their data lives in RGBA32F element
 * textures — see `../point-geometry.ts` / `../gsplat-geometry.ts`.)
 */

import * as THREE from 'three';
import { writeInterleavedAttribute } from '../interleaved-attributes';

/**
 * Write a packed per-attribute source array into the geometry's
 * interleaved buffer at the right strided offset. Internal helper —
 * the pool uses this from updateXxxGeometry instead of poking
 * `attr.set(...)` per-attribute.
 *
 * `opts.fromInstance` passes straight through to
 * {@link writeInterleavedAttribute} (append fast path): the source
 * stays full-length, only the write + dirty range start at the
 * given instance.
 */
export function writePooledAttribute(
  geometry: THREE.InstancedBufferGeometry,
  name: string,
  src: Float32Array,
  count: number,
  opts?: { fromInstance?: number }
): void {
  const view = geometry.getAttribute(name) as THREE.InterleavedBufferAttribute;
  const buffer = view.data as THREE.InstancedInterleavedBuffer;
  writeInterleavedAttribute(
    buffer,
    view.offset,
    view.itemSize,
    src.subarray(0, count * view.itemSize) as Float32Array,
    count,
    opts
  );
}
