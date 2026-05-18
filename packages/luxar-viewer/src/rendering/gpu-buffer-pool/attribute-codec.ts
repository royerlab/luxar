/**
 * Geometry-agnostic helpers for the GPU buffer pool's interleaved
 * attribute layout.
 *
 * Extracted from gpu-buffer-pool.ts so the per-type adapters (Points,
 * Lines, GSplats) carved out in a later refactor step share a single
 * implementation of buffer rebuild + attribute write, rather than each
 * duplicating the strided-copy bookkeeping.
 *
 * Per-type spec arrays (POINTS_BASE_ATTRIBUTE_SPECS,
 * LINES_BASE_ATTRIBUTE_SPECS, GSPLATS_ATTRIBUTE_SPECS, etc.) stay with
 * their adapters — only the generic codec functions live here.
 */

import * as THREE from 'three';
import {
  packInterleavedAttributes,
  writeInterleavedAttribute,
  type InterleavedAttributeSpec,
} from '../interleaved-attributes';

/**
 * Rebuild the interleaved buffer on a geometry with new capacity
 * and/or a new spec-set (e.g. lazily adding scalar attributes).
 * Copies as much of the old buffer as fits into the new layout.
 *
 * Returns the new buffer so the caller can stash it / wire usage.
 */
export function rebuildInterleavedBuffer(
  geometry: THREE.InstancedBufferGeometry,
  newCapacity: number,
  newSpecs: ReadonlyArray<{ name: string; itemSize: 1 | 2 | 3 | 4 }>
): THREE.InstancedInterleavedBuffer {
  // Snapshot the old buffer + per-attribute float offsets *before*
  // we replace anything. Used to copy still-present attribute data
  // across.
  const oldByName = new Map<
    string,
    { buffer: THREE.InstancedInterleavedBuffer; offset: number; itemSize: number }
  >();
  for (const spec of newSpecs) {
    const oldView = geometry.getAttribute(spec.name) as
      | THREE.InterleavedBufferAttribute
      | undefined;
    if (oldView && oldView.data) {
      oldByName.set(spec.name, {
        buffer: oldView.data as THREE.InstancedInterleavedBuffer,
        offset: oldView.offset,
        itemSize: oldView.itemSize,
      });
    }
  }

  const specsWithData: InterleavedAttributeSpec[] = newSpecs.map((spec) => ({
    name: spec.name,
    itemSize: spec.itemSize,
    data: new Float32Array(newCapacity * spec.itemSize),
  }));
  const { buffer: newBuffer, views: newViews } = packInterleavedAttributes(
    specsWithData,
    newCapacity
  );
  newBuffer.setUsage(THREE.DynamicDrawUsage);

  // Carry forward each old attribute's data into the new strided
  // layout. We deinterlace from the old buffer and reinterlace into
  // the new — the new offsets are determined by `newSpecs` order.
  for (const spec of newSpecs) {
    const old = oldByName.get(spec.name);
    if (!old) continue;
    const oldArray = old.buffer.array as Float32Array;
    const oldStride = old.buffer.stride;
    const oldCapacity = Math.floor(oldArray.length / oldStride);
    const carry = Math.min(oldCapacity, newCapacity);
    const newView = newViews[spec.name];
    const newOffset = newView.offset;
    const newStride = newBuffer.stride;
    const newArray = newBuffer.array as Float32Array;
    for (let i = 0; i < carry; i++) {
      const oldStart = i * oldStride + old.offset;
      const newStart = i * newStride + newOffset;
      for (let k = 0; k < spec.itemSize; k++) {
        newArray[newStart + k] = oldArray[oldStart + k];
      }
    }
    geometry.setAttribute(spec.name, newView);
  }
  // Attributes new to the spec-set (e.g. aStartScalar on a colormap
  // toggle) still need to be bound — the loop above only handles
  // names that already existed. Bind any that didn't carry forward.
  for (const spec of newSpecs) {
    if (!oldByName.has(spec.name)) {
      geometry.setAttribute(spec.name, newViews[spec.name]);
    }
  }
  // If the old buffer had attributes the new spec-set drops, remove
  // them so the geometry doesn't dangle stale views.
  for (const name of Object.keys(geometry.attributes)) {
    if (name === 'aQuadCorner') continue;
    if (!newSpecs.find((s) => s.name === name)) {
      geometry.deleteAttribute(name);
    }
  }

  // CRITICAL: r184 caches `_maxInstanceCount` on the geometry; replacing
  // the buffer doesn't invalidate it. Mirrors the standalone-geometry
  // workaround in line-geometry.ts / gsplat-geometry.ts.
  delete (geometry as unknown as { _maxInstanceCount?: number })._maxInstanceCount;

  return newBuffer;
}

/**
 * Write a packed per-attribute source array into the geometry's
 * interleaved buffer at the right strided offset. Internal helper —
 * the pool uses this from updateXxxGeometry instead of poking
 * `attr.set(...)` per-attribute.
 */
export function writePooledAttribute(
  geometry: THREE.InstancedBufferGeometry,
  name: string,
  src: Float32Array,
  count: number
): void {
  const view = geometry.getAttribute(name) as THREE.InterleavedBufferAttribute;
  const buffer = view.data as THREE.InstancedInterleavedBuffer;
  writeInterleavedAttribute(
    buffer,
    view.offset,
    view.itemSize,
    src.subarray(0, count * view.itemSize) as Float32Array,
    count
  );
}
