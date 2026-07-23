/**
 * Shared texture-backed element storage for instanced geometry.
 *
 * Per-element data lives in an RGBA32F **element texture** (gsplats:
 * 4 texels/splat; points: 3 texels/point; lines later — layout
 * authority in `./element-texture-layout`) sampled by the vertex
 * shader via `texelFetch`; the only per-instance attribute is
 * `aSortedIndex` (Uint32), which maps the draw slot to a storage slot
 * so draw order can be permuted without rewriting element data
 * (depth-sorting plan, `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md`
 * §4/§8).
 *
 * Texture lifetime = geometry lifetime: `attachElementStorage`
 * registers a `dispose` listener on the geometry, so every dispose
 * site (pool evictors, `pool.dispose()`, the fallback rebuild swap)
 * frees the texture with the geometry — no site-by-site bookkeeping.
 *
 * @module rendering/element-storage
 */

import * as THREE from 'three';
import {
  type ElementTextureLayout,
  clampElementCapacity,
  getElementTextureWidth,
  elementTextureHeightForCapacity,
} from './element-texture-layout';

/**
 * Above this fraction of a texture's rows being dirty, the per-row
 * `texSubImage2D` call overhead outweighs the bytes a ranged upload
 * saves, so we fall back to three's single full-image upload. Empirical
 * knee: a full-row transfer (~64 KB at width 4096) costs ~4× a bare GL
 * call, so ranges win comfortably up to ~¾ of the rows.
 */
const FULL_UPLOAD_ROW_FRACTION = 0.75;

/**
 * Register the dirty element span `[firstElement, endElement)` on an
 * element texture as per-row `updateRanges`, so the classic WebGLRenderer
 * uploads only those rows (its `updateTexture` takes the whole-image
 * `texSubImage2D` path ONLY when `updateRanges` is empty). Pool
 * growth-headroom + best-fit slack rows past the live count therefore stop
 * riding every commit to the GPU (measured 5.81 MB → 2.36 MB per commit on
 * a real timelapse whose pool was sized by a larger intro frame).
 *
 * Range discipline mirrors {@link collapseSortedIndexRanges}:
 * - Units are FLOAT elements of `image.data` (three's texture-range API is
 *   float-indexed with an implicit RGBA `componentStride` of 4).
 * - Ranges accumulate across commits while a mesh is hidden, and BOTH
 *   WebGPU backends (native + WebGL2-fallback) replay them verbatim and
 *   never clear them — only the classic WebGLRenderer consumes+clears at
 *   flush. So every call collapses the pending set into one contiguous
 *   span and re-splits it (a superset upload is always correct, never
 *   stale; our writers only ever register a `[0, n)` prefix or an append
 *   suffix contiguous with it).
 * - An element is exactly `floatsPerElement` floats and the texture width
 *   is a multiple of the layout's texels-per-element, so elements never
 *   straddle rows — each emitted range stays within one row (the WebGL
 *   path uploads every range with `height = 1` and would reject a
 *   row-straddling range with INVALID_VALUE).
 *
 * On the WebGPU backends this still sets `needsUpdate`; they ignore the
 * ranges and re-upload the whole image (correct, just not yet partial —
 * see the Phase-4 spec's Stage 3).
 */
/**
 * Textures with a FULL-image upload pending (encoded to three as
 * `needsUpdate` + EMPTY `updateRanges`). The empty-ranges encoding is
 * invisible to the range fold below — without this set, a later append
 * would register only its own span and silently DOWNGRADE the pending
 * full upload to a partial one, leaving the prefix rendering the
 * previous commit's texels on the classic WebGL backend (found by
 * model-based fuzzing; deterministic repro: full write ≥75% of rows →
 * append before any flush). Cleared by `texture.onUpdate`, which the
 * classic renderer invokes after it actually consumes the upload (the
 * WebGPU backends may never call it — harmless, they full-upload on
 * every needsUpdate anyway).
 */
const pendingFullUpload = new WeakSet<THREE.DataTexture>();

/**
 * Mark an element texture as needing a FULL image upload (context
 * restore, external invalidation). Registers the pending-full state so
 * later ranged writes cannot downgrade it.
 */
export function markElementTextureFullDirty(texture: THREE.DataTexture): void {
  texture.clearUpdateRanges();
  pendingFullUpload.add(texture);
  texture.needsUpdate = true;
}

export function registerElementTexelDirtyRange(
  texture: THREE.DataTexture,
  floatsPerElement: number,
  firstElement: number,
  endElement: number
): void {
  // A pending full upload covers ANY span — keep full mode (registering
  // a partial range here would downgrade it; see pendingFullUpload).
  if (pendingFullUpload.has(texture)) {
    texture.needsUpdate = true;
    return;
  }
  // Derive row geometry from the texture's OWN dimensions (a multiple of
  // the layout's texels-per-element by construction — see
  // attachElementStorage), never the global `getElementTextureWidth()`: a
  // renderer/backend swap can reconfigure the session width while an
  // existing texture keeps its allocated width, and a mismatch here would
  // split against the wrong row stride and straddle rows.
  const rowFloats = texture.image.width * 4;
  const totalRows = Math.max(1, texture.image.height);

  // Collapse any pending ranges + the new span into one contiguous float
  // span (min start, max end). Every writer registers a contiguous prefix
  // or an append suffix, so the union is an exact or superset cover. An
  // EMPTY new span contributes nothing — seeding the fold with its
  // position would inflate pending ranges up to it (e.g. a pending
  // [100, 300) + an at-capacity no-op append at float 60000 would upload
  // [100, 60000) of clean data).
  let startFloat = Infinity;
  let endFloat = -Infinity;
  if (endElement > firstElement) {
    startFloat = firstElement * floatsPerElement;
    endFloat = endElement * floatsPerElement;
  }
  for (const range of texture.updateRanges) {
    if (range.start < startFloat) startFloat = range.start;
    const end = range.start + range.count;
    if (end > endFloat) endFloat = end;
  }
  texture.clearUpdateRanges();

  if (endFloat <= startFloat) {
    // Nothing dirty anywhere (empty span, no pending ranges). Do NOT set
    // needsUpdate: with empty updateRanges three takes its FULL-image
    // texSubImage2D path, which re-uploads the entire capacity-sized
    // backing store of a reused pool texture (tens of MB at multi-million
    // element capacity) on every empty commit — e.g. nD navigation into an
    // empty slice. Fresh textures are covered by attachElementStorage's
    // own needsUpdate = true.
    return;
  }

  const firstRow = Math.floor(startFloat / rowFloats);
  const lastRow = Math.floor((endFloat - 1) / rowFloats);
  const dirtyRows = lastRow - firstRow + 1;

  if (dirtyRows >= FULL_UPLOAD_ROW_FRACTION * totalRows) {
    // Too much dirty to bother splitting: leave updateRanges empty so
    // three takes its single full-image texSubImage2D path — and record
    // the pending-full state so a later ranged write can't downgrade it
    // before the renderer flushes (see pendingFullUpload).
    pendingFullUpload.add(texture);
    texture.needsUpdate = true;
    return;
  }

  for (let row = firstRow; row <= lastRow; row++) {
    const rowStart = row * rowFloats;
    const spanStart = Math.max(startFloat, rowStart);
    const spanEnd = Math.min(endFloat, rowStart + rowFloats);
    texture.addUpdateRange(spanStart, spanEnd - spanStart);
  }
  texture.needsUpdate = true;
}

/** `geometry.userData` slot carrying the element texture. */
interface ElementStorageUserData {
  elementTexture?: THREE.DataTexture;
}

/**
 * Create the element data texture + `aSortedIndex` attribute pair on a
 * geometry, sized for `capacity` elements of `layout`. Returns the
 * texture.
 *
 * - The texture is RGBA32F, `NearestFilter`, no mips, `flipY: false`
 *   — pure structured storage, addressed by `texelFetch` in the
 *   vertex shader (colormap-LUT precedent).
 * - The texture rides `geometry.userData.elementTexture` and is
 *   disposed BY the geometry's own `dispose` event, so texture
 *   lifetime is structurally pinned to geometry lifetime at every
 *   dispose site. Growth therefore follows the pool contract for
 *   free: release + reacquire swaps in a fresh geometry+texture pair,
 *   never an in-place reallocation (the `Info.memoryMap` strand
 *   class).
 * - `aSortedIndex` is a `Uint32Array` instanced attribute — the GL
 *   type `UNSIGNED_INT` makes three bind it via `vertexAttribIPointer`
 *   (matching the shader's `in uint`), and the WebGPU path derives its
 *   `uint32` vertex format from the array constructor.
 */
export function attachElementStorage(
  geometry: THREE.InstancedBufferGeometry,
  capacity: number,
  layout: ElementTextureLayout
): THREE.DataTexture {
  // STRUCTURAL invariant: every element texture is allocated here, so this
  // clamp alone guarantees texture height ≤ maxTextureSize by construction
  // (and caps aSortedIndex to match). Idempotent for the pool path, which
  // already clamps at acquire time.
  capacity = clampElementCapacity(capacity, layout);
  const sortedIndex = new THREE.InstancedBufferAttribute(new Uint32Array(capacity), 1);
  sortedIndex.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('aSortedIndex', sortedIndex);

  const width = getElementTextureWidth(layout);
  const height = elementTextureHeightForCapacity(capacity, layout);
  const texture = new THREE.DataTexture(
    new Float32Array(width * height * 4),
    width,
    height,
    THREE.RGBAFormat,
    THREE.FloatType
  );
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.generateMipmaps = false;
  texture.flipY = false;
  // The classic renderer invokes onUpdate after consuming an upload —
  // the observable "flush happened" signal that ends a pending FULL
  // upload (see pendingFullUpload). Wired once here; nothing else sets
  // onUpdate on element textures. (No pending-full mark at attach: a
  // fresh texture's GPU storage is zero-initialized and only written
  // texels are ever read, so a ranged first upload is sufficient.)
  texture.onUpdate = () => pendingFullUpload.delete(texture);
  texture.needsUpdate = true;

  (geometry.userData as ElementStorageUserData).elementTexture = texture;
  geometry.addEventListener('dispose', () => texture.dispose());
  return texture;
}

/** The element texture attached by `attachElementStorage`, if any. */
export function getElementTexture(geometry: THREE.BufferGeometry): THREE.DataTexture | null {
  return (geometry.userData as ElementStorageUserData).elementTexture ?? null;
}

/**
 * Number of elements the texture's backing store can hold (its
 * row-padded float capacity, NOT the pool bucket capacity).
 */
export function elementTexelCapacity(texture: THREE.DataTexture, floatsPerElement: number): number {
  const arr = texture.image.data as Float32Array;
  return Math.floor(arr.length / floatsPerElement);
}

/**
 * Fill `aSortedIndex[0..count)` with identity ordering and register a
 * single collapsed prefix update range. Ranges accumulate across
 * commits while a mesh is not drawn and the WebGPU backends replay
 * them verbatim (no flush-time merge), so every write collapses the
 * pending set to one `[0, max-end)` range.
 */
export function writeSortedIndexIdentity(
  geometry: THREE.InstancedBufferGeometry,
  count: number
): void {
  const attr = geometry.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
  const arr = attr.array as Uint32Array;
  const n = Math.min(count, arr.length);
  for (let i = 0; i < n; i++) arr[i] = i;
  collapseSortedIndexRanges(attr, n);
}

/**
 * Extend `aSortedIndex` with identity ordering for the appended suffix
 * `[from, count)`, preserving the existing `[0, from)` permutation
 * (depth-sorting Phase 4 Stage 2, the append fast path). The new splats
 * index themselves until the depth-sort coordinator re-sorts on a
 * subsequent frame; identity is the correct pre-resort placeholder (the
 * same value {@link writeSortedIndexIdentity} would write for them). The
 * collapse still uploads `[0, count)` — `aSortedIndex` is a tiny
 * 4-byte/instance buffer, so a full re-upload of the index is cheap; the
 * expensive splat-texel upload is the one Stage 2 restricts to the suffix.
 */
export function writeSortedIndexIdentityRange(
  geometry: THREE.InstancedBufferGeometry,
  from: number,
  count: number
): void {
  const attr = geometry.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
  const arr = attr.array as Uint32Array;
  const n = Math.min(count, arr.length);
  for (let i = Math.max(0, from); i < n; i++) arr[i] = i;
  collapseSortedIndexRanges(attr, n);
}

/**
 * Write a depth-sort permutation into `aSortedIndex[0..count)` (the
 * SortWorker's back-to-front ordering, depth-sorting Phase 2). Same
 * collapsed-prefix update-range discipline as
 * {@link writeSortedIndexIdentity}. Returns the number of entries
 * written (clamped to both the ordering's and the attribute's length).
 */
export function writeSortedIndexOrdering(
  geometry: THREE.InstancedBufferGeometry,
  ordering: Uint32Array,
  count: number
): number {
  const attr = geometry.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
  const arr = attr.array as Uint32Array;
  const n = Math.min(count, ordering.length, arr.length);
  arr.set(ordering.subarray(0, n));
  collapseSortedIndexRanges(attr, n);
  return n;
}

/**
 * Register a single collapsed `[0, max-end)` update range covering `n`
 * fresh entries plus any still-pending ranges (see the identity writer's
 * doc comment for why ranges must never accumulate).
 */
function collapseSortedIndexRanges(attr: THREE.InstancedBufferAttribute, n: number): void {
  let rangeEnd = n;
  for (const range of attr.updateRanges) {
    const end = range.start + range.count;
    if (end > rangeEnd) rangeEnd = end;
  }
  attr.clearUpdateRanges();
  attr.addUpdateRange(0, rangeEnd);
  attr.needsUpdate = true;
}
