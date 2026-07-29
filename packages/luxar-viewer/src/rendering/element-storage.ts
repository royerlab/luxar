/**
 * Shared texture-backed element storage for instanced geometry.
 *
 * Per-element data lives in an RGBA32F **element texture** (gsplats:
 * 4 texels/splat; points: 3 texels/point; lines: 6 texels/segment —
 * layout authority in `./element-texture-layout`) sampled by the vertex
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

// ────────────────────────────────────────────────────────────────────
// Chunked ordering application (perf lever L8)
//
// A depth-sort resolve on a large node used to apply its whole
// permutation in one shot: a full-array `set()` memcpy plus ONE
// `[0, n)` update range that the classic WebGLRenderer uploads in a
// single `bufferSubData` on the next flush. At 10M splats that is a
// 40 MB upload + 40 MB memcpy on the GL thread — measured on the
// campaign baseline (Mac M4 Max, synthetic 10M-splat orbit) as
// sort-adjacent frame p99 of 119–563 ms vs ~27–35 ms idle-orbit p99.
// Even 5M (20 MB) showed sort-adjacent tail spikes on some runs.
//
// Large orderings are therefore applied CHUNKED: one slice of
// `SORTED_INDEX_CHUNK_ELEMENTS` indices per rendered frame (write the
// slice + register just its update range), driven by a per-frame pump
// (the depth-sort coordinator's scheduler). 1M indices = 4 MB/frame,
// which by the baseline's linear-in-bytes scaling bounds the added
// per-frame cost to roughly 1/10th of the 10M stall (~12–56 ms worst
// case → low-single-digit ms typical) instead of one 119–563 ms hitch.
//
// CORRECTNESS TRADE (documented, deliberate): mid-application the
// attribute holds `new ordering[0, cursor) ∪ old ordering[cursor, n)`.
// Both halves are valid storage-slot indices, but the MIX is not a
// permutation: an element indexed by both halves draws twice and the
// one it displaced draws zero times. Visually that is transient
// shimmer of the same class as the stale-order frames every 3DGS
// renderer shows between camera move and sort resolve — strictly
// bounded by `ceil(n / chunk)` frames (≤ 10 frames at 10M), after
// which the buffer EXACTLY equals the new ordering. Order-independent
// (additive/commutative) modes never receive orderings (identity is
// pinned), so only normal/volumetric see it. The pick path shares the
// same attribute: a transiently duplicated index means two instances
// briefly resolve to the same element id — a hover mid-shimmer can
// pick either duplicate, equally acceptable and equally transient.
//
// WebGPU: both WebGPU backends ignore attribute update ranges and
// re-upload the whole buffer on every `needsUpdate`, so chunking there
// would turn ONE full upload into `ceil(n / chunk)` full uploads
// (only the JS memcpy would be bounded). Chunking is therefore
// feature-gated to the classic WebGL backend via
// {@link configureSortedIndexChunkedApply} (renderer-setup calls it
// with `apiSurface === 'webgl2'`); WebGPU keeps the single-shot path.
// ────────────────────────────────────────────────────────────────────

/**
 * Chunk size AND single-shot threshold, in Uint32 indices (4 MB per
 * chunk). Threshold = chunk size: an ordering that fits one chunk
 * gains nothing from deferral (the chunked path would apply it in one
 * slice anyway), and the baseline showed sort-adjacent tail spikes
 * already at 5M (20 MB) single-shot uploads, so the cutoff sits well
 * below that — a 1M single-shot upload is the spec §6 exit-criterion
 * size (4 MB @ 1M) that held target FPS on the Phase-3 perf gate.
 */
export const SORTED_INDEX_CHUNK_ELEMENTS = 1_000_000;

/** Live chunk size — test-overridable so unit tests stay tiny. */
let sortedIndexChunkElements = SORTED_INDEX_CHUNK_ELEMENTS;

/** Override the chunk size/threshold (tests only). `null` restores the default. */
export function setSortedIndexChunkElementsForTests(elements: number | null): void {
  sortedIndexChunkElements = elements ?? SORTED_INDEX_CHUNK_ELEMENTS;
}

/**
 * Session backend gate (renderer-setup, the
 * `configureElementTextureLayout` pattern): `true` on the classic
 * WebGL backend (partial attribute uploads honored), `false` on the
 * WebGPU backends (ranges ignored — chunking would multiply full
 * uploads; see the module note above). Defaults to `true`: classic
 * WebGL is the production default and headless/unit contexts have no
 * renderer to misbehave.
 */
let chunkedApplyEnabled = true;

/** Configure whether large orderings apply chunked (classic WebGL only). */
export function configureSortedIndexChunkedApply(enabled: boolean): void {
  chunkedApplyEnabled = enabled;
}

/** In-flight chunked application state for one geometry. */
interface ChunkedOrderingApply {
  /** The full new ordering (retained until completion/cancel). */
  ordering: Uint32Array;
  /** Total entries to write (pre-clamped against ordering + attribute). */
  count: number;
  /** Next unwritten index — `[0, cursor)` already holds the new ordering. */
  cursor: number;
  /**
   * At most ONE held newer ordering (latest wins — an even newer arrival
   * replaces it). Started only after the CURRENT stream completes; a
   * mid-apply restart would let a continuous orbit (new orderings every
   * sort round-trip) keep the stream perpetually at slice 0 — measured
   * as an 8× frame-median regression (~65 ms constant jank: nearly every
   * frame became a chunk-upload frame and the buffer never converged).
   * A fully-applied slightly-stale order is strictly better than a
   * never-completing mix. Dropped with the whole entry on every
   * cancellation path (identity write, demotion, release, dispose).
   */
  pending: { ordering: Uint32Array; count: number } | null;
}

/**
 * Active chunked applies, keyed by geometry (at most one per geometry —
 * a new ordering replaces the previous apply wholesale). A plain Map is
 * iterable for teardown sweeps; entries are removed on completion and on
 * every cancellation path (new ordering, identity write, geometry
 * dispose, coordinator release), so nothing lingers.
 */
const chunkedApplies = new Map<THREE.InstancedBufferGeometry, ChunkedOrderingApply>();

/** Geometries whose dispose listener already cancels chunked applies. */
const chunkedDisposeHooked = new WeakSet<THREE.InstancedBufferGeometry>();

/**
 * Write the next pending slice into `aSortedIndex` and register ITS
 * update range (not the collapsed `[0, …)` prefix — re-registering the
 * whole prefix each frame would upload 4+8+…+4·k MB instead of 4 MB/
 * frame, defeating the chunking). Range discipline still honors the
 * never-accumulate rule: any pending ranges (previous chunk not yet
 * flushed — hidden mesh, coalesced frames) are folded WITH the new
 * slice into ONE contiguous span (chunks are consecutive, and every
 * other writer registers a `[0, n)` prefix, so the union is always
 * contiguous and the array data under it is always current — a
 * superset upload is correct, never stale).
 *
 * Returns true when more chunks remain after this one.
 */
function applyNextSortedIndexChunk(
  geometry: THREE.InstancedBufferGeometry,
  state: ChunkedOrderingApply
): boolean {
  const attr = geometry.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute | undefined;
  if (!attr) {
    // Attribute gone (defensive — release paths cancel first).
    chunkedApplies.delete(geometry);
    return false;
  }
  const arr = attr.array as Uint32Array;
  const start = state.cursor;
  const end = Math.min(state.count, start + sortedIndexChunkElements, arr.length);
  arr.set(state.ordering.subarray(start, end), start);
  state.cursor = end;

  let rangeStart = start;
  let rangeEnd = end;
  for (const range of attr.updateRanges) {
    if (range.start < rangeStart) rangeStart = range.start;
    const rEnd = range.start + range.count;
    if (rEnd > rangeEnd) rangeEnd = rEnd;
  }
  attr.clearUpdateRanges();
  attr.addUpdateRange(rangeStart, rangeEnd - rangeStart);
  attr.needsUpdate = true;

  if (state.cursor >= state.count) {
    if (state.pending) {
      // Promote the held newest ordering: the buffer EXACTLY equals the
      // just-completed ordering for this frame (a fully valid
      // permutation renders), and the next pump starts the new stream.
      state.ordering = state.pending.ordering;
      state.count = state.pending.count;
      state.cursor = 0;
      state.pending = null;
      return true;
    }
    chunkedApplies.delete(geometry);
    return false;
  }
  return true;
}

/**
 * True while a chunked ordering application is in flight for `geometry`
 * (including while a held newest ordering is waiting its turn). Doubles
 * as the coordinator's DISPATCH GATE: no new sorts are dispatched for a
 * node while this is true — sorting faster than the stream can apply
 * just churns held orderings (the natural cadence is sort → apply N
 * frames → next sort).
 */
export function hasPendingSortedIndexOrderingApply(
  geometry: THREE.InstancedBufferGeometry
): boolean {
  return chunkedApplies.has(geometry);
}

/**
 * Apply the next pending slice for `geometry` (one call per rendered
 * frame — the depth-sort coordinator's per-frame scheduler is the
 * driver). Returns true when more slices remain (the caller should
 * request another frame); false when the application completed this
 * call or none was pending.
 */
export function pumpSortedIndexOrderingApply(geometry: THREE.InstancedBufferGeometry): boolean {
  const state = chunkedApplies.get(geometry);
  if (!state) return false;
  return applyNextSortedIndexChunk(geometry, state);
}

/**
 * Abort an in-flight chunked application — the streaming ordering AND
 * any held newest ordering — leaving the attribute as-is (a valid mix
 * of old/new indices — same transient class as mid-application frames;
 * the caller is about to overwrite it or has released the geometry).
 * Callers: small single-shot ordering writes, both identity writers
 * (the commit path), LOD-demotion detection in the coordinator's pump,
 * node release, geometry dispose.
 */
export function cancelSortedIndexOrderingApply(geometry: THREE.InstancedBufferGeometry): void {
  chunkedApplies.delete(geometry);
}

/**
 * Abort every in-flight chunked application (dataset-switch teardown /
 * app dispose — the coordinator's `releaseAllDepthSortNodes` /
 * `disposeDepthSort` sweeps).
 */
export function cancelAllSortedIndexOrderingApplies(): void {
  chunkedApplies.clear();
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
  // Ordering slot B starts ALIASED to slot A (same attribute object, so
  // zero extra bytes). Every shader references both names, so the
  // attribute must exist on every geometry — WebGPU's RenderObject
  // dereferences a graph-referenced attribute before its undefined
  // guard, so a missing one throws rather than degrading. Aliasing
  // satisfies that from this single chokepoint while a node that never
  // sorts (any commutative blending mode) pays nothing;
  // `ensureSortedIndexBackBuffer` splits the alias on first ordering.
  geometry.setAttribute('aSortedIndexB', sortedIndex);

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
  // A fresh commit supersedes any in-flight chunked ordering apply —
  // its remaining slices belong to the OLD element population and
  // would scribble a stale permutation over the identity just written.
  cancelSortedIndexOrderingApply(geometry);
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
  // Same supersede rule as writeSortedIndexIdentity: the append commit
  // invalidates the ordering an in-flight chunked apply was streaming
  // (the coordinator re-sorts the grown population on a later frame).
  cancelSortedIndexOrderingApply(geometry);
  const attr = geometry.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
  const arr = attr.array as Uint32Array;
  const n = Math.min(count, arr.length);
  for (let i = Math.max(0, from); i < n; i++) arr[i] = i;
  collapseSortedIndexRanges(attr, n);
}

/**
 * Write a depth-sort permutation into `aSortedIndex[0..count)` (the
 * SortWorker's back-to-front ordering, depth-sorting Phase 2). Returns
 * the number of entries that WILL be written (clamped to both the
 * ordering's and the attribute's length).
 *
 * Routing (perf lever L8 — see the chunked-apply module note):
 * - `n ≤ SORTED_INDEX_CHUNK_ELEMENTS` or chunking disabled (WebGPU
 *   backends): single-shot — full write + the same collapsed-prefix
 *   update-range discipline as {@link writeSortedIndexIdentity}.
 * - Larger orderings apply CHUNKED, and the per-frame pump
 *   ({@link pumpSortedIndexOrderingApply} — the depth-sort
 *   coordinator's scheduler) owns EVERY slice: this call only records
 *   the pending state (the attribute is untouched — still the previous
 *   fully-valid permutation), and each subsequent rendered frame
 *   writes one slice until the buffer EXACTLY equals the ordering.
 *   Deferring slice 1 too keeps the per-frame bound strict — a
 *   synchronous first slice would fold into frame 1's pump slice for
 *   a double-size first upload — and means a never-pumped ordering
 *   degrades to "stale but valid", not "mixed". Mid-application
 *   frames render a bounded old/new mix — the documented
 *   transient-duplication trade.
 *
 * A NEW large ordering arriving while an apply is streaming does NOT
 * restart the stream: it is HELD (at most one — latest wins, an older
 * held ordering is dropped) and starts only after the current apply
 * completes (see ChunkedOrderingApply.pending for why restarting never
 * converges under a continuous orbit). A SMALL (single-shot) ordering
 * cancels the stream instead — the full write leaves the buffer
 * exactly equal to the newest ordering, which dominates anything the
 * stream could still produce. The coordinator's generation guard
 * ensures only current-generation orderings reach this writer either
 * way.
 */
export function writeSortedIndexOrdering(
  geometry: THREE.InstancedBufferGeometry,
  ordering: Uint32Array,
  count: number
): number {
  const attr = geometry.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
  const arr = attr.array as Uint32Array;
  const n = Math.min(count, ordering.length, arr.length);
  if (!chunkedApplyEnabled || n <= sortedIndexChunkElements) {
    cancelSortedIndexOrderingApply(geometry);
    arr.set(ordering.subarray(0, n));
    collapseSortedIndexRanges(attr, n);
    return n;
  }

  const inFlight = chunkedApplies.get(geometry);
  if (inFlight) {
    // Hold-latest: never restart a streaming apply (see the pending
    // field's doc); the newest ordering waits its turn.
    inFlight.pending = { ordering, count: n };
    return n;
  }

  const state: ChunkedOrderingApply = { ordering, count: n, cursor: 0, pending: null };
  chunkedApplies.set(geometry, state);
  if (!chunkedDisposeHooked.has(geometry)) {
    chunkedDisposeHooked.add(geometry);
    // Structural cancellation on geometry death (mirrors the element
    // texture's dispose-listener lifetime pin): a disposed geometry
    // must not be kept alive by the applies map, nor pumped again.
    geometry.addEventListener('dispose', () => cancelSortedIndexOrderingApply(geometry));
  }
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
