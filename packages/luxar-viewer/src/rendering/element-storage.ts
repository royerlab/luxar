/**
 * Shared texture-backed element storage for instanced geometry.
 *
 * Per-element data lives in an RGBA32F **element texture** (gsplats:
 * 4 texels/splat; points: 3 texels/point; lines: 6 texels/segment —
 * layout authority in `./element-texture-layout`) sampled by the vertex
 * shader via `texelFetch`. The only per-instance data is the ordering
 * pair `aSortedIndex` / `aSortedIndexB` (Uint32), which maps the draw
 * slot to a storage slot so draw order can be permuted without
 * rewriting element data (depth-sorting plan,
 * `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md` §4/§8). The pair is
 * DOUBLE-BUFFERED — a new ordering streams into the inactive buffer and
 * the `uSortedIndexSlot` uniform flips on completion (spec §2.1 tier 3),
 * so no frame ever samples a half-applied permutation.
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
// Slices land in the INACTIVE buffer of a double-buffered pair, and the
// `uSortedIndexSlot` uniform flips only once that buffer holds the whole
// permutation (depth-sorting spec §2.1 tier 3). So chunking costs
// nothing in correctness: every rendered frame samples a complete
// ordering.
//
// It did not always. L8 originally streamed into the LIVE attribute and
// accepted the mix `new[0, cursor) ∪ old[cursor, n)` as "transient
// shimmer bounded by ceil(n / chunk) frames". Both halves are valid
// storage slots but the MIX is not a permutation — an element indexed by
// both draws twice and the one it displaced not at all. The bound was
// real; the premise that it stays transient was not. Under a CONTINUOUS
// orbit a new sort arrives about as fast as a stream drains, so the
// mixed state is the steady state: measured on 2026-07-29 at 27–33% of
// sampled frames on a 1.9M-splat volumetric node (27,613 double-drawn)
// and 80% on an 8M-point normal node (59,345), always beginning exactly
// at the chunk boundary. Hence the A/B pair the spec had specced and
// deferred. Only normal/volumetric ever saw it — commutative modes pin
// identity and never receive orderings — but that is precisely the
// population depth sorting exists for.
//
// WebGPU: both WebGPU backends ignore attribute update ranges and
// re-upload the whole buffer on every `needsUpdate`, so SLICING there
// would turn ONE full upload into `ceil(n / chunk)` full uploads (only
// the JS memcpy would be bounded). Slicing is therefore feature-gated to
// the classic WebGL backend via {@link configureSortedIndexChunkedApply}
// (renderer-setup calls it with `apiSurface === 'webgl2'`); the WebGPU
// backends write the ordering in ONE slice and flip on the next pump.
// Double-buffering itself is unconditional — the atomic swap is a
// correctness property, not a per-backend optimisation.
// ────────────────────────────────────────────────────────────────────

/**
 * Per-frame slice size, in Uint32 indices (4 MB per slice). Purely a
 * PACING knob now — no longer a single-shot threshold, because there is
 * no single-shot path: every ordering streams into the inactive buffer
 * and flips when complete, and an ordering that fits one slice simply
 * finishes on its first pump. The baseline showed sort-adjacent tail
 * spikes already at 5M (20 MB) in one shot, so the slice sits well below
 * that — 4 MB @ 1M is the spec §6 exit-criterion size that held target
 * FPS on the Phase-3 perf gate.
 */
export const SORTED_INDEX_CHUNK_ELEMENTS = 1_000_000;

/** Live slice size — test-overridable so unit tests stay tiny. */
let sortedIndexChunkElements = SORTED_INDEX_CHUNK_ELEMENTS;

/** Override the slice size (tests only). `null` restores the default. */
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

/** Outcome of one {@link pumpSortedIndexOrderingApply} call. */
export interface SortedIndexPumpResult {
  /**
   * Slices remain — the caller must keep the on-demand frame loop alive
   * (the per-frame pump is the only thing driving the stream forward).
   */
  more: boolean;
  /**
   * The active slot changed on this call: the newly-written buffer is
   * now the one to draw. The caller must push
   * {@link activeSortedIndexSlot} to the node's materials.
   */
  flipped: boolean;
}

/** `geometry.userData` slot carrying the active ordering-buffer index. */
interface SortedIndexSlotUserData {
  sortedIndexSlot?: 0 | 1;
}

/**
 * Which ordering attribute is currently DRAWN: 0 = `aSortedIndex`,
 * 1 = `aSortedIndexB`. Lives on `geometry.userData` so it survives the
 * pool's release/re-acquire cycle with the buffers it describes.
 */
export function activeSortedIndexSlot(geometry: THREE.InstancedBufferGeometry): 0 | 1 {
  return (geometry.userData as SortedIndexSlotUserData).sortedIndexSlot ?? 0;
}

function setSortedIndexSlot(geometry: THREE.InstancedBufferGeometry, slot: 0 | 1): void {
  (geometry.userData as SortedIndexSlotUserData).sortedIndexSlot = slot;
}

const SLOT_ATTRIBUTE_NAMES = ['aSortedIndex', 'aSortedIndexB'] as const;

/** The ordering attribute currently being drawn. */
export function getActiveSortedIndexAttribute(
  geometry: THREE.InstancedBufferGeometry
): THREE.InstancedBufferAttribute | undefined {
  return geometry.getAttribute(SLOT_ATTRIBUTE_NAMES[activeSortedIndexSlot(geometry)]) as
    | THREE.InstancedBufferAttribute
    | undefined;
}

/** The ordering attribute a new ordering streams into (never drawn). */
function getInactiveSortedIndexAttribute(
  geometry: THREE.InstancedBufferGeometry
): THREE.InstancedBufferAttribute | undefined {
  return geometry.getAttribute(
    SLOT_ATTRIBUTE_NAMES[activeSortedIndexSlot(geometry) === 0 ? 1 : 0]
  ) as THREE.InstancedBufferAttribute | undefined;
}

/**
 * Whether the geometry's ordering pair can safely receive a new
 * ordering: both names present, DISTINCT objects, same length and
 * itemSize.
 *
 * This VALIDATES; it never repairs. An earlier revision materialised a
 * missing/aliased back buffer here instead, and that is precisely the
 * native-WebGPU black-screen bug documented at the allocation site —
 * growing the attribute set behind a cached pipeline shifts every later
 * attribute down a vertex-buffer slot. Repairing here would reintroduce
 * it for exactly the geometries that reach this path.
 *
 * Equal lengths are load-bearing twice over: three derives
 * `_maxInstanceCount` from the SMALLEST instanced attribute (a short
 * back buffer would silently clamp the draw), and the chunk pump clamps
 * each slice against the INACTIVE buffer while the staged count is
 * clamped against the ACTIVE one — so a short back buffer would leave
 * `cursor` permanently below `count`, requesting renders forever and
 * never flipping.
 *
 * `attachElementStorage` satisfies all of this by construction, so every
 * geometry the viewer builds passes. A hand-assembled one that does not
 * simply never sorts, which is the correct failure: a stale-but-whole
 * ordering on screen beats a corrupt one, or a black frame.
 */
function sortedIndexBuffersUsable(geometry: THREE.InstancedBufferGeometry): boolean {
  const front = getActiveSortedIndexAttribute(geometry);
  const back = getInactiveSortedIndexAttribute(geometry);
  if (!front || !back || front === back) return false;
  if (front.itemSize !== back.itemSize) return false;
  return (front.array as Uint32Array).length === (back.array as Uint32Array).length;
}

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
): SortedIndexPumpResult {
  const attr = getInactiveSortedIndexAttribute(geometry);
  if (!attr) {
    // Attribute gone (defensive — release paths cancel first).
    chunkedApplies.delete(geometry);
    return { more: false, flipped: false };
  }
  const arr = attr.array as Uint32Array;
  const start = state.cursor;
  // Backend gate: on WebGL a slice bounds the per-frame memcpy+upload;
  // the WebGPU backends ignore attribute ranges and re-upload the whole
  // buffer per flush, so slicing there would multiply ONE upload into
  // ceil(n / chunk). They write the ordering whole and flip next frame —
  // still atomic, still one upload.
  const sliceEnd = chunkedApplyEnabled ? start + sortedIndexChunkElements : state.count;
  const end = Math.min(state.count, sliceEnd, arr.length);
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

  if (state.cursor < state.count) return { more: true, flipped: false };

  // === The atomic swap ===
  // The inactive buffer now holds the WHOLE new permutation, so it is
  // safe to draw from. Flipping the slot is a single uniform write: no
  // rebind, no recompile, and no frame ever samples a half-written
  // ordering. The caller pushes the new slot to the materials, and the
  // per-frame callback that drives this pump runs BEFORE the render, so
  // this slice's upload and the flip land in the same frame.
  setSortedIndexSlot(geometry, activeSortedIndexSlot(geometry) === 0 ? 1 : 0);

  if (state.pending) {
    // Start the held newest ordering into what just became the inactive
    // buffer. Held rather than restarted mid-stream on purpose: under a
    // continuous orbit a restart-on-arrival never converges (measured as
    // constant jank when L8 shipped) — whereas finishing, flipping, then
    // starting the newest always converges and always shows a complete
    // order.
    state.ordering = state.pending.ordering;
    state.count = state.pending.count;
    state.cursor = 0;
    state.pending = null;
    return { more: true, flipped: true };
  }
  chunkedApplies.delete(geometry);
  return { more: false, flipped: true };
}

/**
 * True while an ordering application is in flight for `geometry`
 * (including while a held newest ordering is waiting its turn).
 *
 * This is NOT a dispatch gate. It once was — no new sort was dispatched
 * for a node while a stream ran, because a stream wrote into the LIVE
 * attribute and sorting faster than it could apply just churned held
 * orderings. Now that a stream writes into the inactive buffer, the
 * displayed ordering is complete throughout, so sorting and applying
 * proceed CONCURRENTLY; the only thing bounding sort traffic is the
 * one-in-flight-per-node rule.
 */
export function hasPendingSortedIndexOrderingApply(
  geometry: THREE.InstancedBufferGeometry
): boolean {
  return chunkedApplies.has(geometry);
}

/**
 * Apply the next slice for `geometry` (one call per rendered frame — the
 * depth-sort coordinator's per-frame scheduler is the driver), and flip
 * the active slot when the slice completes the ordering.
 *
 * The caller must honour BOTH result fields: request another frame while
 * `more`, and push {@link activeSortedIndexSlot} to the node's materials
 * on `flipped`.
 */
export function pumpSortedIndexOrderingApply(
  geometry: THREE.InstancedBufferGeometry
): SortedIndexPumpResult {
  const state = chunkedApplies.get(geometry);
  if (!state) return { more: false, flipped: false };
  return applyNextSortedIndexChunk(geometry, state);
}

/**
 * Abort an in-flight application — the streaming ordering AND any held
 * newest ordering.
 *
 * Safe by construction now that a stream writes into the INACTIVE
 * buffer: abandoning one mid-slice discards a partially-written buffer
 * that was never drawn, and the displayed ordering is untouched. (Before
 * double-buffering this left the live attribute holding a mix of old and
 * new indices.) Callers: both identity writers (the commit path),
 * LOD-demotion detection in the coordinator's pump, node release,
 * geometry dispose.
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
  // Slot B is a DISTINCT buffer from the very first frame — never an
  // alias onto slot A, and never lazily materialised.
  //
  // Every shader references both names, so both attributes must exist on
  // every geometry (WebGPU's `RenderObject.getAttributes` dereferences a
  // graph-referenced attribute before its undefined guard, so a missing
  // one throws rather than degrading). Aliasing the pair onto one
  // attribute object satisfies existence for free, and an earlier
  // revision did exactly that — splitting on a node's first sort so a
  // commutative-mode node paid no extra bytes. That is WRONG on the
  // native WebGPU backend, and silently so:
  //
  //   `WebGPUAttributeUtils.createShaderVertexBuffers` keys the vertex
  //   buffer LAYOUT by BufferAttribute IDENTITY, so an aliased pair
  //   compiles to ONE instanced buffer carrying two shader locations at
  //   offset 0, and a split pair to TWO. But three rebuilds a render
  //   pipeline only on a material/cache-key change: `getGeometryCacheKey`
  //   hashes attribute NAMES, itemSize and normalized — never identity —
  //   and `RenderObjects.get` responds to `needsGeometryUpdate` with a
  //   bare `setGeometry()` that refreshes the attribute list but leaves
  //   `Pipelines`' cached pipeline alone. So after the split the draw
  //   binds three vertex buffers into a two-buffer layout: every
  //   subsequent attribute shifts down a slot and the quad-corner
  //   attribute reads the ordering buffer's u32s as vec2<f32>. Corners
  //   collapse to denormals, every quad degenerates, and the scene
  //   renders BLACK — no validation error, no console warning.
  //
  // WebGL is immune (it binds attributes by program location, not by
  // ordinal slot), which is why the whole unit + TSL-parity suite stayed
  // green: `tsl-shader-parity` runs `WebGPURenderer({ forceWebGL: true })`,
  // which exercises the WGSL-adjacent node graph through the WebGL2
  // bridge and never builds a WebGPU vertex layout at all.
  //
  // Allocating both buffers here makes the attribute SET, and therefore
  // the vertex layout, invariant for the geometry's whole lifetime — the
  // property three's pipeline cache assumes but does not enforce. It
  // costs a flat +4 B/element on nodes that never sort (≈6% of a
  // gsplat's 68 B/element), which is the honest price of not depending
  // on a cache-invalidation path that does not exist.
  const sortedIndexB = new THREE.InstancedBufferAttribute(new Uint32Array(capacity), 1);
  sortedIndexB.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('aSortedIndexB', sortedIndexB);

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
 *
 * Also re-homes the geometry on SLOT 0, which is what makes the split
 * ownership of the swap safe. The slot lives on the geometry (so it
 * survives the pool's release/re-acquire with the buffers it describes)
 * while the selector uniform lives on the material, and the pool re-pairs
 * the two freely: a geometry a sorted tenant left on slot 1, handed to a
 * node whose fresh material defaults to slot 0, would draw the PREVIOUS
 * tenant's permutation over a different element count — and an
 * order-independent tenant is never tracked by the coordinator, so
 * nothing would push the slot to its uniform. A full identity write is
 * precisely the fresh-start signal, so normalising here means a default
 * uniform is always correct and untracked nodes need no sync at all.
 *
 * The append writer ({@link writeSortedIndexIdentityRange}) deliberately
 * does NOT do this: it preserves the prefix permutation in whichever
 * buffer is live.
 */
export function writeSortedIndexIdentity(
  geometry: THREE.InstancedBufferGeometry,
  count: number
): void {
  // A fresh commit supersedes any in-flight chunked ordering apply —
  // its remaining slices belong to the OLD element population and
  // would scribble a stale permutation over the identity just written.
  cancelSortedIndexOrderingApply(geometry);
  // Normalise to slot 0 BEFORE picking the target, so identity always
  // lands in the buffer a default-valued selector uniform reads.
  setSortedIndexSlot(geometry, 0);
  const attr = getActiveSortedIndexAttribute(geometry);
  if (!attr) return;
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
  // Identity targets the ACTIVE buffer: it is a complete permutation by
  // construction, so writing it live is safe and needs no flip.
  const attr = getActiveSortedIndexAttribute(geometry);
  if (!attr) return;
  const arr = attr.array as Uint32Array;
  const n = Math.min(count, arr.length);
  for (let i = Math.max(0, from); i < n; i++) arr[i] = i;
  collapseSortedIndexRanges(attr, n);
}

/**
 * Stage a depth-sort permutation (the SortWorker's back-to-front
 * ordering, depth-sorting Phase 2). Returns the number of entries that
 * WILL be written, clamped to both the ordering's and the attribute's
 * length.
 *
 * The ordering is never written to the buffer being drawn. It streams
 * into the INACTIVE one and the slot flips when that buffer holds the
 * WHOLE permutation (depth-sorting spec §2.1 tier 3), so every rendered
 * frame samples a complete ordering. There is no size threshold and no
 * single-shot branch: one path, and a half-applied ordering — which is
 * not a reordering but a corrupt permutation, drawing elements twice
 * and omitting as many — is structurally impossible.
 *
 * This call only STAGES: {@link pumpSortedIndexOrderingApply} (driven by
 * the coordinator's per-frame callback) owns every slice and the flip.
 * That costs nothing in latency — a resolve lands between frames, so the
 * first slice still rides the very next render — and it keeps the
 * per-frame upload bound strict.
 *
 * A newer ordering arriving mid-stream is HELD (at most one; latest
 * wins) and started after the current stream completes, rather than
 * restarting it: under a continuous orbit a restart-on-arrival never
 * converges. The coordinator's generation guard ensures only
 * current-generation orderings reach this writer.
 */
export function writeSortedIndexOrdering(
  geometry: THREE.InstancedBufferGeometry,
  ordering: Uint32Array,
  count: number
): number {
  const active = getActiveSortedIndexAttribute(geometry);
  if (!active) return 0;
  // A TRUNCATED ordering is not a permutation of the drawn population,
  // and staging it would flip the slot with the tail left holding
  // whatever the inactive buffer happened to contain (zeros on a freshly
  // attached geometry) — elements drawn several times, others not at
  // all. That is the exact corruption double-buffering exists to
  // prevent, so drop it rather than clamp. Matches the coordinator's
  // apply-invariant, which only ever passes `ordering.length` as `count`.
  if (ordering.length < count) return 0;
  // Clamping the other way is pure memory safety: capacity is always
  // >= instanceCount, so a count above it still covers every drawn
  // element.
  const n = Math.min(count, (active.array as Uint32Array).length);
  // An EMPTY ordering stages nothing. Otherwise it would "complete" on its
  // first pump and FLIP — swapping the newest ordering out for the older
  // buffer sitting behind it. (The commit path already returns early on an
  // empty frame, so this guards the writer's own contract rather than a
  // live caller.)
  if (n === 0) return 0;
  // Malformed pair — never repair it here (see `sortedIndexBuffersUsable`).
  if (!sortedIndexBuffersUsable(geometry)) return 0;

  const inFlight = chunkedApplies.get(geometry);
  if (inFlight) {
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
