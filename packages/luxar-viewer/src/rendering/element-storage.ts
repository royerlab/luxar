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
 * so no frame ever samples a half-applied permutation. An ordering
 * reaches the inactive buffer from one of two producers: a HOST array the
 * pump uploads ({@link writeSortedIndexOrdering}), or a DEVICE writer that
 * fills the GPU buffer itself ({@link writeDeviceSortedIndexOrdering}, the
 * WebGPU compute sort of #3162). The flip, acknowledgement, held-draw and
 * supersede rules are shared.
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
import { rendererBackendPolicy } from './renderer-backend-policy';
import { log, Modules } from '../utils/log';

// Three's WebGPU backend recognizes this flag; inherit from the public
// instance so the eager WebGL bundle keeps one BufferAttribute class graph.
class StorageInstancedBufferAttribute extends THREE.InstancedBufferAttribute {
  readonly isStorageInstancedBufferAttribute = true;
}

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
 * - Ranges accumulate across commits while a mesh is hidden (they are
 *   consumed only when the texture is actually uploaded: by the classic
 *   WebGLRenderer at flush, and on both WebGPU backends by
 *   `./element-texture-row-upload`). So every call collapses the pending
 *   set into one contiguous span and re-splits it (a superset upload is
 *   always correct, never
 *   stale; our writers only ever register a `[0, n)` prefix or an append
 *   suffix contiguous with it).
 * - An element is exactly `floatsPerElement` floats and the texture width
 *   is a multiple of the layout's texels-per-element, so elements never
 *   straddle rows — each emitted range stays within one row (the WebGL
 *   path uploads every range with `height = 1` and would reject a
 *   row-straddling range with INVALID_VALUE).
 *
 * three's WebGPU backends (native + WebGL2-fallback) ignore texture
 * update ranges; `./element-texture-row-upload` wraps their
 * `updateTexture` so element textures upload exactly the rows these
 * ranges cover there too (#2944).
 */
/**
 * Textures with a FULL-image upload pending (encoded to three as
 * `needsUpdate` + EMPTY `updateRanges`). The empty-ranges encoding is
 * invisible to the range fold below — without this set, a later append
 * would register only its own span and silently DOWNGRADE the pending
 * full upload to a partial one, leaving the prefix rendering the
 * previous commit's texels (found by model-based fuzzing; deterministic
 * repro: full write ≥75% of rows → append before any flush). Cleared by
 * `texture.onUpdate`, which the classic renderer and three's common
 * `Textures.updateTexture` (both WebGPU backends) invoke after they
 * actually consume the upload.
 */
const pendingFullUpload = new WeakSet<THREE.DataTexture>();

/**
 * Every texture {@link attachElementStorage} allocated. The WebGPU row
 * upload (`./element-texture-row-upload`) acts ONLY on these — every
 * write to them registers its dirty span here, which is what makes a
 * rows-only upload equal to a full one.
 */
const elementTextures = new WeakSet<THREE.Texture>();

/** True for a texture allocated by {@link attachElementStorage}. */
export function isElementTexture(texture: THREE.Texture): texture is THREE.DataTexture {
  return elementTextures.has(texture);
}

/** True while a FULL image upload is pending on an element texture. */
export function hasPendingElementTextureFullUpload(texture: THREE.DataTexture): boolean {
  return pendingFullUpload.has(texture);
}

/**
 * Element textures some renderer has consumed an upload of (its
 * `onUpdate` fired). Until then the pending ranges cover every texel
 * ever written since {@link attachElementStorage} zero-filled the
 * backing store, so a ranged FIRST upload into zero-initialised GPU
 * storage equals a full one — what the classic renderer already does.
 */
const uploadedOnce = new WeakSet<THREE.DataTexture>();
const uploadedVersions = new WeakMap<THREE.DataTexture, number>();

/** True once any renderer has consumed an upload of this element texture. */
export function hasElementTextureBeenUploaded(texture: THREE.DataTexture): boolean {
  return uploadedOnce.has(texture);
}

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
  elementTextureLayout?: ElementTextureLayout;
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
// The pump is BACK-PRESSURED so this bound survives when the mesh is not
// drawn every frame (issue #715). Each written slice arms a per-attribute
// flag (`sliceUploadPending`) that is cleared by the attribute's
// `onUploadCallback` — which three fires from BOTH GPU-upload paths
// (`createBuffer` for the first upload, `updateBuffer` for later ranged
// ones). While the flag is set the previous slice has not reached the GPU
// (hidden mesh, frustum-cull, background tab). Writing the next slice
// anyway would union it onto the still-pending one (see
// `applyNextSortedIndexChunk`), so ten intended 4 MB slices would collapse
// into one 40 MB upload the moment the object became drawable — the very
// hitch chunking removes. Instead the pump STALLS until the upload is
// acknowledged: it advances at most one slice per DRAWN frame, and the
// coordinator pauses a hidden node's stream entirely. (An earlier revision
// keyed on `updateRanges.length`, which DEADLOCKS: `createBuffer` uploads
// the whole buffer without clearing the ranges, so a node first drawn
// while its stream was mid-flight would stall forever.) Double-buffering
// keeps this safe — the drawn buffer is always a complete permutation, so
// a held stream is never observed.
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
// WebGPU: slicing is limited to the classic WebGL backend by the backend
// policy (`renderer-backend-policy.ts`, `ordering.chunked`), because its
// back-pressure waits on the attribute's `onUploadCallback`, which neither
// WebGPU backend fires. The WebGPU backends write the ordering in ONE slice
// and flip on the next pump. They DO honour attribute `updateRanges` (and
// clear them after the upload); the pair's buffer usage there is
// `StaticDrawUsage` (`ordering.usage`), so it uploads only when a write bumps
// its version.
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

/** True only for geometries built with storage-capable ordering buffers. */
export function canComputeSortedIndex(geometry: THREE.InstancedBufferGeometry): boolean {
  const a = geometry.getAttribute('aSortedIndex');
  const b = geometry.getAttribute('aSortedIndexB');
  return (
    !!a &&
    !!b &&
    a !== b &&
    'isStorageInstancedBufferAttribute' in a &&
    'isStorageInstancedBufferAttribute' in b
  );
}

/**
 * A GPU-side producer of one ordering (the WebGPU compute sort, #3162): it
 * fills an ordering attribute's GPU buffer directly instead of handing the
 * pump a CPU array to upload.
 */
export interface DeviceOrderingWrite {
  /** Elements the ordering covers: the committed population it was computed for. */
  readonly count: number;
  /**
   * Write the ordering into `target`'s GPU buffer. The pump calls it once, with
   * the INACTIVE attribute, right before the flip. Return false when the
   * ordering can no longer be written (its population was superseded, the
   * device went away); that abandons it. Must not touch `target.array` or bump
   * its version: three would then upload the host array over the result. The
   * pump waits for any earlier host upload on `target` before calling this.
   */
  write(target: THREE.InstancedBufferAttribute): boolean;
}

/**
 * What one staged ordering carries. A HOST ordering is a CPU array (the
 * SortWorker's, or the commit-time sort's) that the pump copies or adopts into
 * the inactive attribute, slice by slice on classic WebGL. A DEVICE ordering
 * writes the inactive attribute's GPU buffer in one step. Everything after the
 * write (the flip, the held-draw raise, the draw acknowledgement, supersede and
 * cancel) is the same for both, in {@link completeStagedOrdering}.
 */
type StagedOrdering =
  | {
      readonly kind: 'host';
      /** The full new ordering (retained until completion/cancel). */
      readonly ordering: Uint32Array;
      /** The SortWorker transferred this array; the caller will not mutate it. */
      readonly adopt: boolean;
      /** Total entries to write (pre-clamped against ordering + attribute). */
      readonly count: number;
    }
  | { readonly kind: 'device'; readonly op: DeviceOrderingWrite; readonly count: number };

/** A staged ordering plus its lifecycle hooks (issue #713). */
interface StagedOrderingEntry {
  staged: StagedOrdering;
  callbacks: SortedIndexApplyCallbacks | null;
}

/** In-flight application state for one geometry. */
interface OrderingApply extends StagedOrderingEntry {
  /** Next unwritten index of a host ordering: `[0, cursor)` already holds it. */
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
   * Carries its own lifecycle hooks (issue #713): if this held ordering
   * is superseded before it ever streams, its `onAbandoned` fires.
   */
  pending: StagedOrderingEntry | null;
}

/**
 * Active chunked applies, keyed by geometry (at most one per geometry —
 * a new ordering replaces the previous apply wholesale). A plain Map is
 * iterable for teardown sweeps; entries are removed on completion and on
 * every cancellation path (new ordering, identity write, geometry
 * dispose, coordinator release), so nothing lingers.
 */
const orderingApplies = new Map<THREE.InstancedBufferGeometry, OrderingApply>();

/**
 * Orderings whose slot has flipped but whose geometry has not yet completed a
 * render with that slot. A flip only SELECTS the buffer; THREE consumes the
 * pending attribute ranges and issues the draw later in the render pass. The
 * mesh's `onAfterRender` hook acknowledges that point through
 * {@link acknowledgeSortedIndexOrderingDraw}.
 */
const awaitingDrawAcknowledgement = new Map<
  THREE.InstancedBufferGeometry,
  SortedIndexApplyCallbacks
>();

/** Geometries whose dispose listener already cancels chunked applies. */
const applyDisposeHooked = new WeakSet<THREE.InstancedBufferGeometry>();

/**
 * Lifecycle hooks for one staged ordering (the depth-sort coordinator wires
 * these to its 'Depth Sort' profiler session so the pass spans the whole
 * dispatch→applied lifecycle — issue #713). EXACTLY ONE fires per accepted
 * ordering: `onApplied` after THREE renders the geometry with its selected
 * buffer (post-upload/draw acknowledgement), or `onAbandoned` when it is
 * dropped before that render (superseded as a held pending or selected slot,
 * identity-written over, geometry disposed, node released, dataset switched,
 * app disposed). Never both, never neither.
 * They fire ONLY when {@link writeSortedIndexOrdering} accepts the ordering
 * (returns > 0); a rejected write leaves session ownership with the caller.
 */
export interface SortedIndexApplyCallbacks {
  onApplied?: () => void;
  onAbandoned?: () => void;
  /**
   * Who staged this ordering. Opaque here; the per-owner teardown sweep
   * ({@link cancelAllSortedIndexOrderingApplies} with an owner) filters on it,
   * so one depth-sort coordinator's dataset switch or dispose never abandons
   * another coordinator's streams on the same page.
   */
  owner?: object;
  /**
   * "Request one more frame" after a slice upload or a drawn frame that
   * flushed a device writer's target. A failed device write also uses it
   * when releasing a held append draw. The coordinator does not spin on a
   * stall while the mesh is culled; a real draw wakes its next pump.
   * Per ordering rather than module-wide so it wakes the owning render loop.
   */
  requestFrame?: () => void;
  /**
   * When it returns true, `applyNextSortedIndexChunk` advances past the
   * #715 upload-acknowledgement back-pressure. The depth-sort coordinator's
   * offline-capture drain (`resortForCapture`) turns it on for its own
   * streams: that drain never draws, so the ack that releases the stall can
   * never fire and a multi-slice stream (>1 slice, e.g. a 3M-element node)
   * would wedge the drain until its timeout on every captured frame. The
   * stall exists only to bound the per-DRAWN-frame upload for interactive
   * smoothness; offline, folding the slices into one upload on the capture's
   * own render is exactly acceptable, and the range-union discipline keeps
   * the pending span contiguous and current (a superset upload is correct,
   * never stale).
   */
  bypassBackPressure?: () => boolean;
}

/**
 * Ordering attributes whose last-written slice has NOT yet been uploaded
 * to the GPU — the back-pressure signal for chunked apply (issue #715).
 *
 * The mesh may be hidden, frustum-culled, in a background tab, or
 * otherwise not drawn, in which case the previous slice's `bufferSubData`
 * never ran; advancing anyway would fold each new slice into the pending
 * one (see {@link applyNextSortedIndexChunk}), collapsing ten intended
 * 4 MB slices into one 40 MB upload that hits the GPU in a single frame
 * when the object becomes drawable — the exact hitch chunking removes.
 *
 * The flag is set when a slice is written and CLEARED by the attribute's
 * `onUploadCallback`, which three invokes from BOTH `WebGLAttributes`
 * upload paths — `createBuffer` (the attribute's very first upload) AND
 * `updateBuffer` (every later ranged upload). `updateRanges.length` is
 * NOT usable for this: `createBuffer` uploads the whole buffer without
 * clearing the ranges and caches the version, so the first draw of a
 * node committed while culled would leave the ranges lingering forever —
 * a permanent stall (three r184, verified). This mirrors the element
 * texture's `pendingFullUpload` + `texture.onUpdate` acknowledgement.
 */
const sliceUploadPending = new WeakSet<THREE.InstancedBufferAttribute>();

/**
 * Ordering attributes whose GPU buffer a DEVICE ordering wrote after the last
 * host upload from index 0: their host array is BEHIND the buffer and may not
 * even be a permutation of the population it draws (see
 * {@link writeDeviceOrdering}). Two readers must not trust it: the grown-
 * geometry seed ({@link isSortedIndexHostArrayCurrent}) and a context or device
 * restore, which re-uploads the host arrays ({@link repairSortedIndexHostArray}).
 * A host write that uploads from index 0 ({@link collapseSortedIndexRanges})
 * brings it current again.
 */
const hostArrayBehind = new WeakSet<THREE.InstancedBufferAttribute>();

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
  /**
   * This call made NO progress: the previous slice's upload has not been
   * acknowledged (its `onUploadCallback` has not fired — the mesh was
   * hidden, frustum-culled, or otherwise not drawn), so advancing would
   * union the next slice onto the still-pending one and grow the pending
   * upload without bound (issue #715). `more` stays
   * `true` and `flipped` stays `false`; the caller must NOT spin the
   * on-demand loop on a stall (a subsequent DRAWN frame consumes the
   * pending upload and wakes the next pump). A device writer also stalls on
   * WebGPU when its target still has a pending host upload.
   */
  stalled: boolean;
}

/** `geometry.userData` slot carrying the active ordering-buffer index. */
interface SortedIndexSlotUserData {
  sortedIndexSlot?: 0 | 1;
}

/**
 * Which ordering attribute is currently SELECTED for drawing: 0 =
 * `aSortedIndex`, 1 = `aSortedIndexB`. Lives on `geometry.userData` so it survives the
 * pool's release/re-acquire cycle with the buffers it describes.
 */
export function activeSortedIndexSlot(geometry: THREE.InstancedBufferGeometry): 0 | 1 {
  return (geometry.userData as SortedIndexSlotUserData).sortedIndexSlot ?? 0;
}

function setSortedIndexSlot(geometry: THREE.InstancedBufferGeometry, slot: 0 | 1): void {
  (geometry.userData as SortedIndexSlotUserData).sortedIndexSlot = slot;
}

const SLOT_ATTRIBUTE_NAMES = ['aSortedIndex', 'aSortedIndexB'] as const;

/** The ordering attribute currently selected for drawing. */
export function getActiveSortedIndexAttribute(
  geometry: THREE.InstancedBufferGeometry
): THREE.InstancedBufferAttribute | undefined {
  return geometry.getAttribute(SLOT_ATTRIBUTE_NAMES[activeSortedIndexSlot(geometry)]) as
    THREE.InstancedBufferAttribute | undefined;
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
 * validated against the ACTIVE one — so a short back buffer would leave
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

/** Whether the streaming ordering's owner has lifted the #715 back-pressure. */
function isBackPressureBypassed(state: OrderingApply): boolean {
  return state.callbacks?.bypassBackPressure?.() === true;
}

/**
 * Advance the staged ordering for `geometry` by one pump: write (part of) it
 * into the inactive attribute through its producer, and flip once it is whole.
 */
function applyNextSortedIndexChunk(
  geometry: THREE.InstancedBufferGeometry,
  state: OrderingApply
): SortedIndexPumpResult {
  const attr = getInactiveSortedIndexAttribute(geometry);
  if (!attr) {
    // Attribute gone (defensive — release paths cancel first). Abandon the
    // streaming/held orderings and any older selected ordering that still
    // awaited a completed draw (issue #713).
    cancelSortedIndexOrderingApply(geometry);
    return { more: false, flipped: false, stalled: false };
  }
  const staged = state.staged;
  const step =
    staged.kind === 'host'
      ? writeHostOrderingSlice(geometry, state, staged, attr)
      : writeDeviceOrdering(staged, attr);
  if (step === 'stalled') return { more: true, flipped: false, stalled: true };
  if (step === 'more') return { more: true, flipped: false, stalled: false };
  if (step === 'failed') return abandonStreamingOrdering(geometry, state);
  return completeStagedOrdering(geometry, state);
}

/** What writing (part of) a staged ordering into the inactive attribute did. */
type StagedWriteStep = 'complete' | 'more' | 'stalled' | 'failed';

/**
 * Write the next slice of a HOST ordering into the inactive attribute and
 * register ITS update range (not the collapsed `[0, …)` prefix — re-registering
 * the whole prefix each frame would upload 4+8+…+4·k MB instead of 4 MB/frame,
 * defeating the chunking). Any pending ranges (previous chunk not yet flushed —
 * hidden mesh, coalesced frames) are folded WITH the new slice into one span by
 * {@link collapseSortedIndexRanges}.
 *
 * Back-pressure (issue #715): on the chunked (classic WebGL) path, once at
 * least one slice has been written (`state.cursor > 0`), the previous slice may
 * not yet have been uploaded — the mesh was hidden, frustum-culled, or
 * otherwise not drawn since. The signal is {@link sliceUploadPending} (set on
 * write, cleared by the attribute's `onUploadCallback` on the NEXT real draw,
 * from both `createBuffer` and `updateBuffer`); advancing while it is set would
 * fold each new slice into the still-pending upload, collapsing ten intended
 * 4 MB slices into one 40 MB upload that lands in a single frame once the
 * object becomes drawable — exactly the hitch chunking exists to remove. So we
 * STALL instead: write nothing, leave `cursor` untouched, and let the next
 * DRAWN frame acknowledge the pending slice before advancing. The first slice
 * of every stream always proceeds (the guard also avoids blocking on a residual
 * flag from a prior cancelled stream on this attribute), and the WebGPU
 * single-slice path (`!ordering.chunked`) never stalls.
 */
function writeHostOrderingSlice(
  geometry: THREE.InstancedBufferGeometry,
  state: OrderingApply,
  staged: Extract<StagedOrdering, { kind: 'host' }>,
  attr: THREE.InstancedBufferAttribute
): StagedWriteStep {
  // Back-pressure: hold if the previous slice's upload has not been
  // acknowledged (issue #715). Gated on the chunked backend and on having
  // written at least one slice, so WebGPU's single-slice path and every
  // stream's first slice always proceed.
  const policy = rendererBackendPolicy().ordering;
  if (
    policy.chunked &&
    !isBackPressureBypassed(state) &&
    state.cursor > 0 &&
    sliceUploadPending.has(attr)
  ) {
    return 'stalled';
  }
  const arr = attr.array as Uint32Array;
  const start = state.cursor;
  // Backend gate: on WebGL a slice bounds the per-frame memcpy+upload;
  // the WebGPU backends never fire the upload callback the slice
  // back-pressure waits on, so they write the ordering whole and flip next
  // frame — still atomic, still one upload.
  const sliceEnd = policy.chunked ? start + sortedIndexChunkElements : staged.count;
  const end = Math.min(staged.count, sliceEnd, arr.length);
  clearIdentityPrefix(attr);
  // Both WebGPURenderer backends upload from attribute.array at update time.
  // The worker's live ordering is a view of its capacity-sized buffer; adopt
  // that buffer when it exactly matches the attribute's cached GPU buffer.
  // Partial-capacity and borrowed orderings keep the copy path.
  if (policy.adoptHostBuffer && start === 0 && canAdoptHostOrdering(staged, end, arr)) {
    attr.array =
      staged.ordering.length === arr.length
        ? staged.ordering
        : new Uint32Array(staged.ordering.buffer);
  } else {
    arr.set(staged.ordering.subarray(start, end), start);
  }
  state.cursor = end;

  collapseSortedIndexRanges(attr, end, start);

  if (policy.chunked) {
    // Arm the back-pressure flag and wire its acknowledgement (issue
    // #715). `onUpload` sets `attr.onUploadCallback`, which three fires
    // from BOTH `createBuffer` (first upload) and `updateBuffer` (later
    // ranged uploads) — so the flag clears on the very first draw too,
    // which is what defeats the `createBuffer`-never-clears deadlock.
    // A consumed slice means the mesh was actually drawn, so drive the
    // next pump precisely: no spin while hidden/culled (no draw → no
    // callback), and the resume gap is closed when it becomes drawable.
    sliceUploadPending.add(attr);
    attr.onUpload(() => {
      sliceUploadPending.delete(attr);
      // Discard the just-consumed ranges. `updateBuffer` already cleared
      // them, but `createBuffer` does NOT (it uploads the whole array via
      // `bufferData` and leaves the ranges lingering) — and the very next
      // slice would union with that stale range, making the following
      // draw upload TWO slices instead of one, breaking the per-frame
      // bound on exactly the first-drawn-mid-stream path. Always safe: at
      // ack time every byte of the array has just reached the GPU (either
      // path), and three's no-range fallback is a FULL `bufferSubData`,
      // so a cleared range can never lose data.
      attr.clearUpdateRanges();
      orderingApplies.get(geometry)?.callbacks?.requestFrame?.();
    });
  }

  return state.cursor < staged.count ? 'more' : 'complete';
}

/**
 * Whether a whole host ordering can become the attribute's array as is: the
 * worker transferred it, it covers exactly the staged count, and it views the
 * whole of a buffer the size of the attribute's.
 */
function canAdoptHostOrdering(
  staged: Extract<StagedOrdering, { kind: 'host' }>,
  end: number,
  arr: Uint32Array
): boolean {
  const ordering = staged.ordering;
  return (
    staged.adopt &&
    end === staged.count &&
    ordering.length === staged.count &&
    ordering.byteOffset === 0 &&
    ordering.buffer.byteLength === arr.byteLength
  );
}

/**
 * Let a DEVICE ordering write the inactive attribute's GPU buffer.
 *
 * Afterwards the attribute's host array no longer mirrors its GPU buffer
 * ({@link hostArrayBehind}). It still holds what the last HOST write to this
 * slot left, which can be a permutation of an OLDER population: the two slots
 * alternate, so an append's host writes may have gone to the other one. The
 * identity memo for it is void too, since a host write extending the memo
 * would upload only a tail over the device-written order.
 */
function writeDeviceOrdering(
  staged: Extract<StagedOrdering, { kind: 'device' }>,
  attr: THREE.InstancedBufferAttribute
): StagedWriteStep {
  // A pending host upload would overwrite the device result on the next draw.
  // Three clears ranges on WebGPU updates; the draw hook clears them after a
  // first buffer creation, which uploads the full array but retains ranges.
  if (attr.updateRanges.length > 0) return 'stalled';
  let written: boolean;
  try {
    written = staged.op.write(attr);
  } catch (error) {
    // A throw must not strand the apply: it would be retried, and throw, on
    // every later pump. Treat it as a write that could not run.
    log.error(Modules.PERFORMANCE, 'Device ordering write failed', error);
    written = false;
  }
  if (!written) return 'failed';
  clearIdentityPrefix(attr);
  hostArrayBehind.add(attr);
  return 'complete';
}

/** Start a held ordering into the (freshly inactive) buffer. */
function promotePending(state: OrderingApply, pending: StagedOrderingEntry): void {
  state.staged = pending.staged;
  state.callbacks = pending.callbacks;
  state.cursor = 0;
  state.pending = null;
}

/**
 * Drop the streaming ordering without flipping (a device write that could not
 * run). A held newer ordering, if any, starts next; otherwise a draw held for
 * this ordering's append falls back to a repaired permutation. State is
 * updated before the hook fires, matching the cancel paths' discipline.
 */
function abandonStreamingOrdering(
  geometry: THREE.InstancedBufferGeometry,
  state: OrderingApply
): SortedIndexPumpResult {
  const abandoned = state.callbacks;
  let more = false;
  if (state.pending) {
    promotePending(state, state.pending);
    more = true;
  } else {
    orderingApplies.delete(geometry);
    if (heldDrawTargets.get(geometry) === state.staged.count) {
      releaseSortedIndexDrawHold(geometry);
      abandoned?.requestFrame?.();
    }
  }
  abandoned?.onAbandoned?.();
  return { more, flipped: false, stalled: false };
}

/**
 * The inactive buffer now holds the WHOLE new permutation, whichever producer
 * wrote it: flip, raise a held draw, and hand the ordering's hooks to the
 * post-render acknowledgement.
 */
function completeStagedOrdering(
  geometry: THREE.InstancedBufferGeometry,
  state: OrderingApply
): SortedIndexPumpResult {
  // === The atomic swap ===
  // The inactive buffer now holds the WHOLE new permutation, so it is
  // safe to draw from. Flipping the slot is a single uniform write: no
  // rebind, no recompile, and no frame ever samples a half-written
  // ordering. The caller pushes the new slot to the materials, and the
  // per-frame callback that drives this pump runs BEFORE render, so a
  // visible mesh can upload the final slice and draw the new slot in that
  // frame. `onAfterRender` is the proof that this actually happened.
  setSortedIndexSlot(geometry, activeSortedIndexSlot(geometry) === 0 ? 1 : 0);
  // A held append draw becomes the grown population on THIS flip — the first
  // frame that has an ordering for it.
  raiseHeldDraw(geometry, state.staged.count);

  // The ordering that just finished streaming is now SELECTED for the render
  // about to run, but it is not applied yet: THREE has not consumed this
  // attribute's update ranges or drawn the mesh. Move its callbacks into the
  // post-render acknowledgement slot. If an older selected ordering never
  // rendered (hidden/off-screen mesh) this flip supersedes it, so abandon it.
  // Mutate the acknowledgement map before firing the displaced hook so
  // re-entrant cleanup never observes stale ownership.
  const selectedCallbacks = state.callbacks;
  const supersededSelected = awaitingDrawAcknowledgement.get(geometry);
  if (selectedCallbacks) {
    awaitingDrawAcknowledgement.set(geometry, selectedCallbacks);
  } else {
    awaitingDrawAcknowledgement.delete(geometry);
  }
  let result: SortedIndexPumpResult;
  if (state.pending) {
    // Start the held newest ordering into what just became the inactive
    // buffer. Held rather than restarted mid-stream on purpose: under a
    // continuous orbit a restart-on-arrival never converges (measured as
    // constant jank when L8 shipped) — whereas finishing, flipping, then
    // starting the newest always converges and always shows a complete
    // order.
    promotePending(state, state.pending);
    result = { more: true, flipped: true, stalled: false };
  } else {
    orderingApplies.delete(geometry);
    result = { more: false, flipped: true, stalled: false };
  }
  // Fire only after the chunk state is fully promoted/deleted as well as the
  // acknowledgement-map swap above. A re-entrant teardown therefore sees one
  // coherent ownership state and cannot double-close the superseded callback.
  supersededSelected?.onAbandoned?.();
  return result;
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
  return orderingApplies.has(geometry);
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
  const state = orderingApplies.get(geometry);
  if (!state) return { more: false, flipped: false, stalled: false };
  return applyNextSortedIndexChunk(geometry, state);
}

/**
 * Acknowledge that THREE completed a render of `geometry` using its currently
 * selected ordering buffer. This is the first point at which the final update
 * ranges have been consumed and the selected slot has participated in a draw,
 * so only here may the lifecycle report uploaded/applied (issue #713).
 *
 * Called from the render mesh's `onAfterRender` hook. Removal happens before
 * the callback so re-entrant disposal/cancellation cannot double-close it.
 */
export function acknowledgeSortedIndexOrderingDraw(geometry: THREE.InstancedBufferGeometry): void {
  const state = acknowledgeDeviceHostUpload(geometry);
  const callbacks = awaitingDrawAcknowledgement.get(geometry);
  if (callbacks) {
    awaitingDrawAcknowledgement.delete(geometry);
    callbacks.onApplied?.();
  }
  if (state?.staged.kind === 'device' && orderingApplies.get(geometry) === state) {
    state.callbacks?.requestFrame?.();
  }
}

/** Clear a device writer's old host ranges only when this draw consumed them. */
function acknowledgeDeviceHostUpload(
  geometry: THREE.InstancedBufferGeometry
): OrderingApply | undefined {
  // WebGPU has no attribute upload callback. An update clears ranges, but
  // first buffer creation uploads the full array and leaves them set. Both
  // ordering attributes are bound by the draw, so either path has consumed
  // the old host data here. Inspect before callbacks can stage a new upload.
  const state = orderingApplies.get(geometry);
  if (state?.staged.kind === 'device') {
    getInactiveSortedIndexAttribute(geometry)?.clearUpdateRanges();
  }
  return state;
}

/**
 * Abort an in-flight application — the streaming ordering, any held newest
 * ordering, AND a selected ordering still awaiting its first completed draw.
 *
 * Safe by construction now that a stream writes into the INACTIVE
 * buffer: abandoning one mid-slice discards a partially-written buffer
 * that was never drawn, and the displayed ordering is untouched. (Before
 * double-buffering this left the live attribute holding a mix of old and
 * new indices.) Callers: both identity writers (the commit path),
 * LOD-demotion detection in the coordinator's pump, node release,
 * geometry dispose.
 *
 * None of those orderings completed a draw, so all are abandoned (issue
 * #713). State is removed BEFORE hooks fire so re-entrant cleanup cannot
 * observe half-cancelled ownership or close a callback twice.
 */
export function cancelSortedIndexOrderingApply(geometry: THREE.InstancedBufferGeometry): void {
  const state = orderingApplies.get(geometry);
  const selected = awaitingDrawAcknowledgement.get(geometry);
  if (!state && !selected) return;
  orderingApplies.delete(geometry);
  awaitingDrawAcknowledgement.delete(geometry);
  state?.callbacks?.onAbandoned?.();
  state?.pending?.callbacks?.onAbandoned?.();
  selected?.onAbandoned?.();
}

/**
 * Abort every in-flight chunked application staged by `owner` (a depth-sort
 * coordinator's dataset-switch teardown and dispose — its `releaseAllNodes` /
 * `dispose`), or EVERY one when no owner is given (module-state reset in
 * tests). Filtering on the owner rather than on the coordinator's current
 * node list also catches an apply whose geometry no longer belongs to any of
 * its nodes, without touching another coordinator's streams on the same page.
 *
 * Each abandoned ordering's `onAbandoned` fires (issue #713). The matching
 * entries are removed BEFORE any hook runs, so a hook can never observe a
 * half-cleared map or re-enter the sweep.
 */
export function cancelAllSortedIndexOrderingApplies(owner?: object): void {
  const states = takeOwnedEntries(orderingApplies, owner, (state) => applyOwner(state));
  const selected = takeOwnedEntries(awaitingDrawAcknowledgement, owner, (c) => c.owner);
  for (const state of states) {
    state.callbacks?.onAbandoned?.();
    state.pending?.callbacks?.onAbandoned?.();
  }
  for (const callbacks of selected) callbacks.onAbandoned?.();
}

/** The owner tag of an in-flight apply (streaming or held ordering). */
function applyOwner(state: OrderingApply): object | undefined {
  return state.callbacks?.owner ?? state.pending?.callbacks?.owner;
}

/**
 * Remove and return the values of `map` owned by `owner` (all of them when
 * `owner` is undefined).
 */
function takeOwnedEntries<K, V>(
  map: Map<K, V>,
  owner: object | undefined,
  ownerOf: (value: V) => object | undefined
): V[] {
  const taken: V[] = [];
  for (const [key, value] of map) {
    if (owner !== undefined && ownerOf(value) !== owner) continue;
    taken.push(value);
    map.delete(key);
  }
  return taken;
}

/**
 * Run the first writes into a FRESH geometry's element storage (just
 * attached by {@link attachElementStorage}). If they throw — a texel
 * writer's short-source guard — the geometry is disposed, which frees the
 * texture its dispose listener owns, and the error is rethrown: the pair
 * has no owner yet, and every caller keeps its mesh on the OLD geometry
 * when a build throws. The one rule for the points, lines and gsplats
 * non-pool builders.
 */
export function writeFreshElementStorage(
  geometry: THREE.InstancedBufferGeometry,
  write: () => void
): void {
  try {
    write();
  } catch (err) {
    geometry.dispose();
    throw err;
  }
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
  const sortedIndex = rendererBackendPolicy().gpuCompute
    ? new StorageInstancedBufferAttribute(new Uint32Array(capacity), 1)
    : new THREE.InstancedBufferAttribute(new Uint32Array(capacity), 1);
  sortedIndex.setUsage(rendererBackendPolicy().ordering.usage);
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
  const sortedIndexB = rendererBackendPolicy().gpuCompute
    ? new StorageInstancedBufferAttribute(new Uint32Array(capacity), 1)
    : new THREE.InstancedBufferAttribute(new Uint32Array(capacity), 1);
  sortedIndexB.setUsage(rendererBackendPolicy().ordering.usage);
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
  // Every renderer (classic, and three's common Textures for both WebGPU
  // backends) invokes onUpdate after consuming an upload — the observable
  // "flush happened" signal that ends a pending FULL upload (see
  // pendingFullUpload) and records the first upload (see uploadedOnce).
  // Wired once here; nothing else sets onUpdate on element textures. (No
  // pending-full mark at attach: a fresh texture's GPU storage is
  // zero-initialized and only written texels are ever read, so a ranged
  // first upload is sufficient.)
  texture.onUpdate = () => {
    pendingFullUpload.delete(texture);
    uploadedOnce.add(texture);
    uploadedVersions.set(texture, texture.version);
  };
  texture.needsUpdate = true;
  elementTextures.add(texture);

  (geometry.userData as ElementStorageUserData).elementTexture = texture;
  (geometry.userData as ElementStorageUserData).elementTextureLayout = layout;
  geometry.addEventListener('dispose', () => texture.dispose());
  return texture;
}

/** The element texture attached by `attachElementStorage`, if any. */
export function getElementTexture(geometry: THREE.BufferGeometry): THREE.DataTexture | null {
  return (geometry.userData as ElementStorageUserData).elementTexture ?? null;
}

/** True after three has uploaded the current texture version. */
export function isElementTextureCurrent(texture: THREE.DataTexture): boolean {
  return uploadedVersions.get(texture) === texture.version;
}

/** The layout used to allocate a geometry's element texture. */
export function getElementTextureLayout(
  geometry: THREE.BufferGeometry
): ElementTextureLayout | null {
  return (geometry.userData as ElementStorageUserData).elementTextureLayout ?? null;
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
 * How many leading entries of an ordering buffer are known to hold the
 * identity permutation. Order-independent (commutative) nodes write identity
 * on every commit; without this, each commit rewrote and re-uploaded the whole
 * prefix (4 B x count, about 6% of a playback commit's upload bytes, #2944)
 * although the buffer already held it. Every write to an ordering buffer goes
 * through this module, so each non-identity writer clears the entry.
 */
const identityPrefix = new WeakMap<THREE.BufferAttribute, number>();

/** Forget that `attr` holds identity (a permutation is about to be written into it). */
function clearIdentityPrefix(attr: THREE.BufferAttribute): void {
  identityPrefix.delete(attr);
}

/**
 * Held draws (append and grow commits of every instanced type): geometry -> the element count the draw
 * is waiting to reach. While an entry exists, `instanceCount` is deliberately
 * LOWER than the committed population — see
 * {@link holdSortedIndexDrawForAppend}.
 */
const heldDrawTargets = new WeakMap<THREE.InstancedBufferGeometry, number>();

function setInstanceDrawCount(geometry: THREE.InstancedBufferGeometry, count: number): void {
  geometry.instanceCount = count;
  // Force THREE to recompute `_maxInstanceCount` (same as the adapters).
  delete (geometry as unknown as { _maxInstanceCount?: number })._maxInstanceCount;
}

/**
 * Raise a held draw to `count` if `count` is exactly the population it waits
 * for. Called by the two writers that install a COMPLETE ordering in the
 * active buffer (the slot flip and the live write), so the grown population
 * becomes visible on the very frame its own permutation does — never earlier.
 */
function raiseHeldDraw(geometry: THREE.InstancedBufferGeometry, count: number): void {
  if (heldDrawTargets.get(geometry) !== count) return;
  heldDrawTargets.delete(geometry);
  setInstanceDrawCount(geometry, count);
}

/**
 * Start (or continue) a HELD draw for an append commit that grew a node from
 * `drawnCount` drawn elements to `count` committed ones. Returns the count the
 * caller must draw now.
 *
 * WHY. An append grows the population while the displayed permutation still
 * covers only the old one, and every fallback for the grown population is
 * wrong until its sort lands: storage order draws the whole node unsorted
 * (measured as 75-225 ms per rung above `syncSortMaxElements` on a hosted 1.1M
 * splat ladder), and the old prefix plus a storage-order suffix is two
 * independently ordered populations. Holding the draw at the previous
 * population keeps every frame an EXACT back-to-front draw of SOME population:
 * the active buffer's `[0, drawnCount)` is untouched, the suffix texels are
 * uploaded but not yet drawn, and the grown population appears one sort
 * round-trip later, already ordered (`raiseHeldDraw`).
 *
 * The depth-sort coordinator owns the other exit: when no ordering for the
 * grown population will arrive (commutative mode, depth sort off, a failed or
 * superseded-without-successor sort) it calls
 * {@link releaseSortedIndexDrawHold}. A later non-append write (identity or
 * repair) also ends the hold; its caller sets `instanceCount` itself.
 *
 * Declines — full identity over `count`, returning `count` — when nothing is
 * drawn yet (holding would draw nothing), the drawn count is not below `count`
 * (not a real growth: THREE's default `instanceCount` is `Infinity`), or the
 * ordering pair is malformed.
 */
export function holdSortedIndexDrawForAppend(
  geometry: THREE.InstancedBufferGeometry,
  drawnCount: number,
  count: number
): number {
  // Any in-flight apply describes an older generation of this node.
  cancelSortedIndexOrderingApply(geometry);
  const drawn = drawnCount;
  if (!(drawn > 0) || drawn >= count || !sortedIndexBuffersUsable(geometry)) {
    writeSortedIndexIdentity(geometry, count);
    return count;
  }
  heldDrawTargets.set(geometry, count);
  return drawn;
}

/**
 * The GROWN-geometry sibling of {@link holdSortedIndexDrawForAppend}: a
 * commit that extends the previous population but lands in a DIFFERENT
 * geometry (the pool grew the node's buffers — every rung that doubles a
 * ladder does). The new geometry's texels are fully written, so its first
 * `seed.length` elements are the same elements the previous geometry drew
 * (the commit layer proves this with the prefix lineage), and `seed` — the
 * previous geometry's drawn permutation of `[0, seed.length)` — is an exact
 * ordering of them. Writes identity over `count` (normalising slot 0, so the
 * whole buffer stays a valid permutation), overlays `seed` on the prefix, and
 * holds the draw there until the grown population's ordering lands.
 *
 * Declines exactly like {@link holdSortedIndexDrawForAppend}.
 */
export function holdSortedIndexDrawFromSeed(
  geometry: THREE.InstancedBufferGeometry,
  seed: Uint32Array,
  count: number
): number {
  writeSortedIndexIdentity(geometry, count);
  const attr = getActiveSortedIndexAttribute(geometry);
  const drawn = seed.length;
  if (!attr || !(drawn > 0) || drawn >= count || !sortedIndexBuffersUsable(geometry)) {
    return count;
  }
  // The identity write above may have been a no-op (the buffer was already
  // known identity), so the overlay owns its own memo reset and upload range.
  clearIdentityPrefix(attr);
  (attr.array as Uint32Array).set(seed, 0);
  collapseSortedIndexRanges(attr, drawn);
  heldDrawTargets.set(geometry, count);
  return drawn;
}

/** The element count a held draw is waiting for, or `undefined` if none. */
export function sortedIndexDrawHoldTarget(
  geometry: THREE.InstancedBufferGeometry
): number | undefined {
  return heldDrawTargets.get(geometry);
}

/**
 * End a held draw WITHOUT a sorted ordering for the grown population: rebuild
 * the active ordering over the whole population from the drawn permutation
 * ({@link repairSortedIndexForCount}: old order first, the appended suffix
 * after it) and draw everything. The fallback for every path on which no
 * ordering for the grown population will arrive. No-op when nothing is held.
 *
 * @returns true when a hold ended (the drawn population changed), so the
 *   caller can wake a frame under render-on-change.
 */
export function releaseSortedIndexDrawHold(geometry: THREE.InstancedBufferGeometry): boolean {
  const target = heldDrawTargets.get(geometry);
  if (target === undefined) return false;
  repairSortedIndexForCount(geometry, geometry.instanceCount, target);
  setInstanceDrawCount(geometry, target);
  return true;
}

/**
 * Fill `aSortedIndex[0..count)` with identity ordering and register a
 * single collapsed prefix update range. Ranges accumulate across
 * commits while a mesh is not drawn (no backend merges them at flush),
 * so every write collapses the pending set to one `[0, max-end)` range.
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
 * Appends do not call it: they HOLD the draw at the previous population until
 * the grown one is sorted ({@link holdSortedIndexDrawForAppend}).
 */
export function writeSortedIndexIdentity(
  geometry: THREE.InstancedBufferGeometry,
  count: number
): void {
  // A fresh commit supersedes any in-flight chunked ordering apply —
  // its remaining slices belong to the OLD element population and
  // would scribble a stale permutation over the identity just written.
  cancelSortedIndexOrderingApply(geometry);
  // A full write ends any held draw; the caller sets `instanceCount`.
  heldDrawTargets.delete(geometry);
  // Normalise to slot 0 BEFORE picking the target, so identity always
  // lands in the buffer a default-valued selector uniform reads.
  setSortedIndexSlot(geometry, 0);
  const attr = getActiveSortedIndexAttribute(geometry);
  if (!attr) return;
  const arr = attr.array as Uint32Array;
  const n = Math.min(count, arr.length);
  const known = identityPrefix.get(attr) ?? 0;
  // Already identity over [0, n): nothing to write, nothing to upload.
  if (known >= n) return;
  for (let i = known; i < n; i++) arr[i] = i;
  identityPrefix.set(attr, n);
  collapseSortedIndexRanges(attr, n, known);
}

/**
 * Write a COMPLETE depth-sort permutation straight into the ACTIVE ordering
 * buffer, live, with no staging and no slot flip. Returns `count` on success
 * and 0 when the ordering is rejected.
 *
 * This is the one-shot counterpart of {@link writeSortedIndexOrdering}, which
 * streams into the INACTIVE buffer and flips only when the last slice lands —
 * necessarily a frame or more later, because the pump is driven once per
 * rendered frame. That delay is the whole point when the ordering arrives from
 * the SortWorker: it keeps a partially written permutation off screen.
 *
 * It is dead weight when the ordering was computed synchronously inside the
 * commit, because there is no partial state to hide — the permutation is
 * complete before the first write. Writing a complete permutation into the
 * active buffer is safe by construction (the identity writers do the same),
 * and it is what removes the unsorted frame rather than merely improving it.
 *
 * The two rejections mirror `writeSortedIndexOrdering`: a TRUNCATED ordering
 * is not a permutation of the drawn population, and an OVERSIZED count
 * describes a population this geometry does not hold. Neither is clamped —
 * a clamped prefix of a permutation is not a permutation of anything.
 */
export function writeSortedIndexOrderingLive(
  geometry: THREE.InstancedBufferGeometry,
  ordering: Uint32Array,
  count: number
): number {
  if (!Number.isInteger(count) || count <= 0) return 0;
  if (ordering.length < count) return 0;
  const attr = getActiveSortedIndexAttribute(geometry);
  if (!attr) return 0;
  const arr = attr.array as Uint32Array;
  if (count > arr.length) return 0;
  // Supersedes any in-flight chunked apply: its remaining slices belong to an
  // older sort of this same population and would land on top of this one.
  cancelSortedIndexOrderingApply(geometry);
  clearIdentityPrefix(attr);
  arr.set(ordering.subarray(0, count), 0);
  collapseSortedIndexRanges(attr, count);
  raiseHeldDraw(geometry, count);
  return count;
}

/**
 * Scratch "value already emitted" marks for {@link repairSortedIndexForCount}.
 * Module-level and grow-only: the repair runs on the commit hot path for
 * nodes of millions of elements, and a per-call `Uint8Array(count)` would
 * hand the GC exactly the churn the accumulators exist to avoid.
 */
let repairSeenScratch = new Uint8Array(0);

/**
 * Rebuild the ACTIVE ordering into a valid permutation of `[0, count)`,
 * reusing whatever permutation the buffer already holds over
 * `[0, prevCount)` instead of resetting to storage order.
 *
 * The problem this solves: an nD re-slice changes a node's resident element
 * count at almost every step (`cloud`, a 4D Points timelapse, walks
 * 27834 -> 27643 -> 27416 -> 27124 ...), and the commit paths' `preserveOrdering`
 * guard requires `prevCount === count`. So it never fired on a real timelapse
 * and every timepoint fell through to {@link writeSortedIndexIdentity}, drawing
 * at least one frame in pure storage order before the SortWorker's permutation
 * landed ~5 ms later. Under an order-dependent blending mode that frame is
 * visibly wrong — it reads as a flash once per timepoint through playback.
 *
 * The reason the guard was written that way is real and is stated at
 * `commit-gsplats-geometry.ts`: *a permutation of [0,prevCount) is not a
 * permutation of [0,count)*. This closes that gap rather than relaxing the
 * invariant. Measured on the live `cloud` demo at a frozen camera pose — the
 * fraction of sampled element pairs composited in correct back-to-front order,
 * across a real timepoint step:
 *
 * ```
 * ideal sort of the new data                     1.000
 * identity / storage order (the old fallback)    0.617
 * previous permutation, repaired to the count    0.858
 * ```
 *
 * (Freeze the camera pose before measuring anything like this. Demos open in
 * cinematic mode with the camera orbiting at ~13.85 deg/s, and an ordering
 * scored against a pose it was not sorted for reads 0.161 — stale in CAMERA,
 * not in data. That number cost an afternoon.)
 *
 * The walk emits each value below `count` the first time it is seen, then
 * appends whatever is still missing in ascending order. That yields a valid
 * permutation for a GROWN count, a SHRUNK count, **and a malformed input
 * alike** — a partially applied chunked ordering degrades to a valid
 * permutation instead of drawing one element twice and another never, which
 * matters because {@link cancelSortedIndexOrderingApply} exists precisely
 * because partial application is a reachable state.
 *
 * Unlike {@link writeSortedIndexIdentity} this deliberately does NOT re-home
 * the geometry on slot 0. A full identity write is the fresh-start signal that
 * lets an untracked node's default-valued selector uniform be correct; a
 * repair is the opposite claim — its callers fire only when the tenant, the
 * geometry and the attribute buffers are all unchanged, so the slot/uniform
 * pairing is already established, and normalising would throw away the
 * permutation being repaired.
 */
export function repairSortedIndexForCount(
  geometry: THREE.InstancedBufferGeometry,
  prevCount: number,
  count: number
): void {
  // Same supersede rule as the two identity writers: this commit's element
  // population invalidates the ordering an in-flight chunked apply was
  // streaming for the previous one.
  cancelSortedIndexOrderingApply(geometry);
  // A repaired full population ends any held draw; the caller sets
  // `instanceCount` (`releaseSortedIndexDrawHold` for the held case).
  heldDrawTargets.delete(geometry);
  const attr = getActiveSortedIndexAttribute(geometry);
  if (!attr) return;
  const arr = attr.array as Uint32Array;
  const n = Math.min(count, arr.length);
  if (n <= 0) return;
  // Read past `n` is deliberate on a shrink: the dropped values are exactly
  // the ones at or above the new count.
  const readEnd = Math.min(Math.max(prevCount, 0), arr.length);
  const identityKnown = identityPrefix.get(attr) ?? 0;
  if (identityKnown >= Math.min(readEnd, n)) {
    // Repairing an identity prefix yields identity (kept in order, missing
    // values appended ascending), so only the entries past the known prefix
    // are written and uploaded — the cost of an identity append. Every
    // order-independent append takes this path: its held draw is released
    // inside the same commit (`releaseSortedIndexDrawHold`).
    const from = Math.min(identityKnown, n);
    for (let i = from; i < n; i++) arr[i] = i;
    identityPrefix.set(attr, Math.max(identityKnown, n));
    if (from < n) collapseSortedIndexRanges(attr, n, from);
    return;
  }
  repairPermutationInPlace(arr, readEnd, n);
  clearIdentityPrefix(attr);
  collapseSortedIndexRanges(attr, n);
}

/**
 * The walk behind {@link repairSortedIndexForCount}: keep `arr[0, readEnd)`'s
 * values below `n` in their existing relative order, then append whatever is
 * still missing in ascending order, leaving a permutation of `[0, n)`.
 */
function repairPermutationInPlace(arr: Uint32Array, readEnd: number, n: number): void {
  if (repairSeenScratch.length < n) repairSeenScratch = new Uint8Array(n);
  const seen = repairSeenScratch;
  seen.fill(0, 0, n);
  // Pass 1 — keep the surviving prefix in its existing relative order.
  let write = 0;
  for (let i = 0; i < readEnd && write < n; i++) {
    const v = arr[i];
    if (v < n && seen[v] === 0) {
      seen[v] = 1;
      // In-place is safe: `write <= i` always, because every write is
      // preceded by at least one read.
      arr[write++] = v;
    }
  }
  // Pass 2 — anything the prefix did not account for (a grown count's new
  // tail, or a gap left by a malformed input) in ascending order.
  for (let v = 0; v < n && write < n; v++) {
    if (seen[v] === 0) arr[write++] = v;
  }
}

/**
 * Stage a depth-sort permutation (the SortWorker's back-to-front
 * ordering, depth-sorting Phase 2). Returns `count` when the ordering is
 * staged, or 0 when it is rejected (truncated or oversized ordering,
 * non-positive/non-integer count, malformed buffer pair).
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
 *
 * The optional `callbacks` bind a lifecycle to this specific ordering
 * (the coordinator's 'Depth Sort' profiler session — issue #713). They
 * fire ONLY when the ordering is accepted (this call returns > 0):
 * exactly one of `onApplied` (post-render acknowledgement after its buffer
 * flips in) or `onAbandoned` (dropped before that draw). A rejected write
 * (returns 0) fires
 * neither, leaving lifecycle ownership with the caller.
 * `adoptOrdering` is only for a freshly transferred, caller-discarded array.
 * On WebGPU, an array backed by the attribute's full capacity becomes its
 * inactive backing store, avoiding one full CPU copy before the GPU upload.
 */
export function writeSortedIndexOrdering(
  geometry: THREE.InstancedBufferGeometry,
  ordering: Uint32Array,
  count: number,
  callbacks?: SortedIndexApplyCallbacks,
  options?: { adoptOrdering?: boolean }
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
  // An OVERSIZED count (larger than the buffers) is rejected the same
  // way, never clamped to a prefix: capacity is always >= instanceCount,
  // so such an ordering describes a population this geometry does not
  // hold, and the retained prefix of a larger permutation is not a
  // permutation of anything — its entries index past the kept range.
  // (Unreachable from the coordinator: a commit that grew the population
  // acquires a larger geometry and bumps the generation, so an oversized
  // resolve is dropped there first.)
  if (count > (active.array as Uint32Array).length) return 0;
  // A count that is not a positive integer stages nothing. A NON-POSITIVE
  // (or NaN) count would otherwise write no entries on the first pump yet
  // find `cursor >= count` already true, so it would FLIP — publishing
  // whatever stale content the inactive buffer holds as if it were the
  // new ordering, which is exactly the corruption double-buffering exists
  // to prevent. A FRACTIONAL count would likewise flip with only
  // `floor(count)` entries written (`subarray` truncates while the cursor
  // reaches `count` exactly). (The commit path only ever passes
  // `ordering.length`, so this guards the writer's own contract rather
  // than a live caller.)
  if (!Number.isInteger(count) || count <= 0) return 0;
  const n = count;
  // Malformed pair — never repair it here (see `sortedIndexBuffersUsable`).
  if (!sortedIndexBuffersUsable(geometry)) return 0;

  return stageOrdering(
    geometry,
    { kind: 'host', ordering, adopt: options?.adoptOrdering === true, count: n },
    callbacks
  );
}

/**
 * Stage a depth-sort permutation that a GPU-side producer writes itself (the
 * WebGPU compute sort, #3162). The device sibling of
 * {@link writeSortedIndexOrdering}: same validation of `op.count`, same
 * staging, holding and supersede rules, same lifecycle `callbacks`. The pump
 * calls `op.write` with the INACTIVE attribute and flips on success, so the
 * device ordering is just as atomic. Returns `op.count` when staged, 0 when
 * rejected.
 */
export function writeDeviceSortedIndexOrdering(
  geometry: THREE.InstancedBufferGeometry,
  op: DeviceOrderingWrite,
  callbacks?: SortedIndexApplyCallbacks
): number {
  const active = getActiveSortedIndexAttribute(geometry);
  if (!active || !Number.isInteger(op.count) || op.count <= 0) return 0;
  if (op.count > (active.array as Uint32Array).length) return 0;
  if (!sortedIndexBuffersUsable(geometry)) return 0;
  return stageOrdering(geometry, { kind: 'device', op, count: op.count }, callbacks);
}

/**
 * Stage a validated ordering: hold it behind a stream already in flight
 * (latest wins), or start it. Returns the staged count.
 */
function stageOrdering(
  geometry: THREE.InstancedBufferGeometry,
  staged: StagedOrdering,
  callbacks: SortedIndexApplyCallbacks | undefined
): number {
  const inFlight = orderingApplies.get(geometry);
  if (inFlight) {
    // A newer ordering supersedes any previously-held one; that held
    // ordering never became drawn, so abandon it (issue #713). Install the
    // new pending FIRST, then fire the displaced hook — matching the
    // remove-before-fire discipline of the cancel paths, so a re-entrant
    // hook can never observe (or act on) the stale pending.
    const superseded = inFlight.pending;
    inFlight.pending = { staged, callbacks: callbacks ?? null };
    superseded?.callbacks?.onAbandoned?.();
    return staged.count;
  }
  orderingApplies.set(geometry, { staged, callbacks: callbacks ?? null, cursor: 0, pending: null });
  if (!applyDisposeHooked.has(geometry)) {
    applyDisposeHooked.add(geometry);
    // Structural cancellation on geometry death (mirrors the element
    // texture's dispose-listener lifetime pin): a disposed geometry
    // must not be kept alive by the applies map, nor pumped again.
    geometry.addEventListener('dispose', () => cancelSortedIndexOrderingApply(geometry));
  }
  return staged.count;
}

/**
 * Register ONE update range covering the fresh entries `[from, n)` plus any
 * still-pending ranges (see the identity writer's doc comment for why ranges
 * must never accumulate). Every writer's ranges are prefixes or consecutive
 * slices, so the union is contiguous and the data under it is current: a
 * superset upload is correct, never stale.
 */
function collapseSortedIndexRanges(
  attr: THREE.InstancedBufferAttribute,
  n: number,
  from = 0
): void {
  let rangeStart = from;
  let rangeEnd = n;
  for (const range of attr.updateRanges) {
    const end = range.start + range.count;
    if (range.start < rangeStart) rangeStart = range.start;
    if (end > rangeEnd) rangeEnd = end;
  }
  attr.clearUpdateRanges();
  attr.addUpdateRange(rangeStart, rangeEnd - rangeStart);
  attr.needsUpdate = true;
  // The upload starts at 0, so the GPU's drawn prefix is the host array again.
  if (rangeStart === 0) hostArrayBehind.delete(attr);
}

/**
 * Whether the ACTIVE ordering attribute's host array is the order its GPU
 * buffer draws. False after a device write that no host write from index 0
 * has replaced since.
 */
export function isSortedIndexHostArrayCurrent(geometry: THREE.InstancedBufferGeometry): boolean {
  const attr = getActiveSortedIndexAttribute(geometry);
  return !attr || !hostArrayBehind.has(attr);
}

/**
 * Before the host arrays are re-uploaded after a context or device loss, make
 * the ACTIVE one a permutation of the drawn population again if a device write
 * left it behind: its surviving entries keep their order, the missing ones are
 * appended (`repairPermutationInPlace`). It is an older order than the
 * one lost with the GPU buffer, which the next sort replaces, but never a
 * corrupt one. The inactive buffer needs nothing: every flip writes it first.
 */
export function repairSortedIndexHostArray(geometry: THREE.InstancedBufferGeometry): void {
  const attr = getActiveSortedIndexAttribute(geometry);
  if (!attr || !hostArrayBehind.has(attr)) return;
  const arr = attr.array as Uint32Array;
  const drawn = Math.min(geometry.instanceCount, arr.length);
  if (drawn > 0) repairPermutationInPlace(arr, drawn, drawn);
  clearIdentityPrefix(attr);
  hostArrayBehind.delete(attr);
}
