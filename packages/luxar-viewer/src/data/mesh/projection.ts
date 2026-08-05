/**
 * Mesh display-space projection: nD cull, index rebuild, and winding parity.
 *
 * Turns a whole {@link LoadedMeshData} plus a view state into the three things
 * the geometry builder needs: a display-space `position` buffer, the index
 * buffer of currently-visible triangles, and the material `side` the epoch must
 * use.
 *
 * ## Whole-triangle cull, not clipping
 *
 * A triangle is drawn **iff all three of its vertices pass the nD slab test**
 * (`docs/specs/MESH_NODE_SPEC.md` §5.1). Lines clip a segment against the slab
 * and interpolate every attribute at the clip parameter; the triangle
 * equivalent is nD polygon clipping with fan re-triangulation and per-new-vertex
 * attribute interpolation, every frame the slice moves. v1 does not do that.
 *
 * The price is stated plainly rather than hidden: a surface cut by a slice shows
 * a **ragged, triangle-quantized boundary**, not a clean planar cut. On a
 * well-tessellated mesh sliced with a tolerance comparable to the edge length
 * that reads as a slightly jagged edge; on a coarse mesh with a thin tolerance
 * it can drop whole regions.
 *
 * ## No vertex compaction
 *
 * Only the *index* buffer is rebuilt on a slice change. The vertex attribute
 * buffers are uploaded once, in full, and left alone — `drawElements` never
 * fetches an unreferenced vertex, so culled vertices cost nothing to draw, and
 * the mesh is resident in full anyway. The only cost is VRAM for currently
 * invisible vertices, bounded by the mesh size, which is already the resident
 * working set. This is what lets the loader keep native-dtype colours: the
 * alternative, `compact_by_mask`, is `&[f32]`-only.
 *
 * A `displayDims` change is the one case that also rewrites `position`, because
 * `position` is `displayDims`-derived. That is re-extraction of the projection,
 * **not** compaction.
 *
 * @module data/mesh/projection
 */

import { log, Modules } from '../../utils/log';
import { validateProjectionInputs } from '../../workers/data-worker/validation';
import type { WasmModule } from '../../wasm/types';
import type {
  LoadedMeshData,
  MeshProjectionBounds,
  MeshProjectionTargetBuffers,
  MeshViewState,
} from '../../types/mesh';

/** Which faces of the surface an epoch must draw. */
export type MeshSide = 'front' | 'double';

/** Everything the geometry builder needs for one projection epoch. */
export interface ProjectedMeshData {
  /** Display-space positions (`vertexCount * 3`) */
  position: Float32Array;

  /**
   * Vertex indices of the visible triangles (`visibleFaceCount * 3`), in the
   * ORIGINAL (un-remapped) vertex numbering.
   */
  indices: Uint32Array;

  /** Triangles that survived the cull */
  visibleFaceCount: number;

  /** Vertices that passed the slab test (diagnostic; not a draw bound) */
  visibleVertexCount: number;

  /**
   * The material `side` this epoch requires.
   *
   * `'front'` only when the node asked for single-sided AND projected winding is
   * decidable (see {@link resolveWinding}); `'double'` otherwise.
   */
  side: MeshSide;

  /** True when the cull was skipped because there are no hidden dimensions */
  usedFastPath: boolean;

  /**
   * True when `position` was (re)extracted this epoch — i.e. `displayDims` changed,
   * or this is the first projection.
   *
   * Reported explicitly because the position buffer is REUSED across epochs
   * (`LoadedMeshData.projection`), so its array identity can no longer signal a
   * change. Consumers must gate the re-upload and the bounds recompute on this
   * rather than on identity; using identity with a reused buffer silently stops
   * re-uploading after an axis permutation, and using it with a freshly allocated
   * buffer re-uploads the whole mesh on every slice move (#1245).
   */
  positionChanged: boolean;

  /**
   * The `displayDims.join()` these positions were extracted for. The commit stamps it
   * on the geometry and re-uploads whenever it differs from the geometry's last-uploaded
   * key — so an aborted commit (which advanced the loader's `displayDimsKey` but never
   * uploaded) is still repaired by the next commit at that `displayDims`.
   * `positionChanged` alone cannot: it is a projection-time signal and reads false on
   * the epoch after a superseded re-extraction.
   */
  positionKey: string;

  /**
   * Display-space AABB over the vertices the emitted index references, or `null` when
   * nothing is drawn.
   *
   * Same role as `LinesProjectionBounds` / `GSplatsProjectionBounds`: the projection
   * computes it and `computeMeshBounds` sets the geometry's box and sphere from it.
   * Bounding the INDEXED vertices rather than the whole position buffer is the point —
   * under no-compaction `position` holds every vertex of the whole nD mesh, so a 4D
   * surface that translates over time would otherwise frame its entire trajectory
   * (#1252).
   */
  bounds: MeshProjectionBounds | null;

  /**
   * Set when the node asked for single-sided rendering but winding could not be
   * decided, carrying the reason. Threaded out to the caller rather than logged
   * here so {@link resolveWinding} and {@link projectMeshTo3D} stay free of side
   * effects — the projection runs on every slice move, so a warning emitted at
   * this depth would flood the console during a scrub.
   */
  undecidableReason?: string;
}

/** How the winding of the displayed projection relates to the authored frame. */
export interface WindingDecision {
  /** Whether each triangle's indices must be swapped to restore front-facing order */
  reverse: boolean;
  /** The material side the epoch must use */
  side: MeshSide;
  /**
   * Set when single-sided rendering was requested but cannot be honoured, with
   * the reason — the caller logs it once per node rather than per rebuild.
   */
  undecidableReason?: string;
}

/**
 * Decide winding for one `displayDims` epoch.
 *
 * ## Why this is not simply "flip if the permutation is odd"
 *
 * Winding is only *decidable* against the authored winding frame — the axis
 * triple the stored face order is front-facing in, which is `sorted(normal_dims)`
 * (§3.2). Three cases follow, and they are genuinely different:
 *
 * 1. **The displayed triple equals the frame, even permutation.** Draw as
 *    authored.
 * 2. **The displayed triple equals the frame, odd permutation.** Display space
 *    is a reflection of the frame, so every projected triangle's orientation is
 *    reversed — uniformly. Swapping two of each triangle's three indices restores
 *    it. Without this a `double_sided: false` mesh renders **inside-out**, which
 *    for an open surface means it vanishes entirely.
 * 3. **The displayed triple is a DIFFERENT triple** than the frame (e.g.
 *    `[0,1,2]` → `[1,2,3]`), or the mesh declares no frame at all (no stored
 *    normals). Projected orientation is then per-triangle data-dependent — some
 *    triangles project front-facing, some back — and **no index post-pass can fix
 *    it**. The epoch renders double-sided regardless of `double_sided: false`.
 *
 * Case 2's reversal is keyed to the *current* `displayDims` parity, not to the
 * event of `displayDims` changing, so it must run on **every** index build in an
 * odd-parity epoch: initial load, slice move, and `displayDims` change alike.
 * Nothing restricts the opening view to ascending order, so the very first build
 * can already need it.
 *
 * @param displayDims - The displayed axis triple, in x/y/z order.
 * @param normalDims - `normal_dims` when the node has stored normals.
 * @param doubleSided - The node's authored `double_sided`.
 */
export function resolveWinding(
  displayDims: readonly number[],
  normalDims: readonly number[] | undefined,
  doubleSided: boolean
): WindingDecision {
  // An authored double-sided mesh needs no decision: both orientations draw, so
  // parity is unobservable. Returning early also keeps the notice below quiet
  // for the overwhelmingly common case.
  if (doubleSided) return { reverse: false, side: 'double' };

  if (displayDims.length !== 3) {
    return {
      reverse: false,
      side: 'double',
      undecidableReason: `${displayDims.length} displayed dimensions (a winding frame needs 3)`,
    };
  }

  if (!normalDims || normalDims.length !== 3) {
    return {
      reverse: false,
      side: 'double',
      undecidableReason:
        'the mesh declares no winding frame (no stored normals), so projected ' +
        'orientation varies per triangle',
    };
  }

  const frame = [...normalDims].sort((a, b) => a - b);
  const displayed = [...displayDims].sort((a, b) => a - b);
  const sameTriple = frame.every((d, i) => d === displayed[i]);
  if (!sameTriple) {
    return {
      reverse: false,
      side: 'double',
      undecidableReason:
        `displayed axes [${displayDims.join(', ')}] are a different triple than the ` +
        `winding frame [${frame.join(', ')}], so projected orientation varies per ` +
        'triangle and no index reversal can correct it',
    };
  }

  return { reverse: permutationParityIsOdd(displayDims, frame), side: 'front' };
}

/**
 * Whether mapping `frame` onto `order` is an odd permutation.
 *
 * Counts inversions rather than composing cycles: for three elements that is
 * three comparisons and no allocation, and it stays obviously correct.
 * `order` and `frame` are assumed to hold the same three values — the caller
 * has already established that.
 */
function permutationParityIsOdd(order: readonly number[], frame: readonly number[]): boolean {
  // Position of each displayed axis within the (ascending) frame.
  const rank = order.map((d) => frame.indexOf(d));
  let inversions = 0;
  for (let i = 0; i < rank.length; i++) {
    for (let j = i + 1; j < rank.length; j++) {
      if (rank[i] > rank[j]) inversions++;
    }
  }
  return inversions % 2 === 1;
}

/**
 * Reverse triangle winding in place by swapping each face's last two indices.
 *
 * Swapping exactly two of the three is what reverses orientation; swapping all
 * three (a rotation) leaves winding unchanged, which is the easy mistake here.
 */
function reverseWinding(indices: Uint32Array, faceCount: number): void {
  for (let f = 0; f < faceCount; f++) {
    const b = f * 3 + 1;
    const c = f * 3 + 2;
    const tmp = indices[b];
    indices[b] = indices[c];
    indices[c] = tmp;
  }
}

/**
 * AABB of the display-space vertices `indices` references.
 *
 * Walks the INDEX rather than the position buffer, which is what makes it exclude
 * culled and wholly-unreferenced vertices. Returns `null` for an empty index — "no
 * drawn geometry", which the framing walk must skip rather than treat as a point at
 * the origin. Non-finite coordinates are skipped so one bad vertex cannot poison the
 * box into NaN (the cull already drops vertices with non-finite HIDDEN coordinates,
 * but a displayed axis is not filtered).
 */
function computeMeshProjectionBounds(
  position: Float32Array,
  indices: Uint32Array
): MeshProjectionBounds | null {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  let seen = 0;
  for (let i = 0; i < indices.length; i++) {
    const base = indices[i] * 3;
    const x = position[base];
    const y = position[base + 1];
    const z = position[base + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    if (x < min[0]) min[0] = x;
    if (y < min[1]) min[1] = y;
    if (z < min[2]) min[2] = z;
    if (x > max[0]) max[0] = x;
    if (y > max[1]) max[1] = y;
    if (z > max[2]) max[2] = z;
    seen++;
  }
  return seen > 0 ? { min, max } : null;
}

/**
 * The fast path's display-space bounds, measured once per `displayDims` epoch.
 *
 * Falls back to computing without caching when the node has no loader-owned
 * projection slot (the unit-test and placeholder callers), so behaviour is identical
 * either way — only the repeat cost differs.
 */
function fastPathBounds(
  scratch: MeshProjectionTargetBuffers | undefined,
  position: Float32Array,
  indices: Uint32Array,
  displayDimsKey: string
): MeshProjectionBounds | null {
  if (scratch && scratch.fastPathBoundsKey === displayDimsKey && scratch.fastPathBounds) {
    return scratch.fastPathBounds;
  }
  const bounds = computeMeshProjectionBounds(position, indices);
  if (scratch) {
    scratch.fastPathBounds = bounds;
    scratch.fastPathBoundsKey = displayDimsKey;
  }
  return bounds;
}

/**
 * Project a loaded mesh into display space for one view state.
 *
 * @param backend - WASM module or the TypeScript reference, already selected for
 *   this mesh's `ndim` by the caller through `pickBackend`. Both implement the
 *   two cull kernels; the TS one is also the production `ndim > 16` backend, and
 *   `pickBackend` swaps the WHOLE module there — which is why both kernels must
 *   exist on `WasmModule` and in both backends.
 */
export function projectMeshTo3D(
  data: LoadedMeshData,
  viewState: MeshViewState,
  normalDims: readonly number[] | undefined,
  doubleSided: boolean,
  backend: WasmModule
): ProjectedMeshData {
  const { vertices, faces, vertexCount, faceCount, ndim } = data;
  const { displayDims, slicePosition, tolerance } = viewState;

  // The shared guard the worker dispatchers use, for the same reason they use it:
  // the cull kernel indexes `slice_position[dim]` / `tolerance[dim]` for every
  // hidden `dim < ndim`, and the Rust crate is `panic = "abort"` — a short array
  // does not fail as a node-scoped error, it traps and takes down the whole WASM
  // module. Points guards `slicePosition` and Lines guards `tolerance`; mesh reads
  // both, so it checks both.
  validateProjectionInputs(
    'projectMeshTo3D',
    vertices,
    displayDims,
    slicePosition,
    ndim,
    vertexCount
  );
  if (tolerance.length < ndim) {
    throw new Error(
      `projectMeshTo3D: tolerance too short (got ${tolerance.length}, expected ≥ ${ndim})`
    );
  }

  // Reuse the loader-owned buffer when there is one, and re-extract only when the
  // displayed axis triple actually changed — positions depend on `displayDims` alone,
  // so a pure slice move must not touch them (the whole point of #1245).
  const positionScratch = data.projection;
  const displayDimsKey = displayDims.join();
  const reusable =
    positionScratch && positionScratch.position.length === vertexCount * 3 ? positionScratch : null;
  const position = reusable ? reusable.position : new Float32Array(vertexCount * 3);
  // Without a reusable buffer the array is newly allocated and therefore always
  // stale, so it must be extracted every call.
  const positionChanged = !reusable || reusable.displayDimsKey !== displayDimsKey;
  if (positionChanged) {
    backend.extract_3d_positions(
      vertices,
      new Uint32Array(displayDims),
      ndim,
      vertexCount,
      position
    );
    if (reusable) reusable.displayDimsKey = displayDimsKey;
  }

  const winding = resolveWinding(displayDims, normalDims, doubleSided);

  // A discrete nd_transform maps this world slice to no local grid point on some
  // hidden dimension, so nothing in this node belongs to the slice. Mirror the
  // spatial-index loaders (points/lines/gsplats), which return an empty query for
  // `noPreimage` rather than culling against the fractional inverse position — the
  // latter would leak triangles from a neighbouring category. See ViewState.noPreimage.
  if (viewState.noPreimage) {
    return {
      position,
      indices: new Uint32Array(0),
      visibleFaceCount: 0,
      visibleVertexCount: 0,
      side: winding.side,
      usedFastPath: false,
      positionChanged,
      positionKey: displayDimsKey,
      bounds: null,
      undecidableReason: winding.undecidableReason,
    };
  }

  // §5.5 fast path: with no hidden dimensions the mask is trivially all-ones, so
  // the cull is skipped and the index buffer is used verbatim.
  //
  // Skipping the cull is NOT the same as doing no work. A `displayDims` change
  // on this path still re-extracts positions (above) and — for an odd-parity
  // permutation — still reverses winding (below). Only the mask recompute is
  // elided. A 3D axis permutation is exactly this case.
  const hasHiddenDims = displayDims.length < ndim;
  if (!hasHiddenDims) {
    const indices = new Uint32Array(faces);
    if (winding.reverse) reverseWinding(indices, faceCount);
    return {
      position,
      indices,
      visibleFaceCount: faceCount,
      visibleVertexCount: vertexCount,
      side: winding.side,
      usedFastPath: true,
      positionChanged,
      positionKey: displayDimsKey,
      // Computed even here: nothing is culled on the fast path, but a vertex no
      // triangle references still inflates the whole-buffer box.
      //
      // CACHED per `displayDims` epoch, unlike the cull path below. Nothing is culled
      // here, so the emitted index is `faces` verbatim and the box depends only on
      // `position` — it cannot change between sweeps at one `displayDims`. The
      // recompute is O(faces) (measured 26 ms at 5M faces), and a 3D mesh in a scene
      // with navigable dimensions elsewhere is swept on every scrub frame, so leaving
      // it in overran the frame budget to re-derive an identical box.
      bounds: fastPathBounds(positionScratch, position, indices, displayDimsKey),
      undecidableReason: winding.undecidableReason,
    };
  }

  // Reused when the loader supplied buffers of the right size, allocated otherwise. Both
  // of these are rewritten in full every cull, so there is no staleness to track — the
  // only thing reuse buys is not producing garbage proportional to the mesh on every
  // slice move (#1245's argument, applied to the two buffers it did not cover).
  const mask =
    positionScratch && positionScratch.mask.length === vertexCount
      ? positionScratch.mask
      : new Uint8Array(vertexCount);
  const visibleVertexCount = backend.mesh_vertex_visibility_mask(
    vertices,
    new Float32Array(viewState.slicePosition),
    new Float32Array(viewState.tolerance),
    new Uint32Array(displayDims),
    ndim,
    vertexCount,
    mask
  );

  // Worst-case sized: every face could survive. The kernel returns how many
  // actually did, and the result is sliced to exactly that — an oversized buffer
  // uploaded whole would draw stale triangles from its untouched tail.
  const scratch =
    positionScratch && positionScratch.faceScratch.length === faceCount * 3
      ? positionScratch.faceScratch
      : new Uint32Array(faceCount * 3);
  const visibleFaceCount = backend.compact_visible_faces(faces, mask, faceCount, scratch);
  const indices = scratch.subarray(0, visibleFaceCount * 3);
  if (winding.reverse) reverseWinding(indices, visibleFaceCount);

  return {
    position,
    // A copy, not the subarray view — and now REQUIRED rather than merely tidy.
    // `scratch` is loader-owned and reused by the NEXT epoch, so a view onto it would
    // alias data the next projection overwrites, silently changing indices a caller
    // still holds. (The original reason — that `THREE.BufferAttribute` uploads
    // `array.buffer`, so a view would upload the whole worst-case allocation — no longer
    // applies: `applyMeshIndices` always `set`s into its own capacity buffer. The copy
    // survives because reuse gave it a better reason.)
    indices: new Uint32Array(indices),
    visibleFaceCount,
    visibleVertexCount,
    side: winding.side,
    usedFastPath: false,
    positionChanged,
    positionKey: displayDimsKey,
    bounds: computeMeshProjectionBounds(position, indices),
    undecidableReason: winding.undecidableReason,
  };
}

/**
 * Log the one-time notice when single-sided rendering was requested but winding
 * is undecidable.
 *
 * Kept separate from {@link resolveWinding} so that function stays pure and
 * cheap to test: it is called on every index build, and a log side-effect there
 * would fire per rebuild rather than once per node.
 */
export function noticeUndecidableWinding(
  path: string,
  reason: string,
  alreadyNoticed: Set<string>
): void {
  if (alreadyNoticed.has(path)) return;
  alreadyNoticed.add(path);
  log.warning(
    Modules.SCENE_LOADER,
    `Mesh ${path} asked for single-sided rendering, but ${reason}. ` +
      'Rendering double-sided for this view instead.'
  );
}
