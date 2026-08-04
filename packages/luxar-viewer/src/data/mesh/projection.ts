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
import type { LoadedMeshData, MeshViewState } from '../../types/mesh';

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
   * Set when the node asked for single-sided rendering but winding could not be
   * decided, carrying the reason. Threaded out to the caller rather than logged
   * here so {@link resolveWinding} and {@link projectMesh} stay free of side
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
 * Project a loaded mesh into display space for one view state.
 *
 * @param backend - WASM module or the TypeScript reference, already selected for
 *   this mesh's `ndim` by the caller through `pickBackend`. Both implement the
 *   two cull kernels; the TS one is also the production `ndim > 16` backend, and
 *   `pickBackend` swaps the WHOLE module there — which is why both kernels must
 *   exist on `WasmModule` and in both backends.
 */
export function projectMesh(
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
  validateProjectionInputs('projectMesh', vertices, displayDims, slicePosition, ndim, vertexCount);
  if (tolerance.length < ndim) {
    throw new Error(
      `projectMesh: tolerance too short (got ${tolerance.length}, expected ≥ ${ndim})`
    );
  }

  const position = new Float32Array(vertexCount * 3);
  backend.extract_3d_positions(vertices, new Uint32Array(displayDims), ndim, vertexCount, position);

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
      undecidableReason: winding.undecidableReason,
    };
  }

  const mask = new Uint8Array(vertexCount);
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
  const scratch = new Uint32Array(faceCount * 3);
  const visibleFaceCount = backend.compact_visible_faces(faces, mask, faceCount, scratch);
  const indices = scratch.subarray(0, visibleFaceCount * 3);
  if (winding.reverse) reverseWinding(indices, visibleFaceCount);

  return {
    position,
    // A copy, not the subarray view: the view shares `scratch`'s whole buffer,
    // and `THREE.BufferAttribute` uploads `array.buffer` — so handing over the
    // view would upload the full worst-case allocation, including the stale
    // tail past `visibleFaceCount * 3`.
    indices: new Uint32Array(indices),
    visibleFaceCount,
    visibleVertexCount,
    side: winding.side,
    usedFastPath: false,
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
