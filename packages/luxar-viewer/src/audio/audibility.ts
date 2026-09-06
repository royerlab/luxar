/**
 * The slab rule for sound nodes — pure functions, no DOM, no Web Audio.
 *
 * A sound node's audibility IS the hidden-dimension slab rule the geometry
 * nodes use for visibility (`SOUND_SPEC.md` §3.1, §7): its `(K, ndim)` rows are
 * tested against the current slice with the same per-vertex kernel a mesh runs
 * (`wasm/typescript/mesh-culling.ts`), through the same `deriveNodeViewState`
 * that folds `extend_to_all` and the inverse `nd_transform` in. So "sounds react
 * to hidden dimensions" is one mechanism, not a second one.
 *
 * The base view state is rebuilt from the dims manager's `SimpleDims` on every
 * dimension change — the scene loader's own view state is private and its sweep
 * is async and lock-serialised, while audibility is a synchronous boolean.
 *
 * @module audio/audibility
 */

import type { SimpleDims } from '../types/dims';
import type { SceneNode, ViewState } from '../data/data-loader-types';
import type { SoundSourceDescriptor } from '../types/audio';
import {
  computeTolerance,
  type DimensionInfo,
} from '../data/loaders/spatial-query/tolerance-computer';
import { deriveNodeViewState } from '../data/scene-loader/view-state/derive-node-view-state';
import { mesh_vertex_visibility_mask } from '../wasm/typescript/mesh-culling';

/**
 * The query the slab kernel runs against, built from a dims snapshot.
 *
 * Tolerance borrows the MESH rule — a discrete hidden dim (story, channel, time
 * as index) admits half a step, a continuous one a one-cell slab — because a
 * sound, like a mesh, has no per-element extent of its own to widen the slab
 * with.
 */
export function buildSoundBaseViewState(dims: SimpleDims): ViewState {
  const metadata = dims.metadata as unknown as DimensionInfo[] | undefined;
  const tolerance = computeTolerance('mesh', dims.displayed, dims.ndim, metadata, {});
  return {
    displayDims: [...dims.displayed],
    slicePosition: [...dims.currentStep],
    tolerance,
    dimensions: dims.metadata,
  } as ViewState;
}

/**
 * Fill `out` (length `nPositions`, 1 = audible) for every row of `desc` and
 * return the audible count. A descriptor without positions is the caller's
 * business (it is live everywhere); this function expects rows.
 *
 * A node whose `nd_transform` leaves the slice with no preimage is silent, like
 * the geometry it would accompany.
 */
export function computeRowAudibility(
  desc: Pick<SoundSourceDescriptor, 'path' | 'positions' | 'nPositions' | 'ndim'>,
  extendToAll: string[] | undefined,
  base: ViewState,
  sceneGraph: SceneNode | null,
  out: Uint8Array
): number {
  if (!desc.positions || desc.nPositions === 0) {
    out.fill(0);
    return 0;
  }
  const derived = deriveNodeViewState(desc.path, { extend_to_all: extendToAll }, base, sceneGraph, {
    applyPartialExtendTolerance: true,
  });
  const vs = derived.viewState;
  if (vs.noPreimage) {
    out.fill(0);
    return 0;
  }
  return mesh_vertex_visibility_mask(
    desc.positions,
    Float32Array.from(vs.slicePosition),
    Float32Array.from(vs.tolerance),
    Uint32Array.from(vs.displayDims),
    desc.ndim,
    desc.nPositions,
    out
  );
}

/**
 * The displayed 3-space coordinates of one row (`displayDims` picks which
 * columns are x/y/z; fewer than three displayed dims pad with 0).
 */
export function displayedXYZ(
  positions: Float32Array,
  row: number,
  ndim: number,
  displayDims: readonly number[]
): [number, number, number] {
  const base = row * ndim;
  const pick = (i: number): number => {
    const d = displayDims[i];
    return d === undefined ? 0 : positions[base + d];
  };
  return [pick(0), pick(1), pick(2)];
}
