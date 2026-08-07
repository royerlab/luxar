/**
 * Mesh data-processor concern — the Mesh sibling of
 * `data-processor-lines.ts` / `data-processor-gsplats.ts`.
 *
 * Runs the display-space projection: `extract_3d_positions`, the whole-triangle nD
 * cull, and the winding post-pass (`data/mesh/projection.ts`). Unlike its two
 * async siblings there is no worker RPC — mesh projects **in-process**, and for a
 * different reason than Points does.
 *
 * Points runs in-process because its projection is memory-bound per update, so
 * offloading would pay transfer cost both ways for negligible compute savings.
 * Mesh runs in-process because it is whole-node resident, so its payload is ONE
 * large transfer rather than lines' many small ones — a genuinely different
 * benefit profile that should be measured before a worker is built for it, not
 * assumed. That keeps the measure-first performance doctrine intact rather than
 * copying the lines shape because it exists.
 *
 * It is still `async`, because selecting the backend is (`pickBackend` may await
 * the WASM module's first load).
 *
 * @module data/scene-loader/process/data-processor-mesh
 */

import { getMeshBackend } from '../../../workers/data-worker/projection/in-process';
import {
  projectMeshTo3D,
  noticeUndecidableWinding,
  type ProjectedMeshData,
} from '../../mesh/projection';
import { log, Modules } from '../../../utils/log';
import { computeTolerance } from '../../loaders';
import { EXTEND_TO_ALL_TOLERANCE } from '../view-state/extend-tolerance';
import type { LoadedMeshData, MeshMetadata, MeshViewState } from '../../../types/mesh';

/** Everything `commitMeshGeometry` needs, staged and ready for GPU upload. */
export interface StagedMeshCommit {
  path: string;
  /** The whole loaded mesh — held so the commit can read colours and counts. */
  data: LoadedMeshData;
  /** The projection for this epoch. */
  projected: ProjectedMeshData;
}

/**
 * Nodes already warned about undecidable winding.
 *
 * Module-scoped so the notice is once per node for the lifetime of the tab rather
 * than once per rebuild — the projection runs on every slice move, and a per-call
 * warning would flood the console during a scrub. A `Set` of paths is enough: paths
 * are unique per scene, and re-warning after a dataset switch is harmless.
 */
const noticedWinding = new Set<string>();

/**
 * Project loaded mesh data for the given view state and stage it for commit.
 *
 * @param attrs - The node's metadata; `normal_dims` supplies the winding frame,
 *   `double_sided` the authored side, and `extend_to_all` the extended
 *   (slice-invariant) dimensions whose membership slab is infinite.
 */
export async function processMeshData(
  path: string,
  data: LoadedMeshData,
  viewState: MeshViewState,
  attrs: Pick<MeshMetadata, 'normal_dims' | 'double_sided' | 'extend_to_all'>
): Promise<StagedMeshCommit> {
  // The whole-triangle cull is a MEMBERSHIP gate applied after the node is fully
  // resident, so it must run on the mesh's own per-dimension slab tolerance — the
  // half-cell for a discrete hidden dim, `step × meshSlabTolerance` for a continuous
  // one — NOT the navigation ride-along `viewState.tolerance` (`simpleDimsToViewState`'s
  // flat 0.5 for discrete dims and the scene `maxRadius`, a point-radius quantity
  // unrelated to a mesh's cell size, for continuous ones). Recompute it here, exactly
  // as `processLinesData` does for the lines clipping slab. `computeTolerance('mesh', …)`
  // always uses the membership role — mesh has no query path.
  let tolerance = computeTolerance('mesh', viewState.displayDims, data.ndim, viewState.dimensions);

  // Re-apply extend_to_all: an extended dim is slice-invariant, so its slab is
  // infinite. Mirrors the lines processor — the fresh recompute above dropped the
  // sentinel the derived view state carried.
  const extendDims = attrs.extend_to_all ?? [];
  if (extendDims.length > 0 && viewState.dimensions) {
    const dims = viewState.dimensions;
    tolerance = [...tolerance];
    for (const dimName of extendDims) {
      const dimIndex = dims.findIndex((d: { name?: string }) => d.name === dimName);
      if (dimIndex >= 0 && dimIndex < tolerance.length) {
        tolerance[dimIndex] = EXTEND_TO_ALL_TOLERANCE;
      }
    }
  }

  const backend = await getMeshBackend(data.ndim);
  const projected = projectMeshTo3D(
    data,
    { ...viewState, tolerance },
    attrs.normal_dims,
    attrs.double_sided,
    backend
  );

  // Reported here rather than inside `resolveWinding`, which stays pure so it can
  // be called on every index build without a logging side effect.
  if (projected.undecidableReason) {
    noticeUndecidableWinding(path, projected.undecidableReason, noticedWinding);
  }

  noticeContinuousHiddenDim(path, viewState, data.ndim, extendDims);

  return { path, data, projected };
}

/** Dedup key for the continuous-hidden-dimension notice. NUL cannot occur in a name or unit. */
function noticeKey(path: string, described: string): string {
  return `${path}\u0000${described}`;
}

/**
 * Which continuous hidden dimensions have already been reported, per node.
 *
 * Keyed by `path` **and the dimension as the message describes it** — name and unit — not
 * by path alone. Path alone would defeat the point of the notice: it exists to COLLECT
 * evidence about which axes turn up hidden-and-continuous on real data, and a 4D mesh that
 * first reports a continuous time axis would then have the early return suppress a
 * continuous Z forever once `displayDims` changed. The one configuration the measurement is
 * actually looking for is the one it would never see. The unit is in the key for the same
 * reason it is in the message: it is half the evidence, so a same-named axis that comes
 * back with a different unit after a dataset switch still has something to say.
 *
 * Still module-scoped and still per-node for a given dimension, so a scrub warns once. That
 * once-per-tab granularity matches `noticedWinding` above, and is the right one here: the
 * question this measures is whether the configuration occurs at all, not how often.
 */
const noticedContinuousHidden = new Set<string>();

/**
 * Report — at `info`, once per node and hidden dimension — a mesh whose hidden dims include
 * a CONTINUOUS one. A node's not-yet-reported dimensions are named together in ONE line
 * rather than one line each; the dedup is per (node, dimension), so an axis that becomes
 * hidden later still gets its own line.
 *
 * This is the measurement behind spec §9's deferral of exact nD triangle clipping. §5
 * culls whole triangles by per-vertex slab membership, which is a true cut when the
 * hidden dims are discrete (time, channel — the dominant real case, and why the cheap
 * kernel was chosen) but only a THICK SLAB when a hidden dim is continuous and spatial.
 * Exact clipping would fix that, at roughly 1500 LOC across two backends.
 *
 * The spec's promotion condition is "if continuous hidden spatial dims turn out to be a
 * real use case" — which was unfalsifiable, because nothing measured it. This line is
 * the experiment. Promote exact clipping when it starts appearing on real data; leave it
 * deferred while it does not.
 *
 * **Hidden-and-continuous tracks §9's "continuous hidden spatial" condition closely**, so the
 * filter below is not a loose proxy for it. `DimensionMetadata.spatial` does exist, but the
 * writer's own invariant makes it near-redundant here: `core/dimensions.py` forces a dimension
 * AUTHORED non-displayed and non-spatial to be discrete, so an axis authored hidden can only
 * reach this loop with `spatial=true`. The one gap is runtime rather than authoring — a dimension
 * authored `display=true, spatial=false` keeps the flag false and becomes hidden the moment the
 * user swaps display axes. That argues for reporting it, not filtering it out: testing the flag
 * would narrow nothing on the authored case while dropping evidence both there and from any
 * scene whose metadata omits it.
 *
 * The name and unit are reported because one judgement is genuinely left over that no flag
 * can make: a dimension may be *declared* spatial and still be a time axis, where a slab is
 * a reasonable thing to want. Naming it is what lets a reader tell that benign case from the
 * continuous Z the promotion condition is actually about.
 *
 * `info`, not `warning`: nothing is wrong. The slab is the documented behaviour, and the
 * node renders correctly under it.
 */
function noticeContinuousHiddenDim(
  path: string,
  viewState: MeshViewState,
  ndim: number,
  extendDims: readonly string[]
): void {
  const dims = viewState.dimensions;
  if (!dims) return;

  const displayed = new Set(viewState.displayDims);
  const extended = new Set(extendDims);
  const fresh: string[] = [];
  for (let i = 0; i < ndim && i < dims.length; i++) {
    const dim = dims[i];
    // An extended dim has an infinite slab, so the approximation cannot bite there.
    if (displayed.has(i) || dim?.discrete || extended.has(dim?.name ?? '')) continue;
    // The message text IS the dedup key, so the two cannot drift apart. Per (node,
    // dimension): a dimension that becomes hidden later is NEW evidence and must still be
    // reported even though this node has already been noticed once.
    const described = `${dim?.name ?? `dim${i}`}${dim?.unit ? ` [${dim.unit}]` : ''}`;
    if (noticedContinuousHidden.has(noticeKey(path, described))) continue;
    fresh.push(described);
  }
  if (fresh.length === 0) return;

  for (const described of fresh) noticedContinuousHidden.add(noticeKey(path, described));
  log.info(
    Modules.SCENE_LOADER,
    `Mesh ${path} has continuous hidden dimension(s) ${fresh.join(', ')}. Triangles ` +
      'are culled by whole-triangle slab membership, not clipped, so the surface shown is ' +
      'a slab of finite thickness rather than an exact cross-section ' +
      '(MESH_NODE_SPEC.md §5.2.1). Correct and intended; noted because exact nD clipping ' +
      'is deferred until this configuration shows up on real data (§9).'
  );
}

/** Test seam: forget every notice this module has already emitted. */
export function resetMeshNoticesForTesting(): void {
  noticedWinding.clear();
  noticedContinuousHidden.clear();
}
