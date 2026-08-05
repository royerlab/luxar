/**
 * Initial-load path for a single Mesh leaf node.
 *
 * Builds the whole-node loader, attaches an empty placeholder so commit / retry /
 * update always have a target, then routes the fetch through the same
 * `deriveNodeViewState` + `processMeshData` / `commitMeshGeometry` path the update
 * sweep and the retry path use. Structurally the same as `load-points-node.ts`,
 * with two differences that follow from mesh having no LOD:
 *
 * - **No cheap/expensive split.** The sibling loaders split so a lazily-activated
 *   `kind=lod` level can attach its placeholder up front and defer the costly
 *   geometry load. A mesh can never be a LOD-group child (§9, and the Python writer
 *   refuses it), so there is no deferred-activation caller and the split would be
 *   machinery with one call site.
 * - **No progressive branch.** `n_additive_sublods` on a mesh is a malformed store;
 *   `createProgressiveMeshLoader` rejects it with an explanation.
 *
 * @module data/scene-loader/nodes/load-mesh-node
 */

import type * as THREE from 'three';
import type * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';
import { LoaderError, classifyLoaderError } from './load-leaf-error-dispatch';
import { createMeshLoader } from '../loaders/loader-factory';
import type { SceneNode } from '../../data-loader-types';
import type { MeshDataLoader, MeshMetadata } from '../../../types/mesh';
import type { NodeBuildCtx } from './build-ctx';

/**
 * Load a single Mesh node on initial scene construction.
 *
 * Returns the placeholder once data has been committed (or empty when no triangles
 * are currently visible — the slice-update path will populate it later). Throws
 * `LoaderError` on failure so `loadLeafNode` can dispatch by kind and keep the rest
 * of the scene alive.
 */
export async function loadMeshNode(
  node: SceneNode,
  parentThree: THREE.Object3D,
  loc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): Promise<THREE.Object3D | null> {
  log.custom('🔺', Modules.SCENE_LOADER, `Loading mesh: ${node.path}`);
  log.info(
    Modules.SCENE_LOADER,
    `  ${String(node.attrs.n_vertices ?? 'unknown')} vertices, ` +
      `${String(node.attrs.n_faces ?? 'unknown')} faces`
  );

  const loader = createMeshLoader(node, loc, ctx.factoryDeps);
  ctx.connectLoaderToMonitor(node.path, loader);

  // Attach the placeholder BEFORE fetching, so an initial-load failure leaves a
  // recoverable scene state that `retryFailedLoader` can write into.
  const attrs = ctx.applyEffectiveAttrs(node) as unknown as MeshMetadata;
  const placeholder = ctx.nodeFactory.createEmptyMeshNode(node.path, attrs, loader);
  parentThree.add(placeholder);

  try {
    // Through `deriveNodeViewState` like every other path, so initial / update /
    // retry can never silently load different query regions. Mesh takes the
    // partial-extend tolerance (unlike Lines, which opts out because its segment
    // bounds already encode the non-displayed extent — a mesh has no such bounds).
    const derived = ctx.deriveNodeViewState(node.path, node.attrs, {
      applyPartialExtendTolerance: true,
    });
    // Captured at DERIVE time so the committed geometry is stamped for the slice it
    // actually loaded, matching the sibling loaders.
    const loadedViewVersion = ctx.getViewVersion();

    const data = await (loader as MeshDataLoader).loadMesh(derived.viewState);

    // Liveness gate: this can resolve after a dataset switch disposed the scene.
    // Committing then would write into a stale root group, so drop it silently —
    // the abort-discard policy the sibling loaders use.
    if (!ctx.isDatasetLive()) return placeholder;

    const staged = await ctx.processMeshData(node.path, data, derived.viewState, attrs);
    // Re-checked after the second await: projection is async, so the dataset can
    // die between the fetch and the commit.
    if (!ctx.isDatasetLive()) return placeholder;
    ctx.commitMeshGeometry(staged, undefined, loadedViewVersion);

    // A settled successful load clears any prior failure record, so a recovered
    // node stops counting against the outcome report and the auto-retry budget.
    // Unconditional like the points twin: staging a mesh always yields a commit,
    // so there is no landed signal to gate on.
    ctx.registry.clearFailure(node.path);
    log.success(
      Modules.SCENE_LOADER,
      `Loaded mesh ${node.path}: ${staged.projected.visibleFaceCount.toLocaleString()} of ` +
        `${data.faceCount.toLocaleString()} triangles visible`
    );
  } catch (error) {
    // Expected dispose-crossing: a read that passed its abort check can still throw
    // a non-AbortError against a torn-down store. The dataset is dead, so a failure
    // record and an error-level LoaderError would be pure noise (and a spurious
    // toast). Symmetric with the liveness gate above.
    if (!ctx.isDatasetLive()) return placeholder;
    ctx.registry.recordFailure(node.path, error as Error);
    throw new LoaderError(classifyLoaderError(error), node.path, error);
  } finally {
    // Register only once the initial load has SETTLED, success or failure.
    // Registering before the await would let a concurrent updateView sweep run on
    // the same instance mid-flight; registering on failure too is deliberate, so a
    // failed initial load stays retryable.
    ctx.registry.registerMeshLoader(node.path, loader);
  }

  return placeholder;
}
