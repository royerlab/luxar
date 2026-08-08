/**
 * Initial-load path for a single Mesh leaf node.
 *
 * Builds the whole-node loader, attaches an empty placeholder so commit / retry /
 * update always have a target, then routes the fetch through the same
 * `deriveNodeViewState` + `processMeshData` / `commitMeshGeometry` path the update
 * sweep and the retry path use. Structurally the same as `load-points-node.ts`,
 * with two differences that follow from mesh having no LOD:
 *
 * - **No progressive branch.** `n_additive_sublods` on a mesh is a malformed store;
 *   `createProgressiveMeshLoader` rejects it with an explanation. That rejection is
 *   NOT dead code and must not be "tidied": a mesh can be a SUBSTITUTIVE level (see
 *   below), and substitutive levels are siblings under a `kind=lod` group — an
 *   additive ladder would be `additive_<i>/` subgroups inside this leaf, which is a
 *   different thing and still impossible for a surface.
 *
 * It DOES have the cheap/expensive split, and gained it late. The sibling loaders
 * split so a lazily-activated `kind=lod` level can attach its placeholder up front
 * and defer the costly geometry load; a mesh could not be a LOD-group child until
 * `luxar.mesh.decimate` gave it a producer, so the split had no second caller and
 * would have been machinery for one. Now the finest level of a mesh ladder is the
 * full-resolution surface, and without deferral every level — including that one —
 * would be fetched eagerly at scene load, which defeats the entire point of the
 * ladder.
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

/** What {@link loadMeshNodeCheap} hands the caller. Mirrors `PointsCheapLoad`. */
export interface MeshCheapLoad {
  /** The empty placeholder mesh, already attached to the parent. */
  placeholder: THREE.Mesh;
  /** The constructed loader (NOT yet registered — the expensive half does that). */
  loader: MeshDataLoader;
}

/**
 * Cheap half: build the loader and attach an empty placeholder, fetching nothing.
 *
 * A deferred `kind=lod` level runs only this at scene load, so a four-level mesh
 * ladder costs four placeholders instead of four full surfaces.
 */
export async function loadMeshNodeCheap(
  node: SceneNode,
  parentThree: THREE.Object3D,
  loc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): Promise<MeshCheapLoad> {
  log.custom('🔺', Modules.SCENE_LOADER, `Loading mesh: ${node.path}`);
  log.info(
    Modules.SCENE_LOADER,
    `  ${String(node.attrs.n_vertices ?? 'unknown')} vertices, ` +
      `${String(node.attrs.n_faces ?? 'unknown')} faces`
  );

  const loader = createMeshLoader(node, loc, ctx.factoryDeps) as MeshDataLoader;
  ctx.connectLoaderToMonitor(node.path, loader);

  // Attach the placeholder BEFORE fetching, so an initial-load failure leaves a
  // recoverable scene state that `retryFailedLoader` can write into.
  const attrs = ctx.applyEffectiveAttrs(node) as unknown as MeshMetadata;
  // The RAW leaf attrs ride along too: `resolveColormapWindow` needs both bags to
  // tell a leaf-authored scalar window from an inherited ancestor gain (#936).
  const placeholder = ctx.nodeFactory.createEmptyMeshNode(
    node.path,
    attrs,
    loader,
    node.attrs as Partial<MeshMetadata>
  );
  parentThree.add(placeholder);
  return { placeholder, loader };
}

/**
 * Expensive half: fetch, project and commit.
 *
 * Split out so a deferred LOD level can run it on first activation. Reads the
 * effective attrs again rather than threading them from the cheap half — a level
 * can be activated long after its placeholder was attached, and the Layers panel
 * may have changed them in between.
 *
 * Deliberately does NOT register the loader — that is the EAGER caller's job (see
 * {@link loadMeshNode}), exactly as in the three sibling loaders. A lazy LOD level
 * must stay out of the per-slice update sweep: the registry drives its reloads on a
 * settled slice change, and a registered level would additionally be re-fetched and
 * re-committed by the sweep on every scrub — including while it is hidden, and
 * concurrently with the registry's own `ensureLoaded`. Gating each scrub on
 * projecting the full-resolution surface is precisely what deferring it avoids.
 */
export async function loadMeshNodeExpensive(
  node: SceneNode,
  ctx: NodeBuildCtx,
  loader: MeshDataLoader
): Promise<void> {
  const attrs = ctx.applyEffectiveAttrs(node) as unknown as MeshMetadata;
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
    if (!ctx.isDatasetLive()) return;

    const staged = await ctx.processMeshData(node.path, data, derived.viewState, attrs);
    // Re-checked after the second await: projection is async, so the dataset can
    // die between the fetch and the commit.
    if (!ctx.isDatasetLive()) return;
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
    if (!ctx.isDatasetLive()) return;
    ctx.registry.recordFailure(node.path, error as Error);
    throw new LoaderError(classifyLoaderError(error), node.path, error);
  }
}

/**
 * Load a single Mesh node on initial scene construction (the EAGER path).
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
  const { placeholder, loader } = await loadMeshNodeCheap(node, parentThree, loc, ctx);
  try {
    await loadMeshNodeExpensive(node, ctx, loader);
  } finally {
    // Register only once the initial load has SETTLED, success or failure, and
    // only on THIS path — see the note on `loadMeshNodeExpensive`. Registering
    // before the await would let a concurrent updateView sweep run on the same
    // instance mid-flight; registering on failure too is deliberate, so a failed
    // initial load stays retryable through `retryFailedLoader`.
    ctx.registry.registerMeshLoader(node.path, loader);
  }
  return placeholder;
}
