/**
 * Narrow context shared by the per-geometry leaf loaders
 * (`load-points-node`, `load-lines-node`, `load-gsplats-node`).
 *
 * Bundles the snapshot of orchestrator state that the leaves read
 * (`viewState`, `factoryDeps`, registry, nodeFactory) plus the
 * orchestrator-side callbacks they need (attrs composition, view-state
 * derivation, monitor wiring, geometry commit). Never carries `this`.
 *
 * The orchestrator builds one via `makeNodeBuildCtx()` per call into
 * `loadSceneNodes` — re-snapshotting each time keeps the leaves immune
 * to mid-flight viewState mutation by a concurrent updateView.
 */

import type { LoaderRegistry } from '../loaders/loader-registry';
import type { LoaderFactoryDeps } from '../loaders/loader-factory';
import type { NodeFactory } from '../../../rendering/node-factory';
import type { LODGroupRegistry } from '../../../scene/lod-group-registry';
import type { SceneNode, ViewState, LoadedPointsData, DataLoader } from '../../data-loader-types';
import type { LinesDataLoader, LinesViewState, LoadedLinesData } from '../../../types/lines';
import type {
  GSplatsDataLoader,
  GSplatsViewState,
  LoadedGSplatsData,
} from '../../../types/gsplats';
import type { UpdateSession } from '../../../profiling/update-profiler';
import type { DerivedNodeViewState, DeriveOpts } from '../view-state/derive-node-view-state';
import type { StagedLinesCommit } from '../process/data-processor-lines';
import type { StagedPointsCommit } from '../process/data-processor-points';
import type { StagedGSplatsCommit } from '../process/data-processor-gsplats';

export interface NodeBuildCtx {
  /** Shared loader bookkeeping (registration + failure recording). */
  registry: LoaderRegistry;
  /**
   * Per-scene LOD-group registry. Optional — when absent, lod_group
   * nodes still load (default level renders) but the per-frame
   * selector is a no-op. Set by SceneLoader when the scene supports
   * LOD-group selection.
   */
  lodGroupRegistry?: LODGroupRegistry;
  /** THREE.js node factory for placeholder construction. */
  nodeFactory: NodeFactory;
  /**
   * Snapshot of the orchestrator's `viewState` at the time the leaf is
   * invoked. Captured by value so a concurrent updateView mutating the
   * orchestrator's field doesn't corrupt the initial-load query region.
   *
   * NOTE: for a DEFERRED reload (a lazy lod_group level re-fired after the user
   * scrubbed) this snapshot is stale — the reload paths re-run
   * `deriveNodeViewState()` (from the orchestrator's live view state) to load
   * for the CURRENT slice.
   */
  viewState: ViewState;
  /**
   * The orchestrator's CURRENT view-update version (live, not the snapshot). A
   * deferred / registry-driven reload captures this at derive-time and stamps
   * the committed geometry with it (see the commit callbacks below) so the LOD
   * registry's freshness check reflects the slice actually loaded.
   */
  getViewVersion(): number;
  /** Factory-deps snapshot for `createX*Loader` helpers. */
  factoryDeps: LoaderFactoryDeps;
  /** Compose effective rendering attrs along the scene-graph ancestry. */
  applyEffectiveAttrs(node: SceneNode): SceneNode['attrs'];
  /** Derive the per-node view state (same single source of truth used by retry/update). */
  deriveNodeViewState(
    path: string,
    attrs: { extend_to_all?: string[] } | undefined,
    opts: DeriveOpts
  ): DerivedNodeViewState;
  /** Wire a loader to the data-loading monitor when one is connected. */
  connectLoaderToMonitor(
    path: string,
    loader: DataLoader | LinesDataLoader | GSplatsDataLoader
  ): void;
  /**
   * Kick the progressive refinement orchestrator if no update holds the
   * serialization lock (re-checks next frame when one does). Called after a
   * deferred lod_group SUBTREE activation registers new sweep loaders —
   * refinement is otherwise only scheduled at update-view tails, so without
   * this the freshly-activated branch would sit at its first additive chunk
   * per part until the next slice change. See
   * ``SceneLoader.kickRefinementIfIdle``.
   */
  kickRefinementIfIdle(): void;
  /**
   * True while the dataset that created this ctx is still the live one.
   * Returns false once that dataset has been aborted/disposed or
   * replaced by a later `loadScene`. Deferred loads (e.g. lazily-loaded
   * lod_group levels triggered after initial load returns) MUST check
   * this before committing geometry, so a load in flight when the user
   * switches datasets never writes into a disposed/replaced scene.
   */
  isDatasetLive(): boolean;
  /**
   * Release a lazily-loaded gsplats level's GPU geometry back to the
   * evictable buffer pool and unregister its loader. Called when the
   * lod_group selector swaps away from a substitutive level, so resident
   * geometry stays bounded to ≈ the visible set rather than accumulating
   * every level ever shown. The raw chunks remain in the decompressed
   * cache, so re-selection re-projects cheaply (no network).
   */
  releaseLazyGSplats(path: string): void;

  /**
   * Release a lazily-loaded points level's GPU geometry back to the evictable
   * buffer pool and unregister its loader. Peer of :meth:`releaseLazyGSplats`
   * for points lod-group children (the finest level of a points-substitutive
   * ladder). Raw chunks remain in the decompressed cache, so re-selection
   * re-projects cheaply.
   */
  releaseLazyPoints(path: string): void;

  /**
   * Release a lazily-loaded lines level's GPU geometry back to the evictable
   * buffer pool and unregister its loader. Peer of :meth:`releaseLazyPoints` /
   * :meth:`releaseLazyGSplats` for lines lod-group children (the finest level of
   * a lines-substitutive ladder).
   */
  releaseLazyLines(path: string): void;

  // Per-type `process`/`commit` pairs — each leaf only uses the one for its
  // type. ``loadedViewVersion`` stamps the committed mesh for the LOD freshness
  // check; omit it to default to the live ``_updateVersion`` (correct for the
  // sweep), or pass the derive-time version from a deferred reload.
  processPointsData(path: string, data: LoadedPointsData): StagedPointsCommit;
  commitPointsGeometry(
    staged: StagedPointsCommit,
    session?: UpdateSession,
    loadedViewVersion?: number
  ): void;
  processLinesData(
    path: string,
    data: LoadedLinesData,
    viewState: LinesViewState,
    session?: UpdateSession
  ): Promise<StagedLinesCommit | null>;
  commitLinesGeometry(
    staged: StagedLinesCommit,
    session?: UpdateSession,
    loadedViewVersion?: number
  ): void;
  processGSplatsData(
    path: string,
    data: LoadedGSplatsData,
    viewState: GSplatsViewState,
    session?: UpdateSession
  ): Promise<StagedGSplatsCommit | null>;
  commitGSplatsGeometry(
    staged: StagedGSplatsCommit,
    session?: UpdateSession,
    loadedViewVersion?: number
  ): void;
}
