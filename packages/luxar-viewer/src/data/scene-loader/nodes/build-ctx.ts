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
   */
  viewState: ViewState;
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

  // Per-type commit callbacks — each leaf only uses the one for its type.
  updatePointsGeometry(path: string, data: LoadedPointsData, session?: UpdateSession): void;
  processLinesData(
    path: string,
    data: LoadedLinesData,
    viewState: LinesViewState,
    session?: UpdateSession
  ): Promise<StagedLinesCommit | null>;
  commitLinesGeometry(staged: StagedLinesCommit, session?: UpdateSession): void;
  processGSplatsData(
    path: string,
    data: LoadedGSplatsData,
    viewState: GSplatsViewState,
    session?: UpdateSession
  ): Promise<StagedGSplatsCommit | null>;
  commitGSplatsGeometry(staged: StagedGSplatsCommit, session?: UpdateSession): void;
}
