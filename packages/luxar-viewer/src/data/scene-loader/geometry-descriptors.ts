/**
 * One table describing what each geometry kind needs from the shared
 * scene-loader machinery.
 *
 * Three sites used to switch on the geometry type by hand — `loadSceneNodes`,
 * `retryFailedLoaderUnlocked` and `SlicePrefetcher.getShadow`. Each wrote the
 * same three-way branch with slightly different shapes, and the retry chain in
 * particular expressed a per-type capability as the *presence of an `if`*: a
 * kind with no arm there is not a compile error, it is a load that can never be
 * retried, manually or on reconnect.
 *
 * `Record<GeometryKind, GeometryDescriptor>` makes that structural instead —
 * the vocabulary comes from the format contract, so a kind added there without
 * an entry here fails the build.
 *
 * Only capabilities with a real consumer live here. Deliberately absent: any
 * "supports X" flag that every kind currently answers the same way, since a
 * field no branch reads is indistinguishable from a field that is wrong.
 *
 * @module data/scene-loader/geometry-descriptors
 */

import type * as THREE from 'three';
import type * as zarr from '../zarr';
import type { GeometryKind, SceneNode, ViewState, DataLoader } from '../data-loader-types';
import type { LinesDataLoader, LinesViewState } from '../../types/lines';
import type { GSplatsDataLoader, GSplatsViewState } from '../../types/gsplats';
import type { AnyDataLoader } from './loaders/loader-registry';
import type { LoaderFactoryDeps } from './loaders/loader-factory';
import type { NodeBuildCtx } from './nodes/build-ctx';
import type { RetryCtx } from './lifecycle/retry';
import { loadPointsNode } from './nodes/load-points-node';
import { loadLinesNode } from './nodes/load-lines-node';
import { loadGSplatsNode } from './nodes/load-gsplats-node';
import {
  createPointsLoader,
  createLinesLoader,
  createGSplatsLoader,
  createProgressivePointsLoader,
  createProgressiveLinesLoader,
  createProgressiveGSplatsLoader,
} from './loaders/loader-factory';

export interface GeometryDescriptor {
  /** Initial-load entry point for a leaf of this kind. */
  loadNode(
    node: SceneNode,
    parentThree: THREE.Object3D,
    parentLoc: zarr.Location<zarr.Readable>,
    ctx: NodeBuildCtx
  ): Promise<THREE.Object3D | null>;

  /**
   * Whether `deriveNodeViewState` applies the partial-extend tolerance for this
   * kind. Lines opt out: their segment bounds already encode the non-displayed
   * extent, so applying it again double-counts during clipping.
   */
  readonly applyPartialExtendTolerance: boolean;

  /**
   * Re-fetch and commit one failed node, given the view state the retry path
   * already derived. Owns the per-kind loader cast and the process/commit pair
   * so `retryFailedLoaderUnlocked` stays type-agnostic.
   */
  retryCommit(
    ctx: RetryCtx,
    path: string,
    loader: AnyDataLoader,
    viewState: ViewState
  ): Promise<void>;

  /** Non-progressive loader factory (shadow prefetch). */
  createLoader(
    node: SceneNode,
    loc: zarr.Location<zarr.Readable>,
    deps: LoaderFactoryDeps
  ): AnyDataLoader;

  /** Progressive (additive-ladder) loader factory. */
  createProgressiveLoader(
    node: SceneNode,
    nAdditive: number,
    effectiveAttrs: Record<string, unknown>,
    deps: LoaderFactoryDeps
  ): Promise<AnyDataLoader>;
}

/**
 * GSplats' view state is structurally the same as the base `ViewState` but is
 * spelled as an explicit object at every construction site, matching the shape
 * `loadGSplatsNode` builds on the initial-load path.
 */
function toGSplatsViewState(viewState: ViewState): GSplatsViewState {
  return {
    displayDims: viewState.displayDims,
    slicePosition: viewState.slicePosition,
    tolerance: viewState.tolerance,
    dimensions: viewState.dimensions,
  };
}

export const GEOMETRY_DESCRIPTORS: Record<GeometryKind, GeometryDescriptor> = {
  points: {
    loadNode: loadPointsNode,
    applyPartialExtendTolerance: true,
    async retryCommit(ctx, path, loader, viewState) {
      const data = await (loader as DataLoader).updateView(viewState);
      // No staged null-check, unlike lines/gsplats below: `processPointsData`
      // is synchronous and always yields a staged commit.
      if (data) ctx.commitPointsGeometry(ctx.processPointsData(path, data));
    },
    createLoader: createPointsLoader,
    createProgressiveLoader: createProgressivePointsLoader,
  },
  lines: {
    loadNode: loadLinesNode,
    applyPartialExtendTolerance: false,
    async retryCommit(ctx, path, loader, viewState) {
      const linesViewState = viewState as LinesViewState;
      const data = await (loader as LinesDataLoader).updateView(linesViewState);
      if (data) {
        const staged = await ctx.processLinesData(path, data, linesViewState);
        if (staged) ctx.commitLinesGeometry(staged);
      }
    },
    createLoader: createLinesLoader,
    createProgressiveLoader: createProgressiveLinesLoader,
  },
  gsplats: {
    loadNode: loadGSplatsNode,
    applyPartialExtendTolerance: true,
    async retryCommit(ctx, path, loader, viewState) {
      const gsplatsViewState = toGSplatsViewState(viewState);
      const data = await (loader as GSplatsDataLoader).updateView(gsplatsViewState);
      if (data) {
        const staged = await ctx.processGSplatsData(path, data, gsplatsViewState);
        if (staged) ctx.commitGSplatsGeometry(staged);
      }
    },
    createLoader: createGSplatsLoader,
    createProgressiveLoader: createProgressiveGSplatsLoader,
  },
};
