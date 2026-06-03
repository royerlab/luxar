/**
 * Direct tests for runGSplatsRefinement — the progressive GSplats LOD
 * refinement loop.
 *
 * End-to-end behaviour (rAF integration with real browser paint timing) is
 * covered by the e2e suite. The unit loop semantics — completion, lock
 * release, cancellation hand-off, error isolation, skip-path — are shared
 * across all three geometry types and live in
 * `../_shared/refinement-loop-contract`. GSplats has no behaviour beyond that
 * contract, so this file just binds the contract to `runGSplatsRefinement`.
 */

import { vi } from 'vitest';
import * as THREE from 'three';
import {
  runGSplatsRefinement,
  type GSplatsRefinementCtx,
} from '../../../../data/gsplats/lod-refinement';
import { defineRefinementLoopContract } from '../_shared/refinement-loop-contract';

defineRefinementLoopContract('runGSplatsRefinement', (w) =>
  runGSplatsRefinement({
    rootGroup: new THREE.Group(),
    viewStateQueue: w.viewStateQueue,
    gsplatLoaders: w.loaders as GSplatsRefinementCtx['gsplatLoaders'],
    deriveNodeViewState: w.deriveNodeViewState as GSplatsRefinementCtx['deriveNodeViewState'],
    processGSplats: w.processSpy as GSplatsRefinementCtx['processGSplats'],
    commitGSplats: vi.fn(),
    updateVisibleCountsInMonitor: w.updateVisibleCountsInMonitor,
    releaseLock: w.releaseLock,
    retriggerUpdate: w.retriggerUpdate,
  })
);
