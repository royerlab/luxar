/**
 * Worker-side SortWorker task tests (depth-sorting Phase 2, spec §5).
 *
 * Exercises the REAL task bodies against the TypeScript reference
 * backend (`TypeScriptFallback` satisfies `WasmModule`), so the
 * registration/generation/release contracts are tested without a live
 * worker. Comlink's `transfer()` in a non-worker context only tags the
 * value, so returned objects are directly inspectable.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('../../../../utils/log', () => ({
  log: { info: vi.fn(), warning: vi.fn(), error: vi.fn() },
  Modules: new Proxy({}, { get: (_t, p) => String(p) }),
}));

import { TypeScriptFallback } from '../../../../wasm/typescript';
import type { SortWorkerCtx } from '../../../../workers/sort-worker/state';
import {
  registerNode,
  sortNode,
  releaseNode,
  releaseAllNodes,
} from '../../../../workers/sort-worker/sorting';
import { NOT_INITIALIZED_MSG } from '../../../../workers/sort-worker/state';

const IDENTITY_MV = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

/** Centers with view z = -10, -1, -5 → back-to-front ordering [0, 2, 1]. */
function threeSplats(): Float32Array {
  return new Float32Array([0, 0, -10, 1, 0, -1, 2, 0, -5]);
}

describe('sort-worker sorting tasks', () => {
  let ctx: SortWorkerCtx;

  beforeEach(() => {
    ctx = { wasm: new TypeScriptFallback(), nodes: new Map() };
  });

  it('registerNode stores the node and sortNode returns a back-to-front ordering', () => {
    registerNode(ctx, { nodeId: 'n1', generation: 1, centers3: threeSplats(), count: 3 });
    const result = sortNode(ctx, { nodeId: 'n1', generation: 1, modelView: IDENTITY_MV });
    expect(result).not.toBeNull();
    expect(result!.generation).toBe(1);
    expect(Array.from(result!.ordering)).toEqual([0, 2, 1]);
  });

  it('sortNode returns null for an unknown node', () => {
    expect(sortNode(ctx, { nodeId: 'ghost', generation: 1, modelView: IDENTITY_MV })).toBeNull();
  });

  it('generation guard: a stale sort request returns null', () => {
    registerNode(ctx, { nodeId: 'n1', generation: 1, centers3: threeSplats(), count: 3 });
    // A newer commit re-registered the node...
    registerNode(ctx, { nodeId: 'n1', generation: 2, centers3: threeSplats(), count: 3 });
    // ...so the request issued for generation 1 is stale.
    expect(sortNode(ctx, { nodeId: 'n1', generation: 1, modelView: IDENTITY_MV })).toBeNull();
    // The current generation still sorts.
    expect(sortNode(ctx, { nodeId: 'n1', generation: 2, modelView: IDENTITY_MV })).not.toBeNull();
  });

  it('registerNode clamps count to the centers3 capacity', () => {
    registerNode(ctx, { nodeId: 'n1', generation: 1, centers3: threeSplats(), count: 99 });
    const result = sortNode(ctx, { nodeId: 'n1', generation: 1, modelView: IDENTITY_MV });
    expect(result!.ordering.length).toBe(3);
  });

  it('releaseNode drops the registration', () => {
    registerNode(ctx, { nodeId: 'n1', generation: 1, centers3: threeSplats(), count: 3 });
    releaseNode(ctx, 'n1');
    expect(sortNode(ctx, { nodeId: 'n1', generation: 1, modelView: IDENTITY_MV })).toBeNull();
  });

  it('releaseAllNodes drops every registration', () => {
    registerNode(ctx, { nodeId: 'a', generation: 1, centers3: threeSplats(), count: 3 });
    registerNode(ctx, { nodeId: 'b', generation: 1, centers3: threeSplats(), count: 3 });
    releaseAllNodes(ctx);
    expect(ctx.nodes.size).toBe(0);
  });

  it('sortNode throws the not-initialized error when wasm is missing', () => {
    ctx.wasm = null;
    registerNode(ctx, { nodeId: 'n1', generation: 1, centers3: threeSplats(), count: 3 });
    expect(() => sortNode(ctx, { nodeId: 'n1', generation: 1, modelView: IDENTITY_MV })).toThrow(
      NOT_INITIALIZED_MSG
    );
  });
});
