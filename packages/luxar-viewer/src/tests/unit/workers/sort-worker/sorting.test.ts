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

  it('sortNode reports the kernelMs/workerMs timing split', () => {
    registerNode(ctx, { nodeId: 'n1', generation: 1, centers3: threeSplats(), count: 3 });
    const result = sortNode(ctx, { nodeId: 'n1', generation: 1, modelView: IDENTITY_MV });
    expect(result).not.toBeNull();
    // Both finite and non-negative...
    expect(Number.isFinite(result!.kernelMs)).toBe(true);
    expect(Number.isFinite(result!.workerMs)).toBe(true);
    expect(result!.kernelMs).toBeGreaterThanOrEqual(0);
    expect(result!.workerMs).toBeGreaterThanOrEqual(0);
    // ...and the kernel is a sub-interval of the whole body.
    expect(result!.kernelMs).toBeLessThanOrEqual(result!.workerMs);
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
    // Register with a live backend (registration now CONSTRUCTS the
    // backend-resident sorter, so it needs one), then simulate the
    // backend vanishing before the sort call.
    registerNode(ctx, { nodeId: 'n1', generation: 1, centers3: threeSplats(), count: 3 });
    ctx.wasm = null;
    expect(() => sortNode(ctx, { nodeId: 'n1', generation: 1, modelView: IDENTITY_MV })).toThrow(
      NOT_INITIALIZED_MSG
    );
  });

  it('registerNode throws the not-initialized error when wasm is missing', () => {
    // Since the sorter became backend-resident, registration itself
    // requires the backend (the coordinator always awaits initialize()
    // before its first register RPC).
    ctx.wasm = null;
    expect(() =>
      registerNode(ctx, { nodeId: 'n1', generation: 1, centers3: threeSplats(), count: 3 })
    ).toThrow(NOT_INITIALIZED_MSG);
  });

  describe('sorter lifecycle (leak safety)', () => {
    /** Wrap create_depth_sorter to capture every handle + spy its free(). */
    function trackSorters(): { frees: () => number[]; handles: () => number } {
      const wasm = ctx.wasm!;
      const freed: boolean[] = [];
      const original = wasm.create_depth_sorter.bind(wasm);
      let created = 0;
      vi.spyOn(wasm, 'create_depth_sorter').mockImplementation((centers3, count) => {
        const handle = original(centers3, count);
        const index = created++;
        freed.push(false);
        const realFree = handle.free.bind(handle);
        handle.free = () => {
          freed[index] = true;
          realFree();
        };
        return handle;
      });
      return {
        frees: () => freed.map((f, i) => (f ? i : -1)).filter((i) => i >= 0),
        handles: () => created,
      };
    }

    it('re-registration frees the replaced sorter (no leak on generation bump)', () => {
      const tracker = trackSorters();
      registerNode(ctx, { nodeId: 'n1', generation: 1, centers3: threeSplats(), count: 3 });
      registerNode(ctx, { nodeId: 'n1', generation: 2, centers3: threeSplats(), count: 3 });
      expect(tracker.handles()).toBe(2);
      expect(tracker.frees()).toEqual([0]); // old sorter freed, new one live
      // The replacement sorter is fully functional.
      const result = sortNode(ctx, { nodeId: 'n1', generation: 2, modelView: IDENTITY_MV });
      expect(Array.from(result!.ordering)).toEqual([0, 2, 1]);
    });

    it('releaseNode frees the sorter', () => {
      const tracker = trackSorters();
      registerNode(ctx, { nodeId: 'n1', generation: 1, centers3: threeSplats(), count: 3 });
      releaseNode(ctx, 'n1');
      expect(tracker.frees()).toEqual([0]);
      // Releasing an unknown node stays a safe no-op.
      releaseNode(ctx, 'ghost');
    });

    it('releaseAllNodes frees every sorter', () => {
      const tracker = trackSorters();
      registerNode(ctx, { nodeId: 'a', generation: 1, centers3: threeSplats(), count: 3 });
      registerNode(ctx, { nodeId: 'b', generation: 1, centers3: threeSplats(), count: 3 });
      registerNode(ctx, { nodeId: 'c', generation: 1, centers3: threeSplats(), count: 3 });
      releaseAllNodes(ctx);
      expect(tracker.frees()).toEqual([0, 1, 2]);
      expect(ctx.nodes.size).toBe(0);
    });

    it('a failing sorter construction keeps the previous registration intact', () => {
      registerNode(ctx, { nodeId: 'n1', generation: 1, centers3: threeSplats(), count: 3 });
      const survivor = ctx.nodes.get('n1')!.sorter;
      const freeSpy = vi.spyOn(survivor, 'free');
      vi.spyOn(ctx.wasm!, 'create_depth_sorter').mockImplementation(() => {
        throw new Error('simulated wasm OOM');
      });
      expect(() =>
        registerNode(ctx, { nodeId: 'n1', generation: 2, centers3: threeSplats(), count: 3 })
      ).toThrow('simulated wasm OOM');
      // Old sorter neither freed nor replaced — generation-1 sorts still work.
      expect(freeSpy).not.toHaveBeenCalled();
      expect(ctx.nodes.get('n1')!.sorter).toBe(survivor);
      const result = sortNode(ctx, { nodeId: 'n1', generation: 1, modelView: IDENTITY_MV });
      expect(Array.from(result!.ordering)).toEqual([0, 2, 1]);
    });
  });
});
