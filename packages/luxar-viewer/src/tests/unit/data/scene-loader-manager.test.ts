// @vitest-environment jsdom
/**
 * Tests for SceneLoaderManager singleton lifecycle.
 *
 * [architecture.md/O2][P10] Moved from tests/unit/architecture/global-state.test.ts —
 * `SceneLoaderManager` describe block is a behavioral singleton test for the
 * data layer; belongs alongside data tests, not under "architecture".
 *
 * Also asserts that managing loaders does not pollute the global `window`
 * object, originally the C1/C2 hardening of the audit.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { SceneLoaderManager, dispose } from '../../../data';
import { DataMonitorManager } from '../../../ui/data-monitor-manager';
import { RefinementDensityGate } from '../../../data/scene-loader/progressive/density-gate';
import {
  RESIDENCY_DECLINED_PATH_SAMPLE,
  type RefinementResidencyReporter,
} from '../../../data/scene-loader/progressive/residency-budget';
import type { PoolStats } from '../../../rendering/gpu-buffer-pool/pool-stats';

describe('SceneLoaderManager', () => {
  beforeEach(() => {
    SceneLoaderManager.disposeInstance();
    DataMonitorManager.disposeInstance();
  });

  afterEach(() => {
    SceneLoaderManager.disposeInstance();
    DataMonitorManager.disposeInstance();
  });

  it('should manage instances without global variables', () => {
    const manager = SceneLoaderManager.getInstance();

    // Should start with no loaders
    expect(manager.getLoaderCount()).toBe(0);
    expect(manager.getDefaultLoader()).toBeNull();

    // Create a loader
    const loader1 = manager.createLoader('test1');
    expect(manager.getLoaderCount()).toBe(1);
    expect(manager.getDefaultLoader()).toBe(loader1);

    // Create another loader
    const loader2 = manager.createLoader('test2', undefined, false);
    expect(manager.getLoaderCount()).toBe(2);
    expect(manager.getDefaultLoader()).toBe(loader1); // Should still be loader1

    // Get loaders by ID
    expect(manager.getLoader('test1')).toBe(loader1);
    expect(manager.getLoader('test2')).toBe(loader2);
    expect(manager.getLoader('nonexistent')).toBeNull();

    // Destroy a loader
    manager.destroyLoader('test1');
    expect(manager.getLoaderCount()).toBe(1);
    expect(manager.getDefaultLoader()).toBe(loader2); // Should switch to loader2
  });

  it('createLoaderAsync awaits prior loader disposal before building the replacement', async () => {
    const manager = SceneLoaderManager.getInstance();
    const first = manager.createLoader('switch');
    expect(manager.getLoader('switch')).toBe(first);

    // Make the prior loader's dispose hang on a deferred promise so we can
    // observe the ordering deterministically.
    let resolveDispose!: () => void;
    const disposeGate = new Promise<void>((resolve) => {
      resolveDispose = resolve;
    });
    let disposeStarted = false;
    (first as unknown as { dispose: () => Promise<void> }).dispose = () => {
      disposeStarted = true;
      return disposeGate;
    };

    // Kick off the async replacement. It must NOT resolve while the prior
    // loader's dispose is still in flight.
    let created: unknown = null;
    const pending = manager.createLoaderAsync('switch').then((loader) => {
      created = loader;
    });

    // Flush microtasks: destroyLoaderAsync should have detached the old loader
    // and begun awaiting its dispose, but the replacement is not built yet.
    await Promise.resolve();
    await Promise.resolve();
    expect(disposeStarted).toBe(true);
    expect(created).toBeNull();

    // Complete the prior dispose; only now may the replacement be constructed.
    resolveDispose();
    await pending;
    expect(created).not.toBeNull();
    expect(created).not.toBe(first);
    expect(manager.getLoader('switch')).toBe(created);
  });

  it('should use singleton pattern correctly', () => {
    const manager1 = SceneLoaderManager.getInstance();
    const manager2 = SceneLoaderManager.getInstance();

    // Should be the same instance
    expect(manager1).toBe(manager2);

    // Changes in one should be visible in the other
    manager1.createLoader('singleton-test');
    expect(manager2.hasLoader('singleton-test')).toBe(true);
  });

  it('should reset properly for testing', () => {
    const manager = SceneLoaderManager.getInstance();
    manager.createLoader('test');
    expect(manager.getLoaderCount()).toBe(1);

    // Reset should clear everything
    SceneLoaderManager.disposeInstance();

    // New instance should be empty
    const newManager = SceneLoaderManager.getInstance();
    expect(newManager.getLoaderCount()).toBe(0);
  });

  it('disposes the previous KTX2 decoder when replaced or reset', () => {
    const manager = SceneLoaderManager.getInstance();
    const first = Object.assign(vi.fn(), { dispose: vi.fn() });
    const second = Object.assign(vi.fn(), { dispose: vi.fn() });

    manager.setKTX2TextureDecoder(first);
    manager.setKTX2TextureDecoder(second);
    expect(first.dispose).toHaveBeenCalledOnce();

    SceneLoaderManager.disposeInstance();
    expect(second.dispose).toHaveBeenCalledOnce();
  });

  it('forwards the KTX2 decoder to each created SceneLoader', () => {
    const manager = SceneLoaderManager.getInstance();
    const decodeKTX2 = Object.assign(vi.fn(), { dispose: vi.fn() });

    manager.setKTX2TextureDecoder(decodeKTX2);
    const loader = manager.createLoader('ktx2');

    expect((loader as unknown as { decodeKTX2: unknown }).decodeKTX2).toBe(decodeKTX2);
  });

  it('forwards refinement density changes to existing and future SceneLoaders', () => {
    const manager = SceneLoaderManager.getInstance();
    const caps = { blendable: 4, nonBlendable: 1 };
    const gateOf = (loader: unknown) =>
      (loader as { refinementDensityGate: unknown }).refinementDensityGate;

    // No provider wired (guard off, embedders): bytes-only admission, no gate.
    const first = manager.createLoader('density-first');
    const second = manager.createLoader('density-second');
    expect(gateOf(first)).toBeNull();
    expect(gateOf(second)).toBeNull();

    manager.setRefinementDensityProvider(() => undefined, caps);
    expect(gateOf(first)).toBeInstanceOf(RefinementDensityGate);
    expect(gateOf(second)).toBeInstanceOf(RefinementDensityGate);
    expect(gateOf(manager.createLoader('density-future'))).toBeInstanceOf(RefinementDensityGate);

    manager.setRefinementDensityProvider(null, caps);
    expect(gateOf(first)).toBeNull();
    expect(gateOf(second)).toBeNull();
    expect(gateOf(manager.createLoader('density-after-off'))).toBeNull();
  });

  it('forwards retryable-failure notifications to existing and future SceneLoaders', () => {
    const manager = SceneLoaderManager.getInstance();
    const existing = manager.createLoader('existing');
    const notify = vi.fn();
    manager.setAutoRetryableFailureCallback(notify);
    const future = manager.createLoader('future');

    const registryOf = (loader: unknown) =>
      (
        loader as {
          registry: { recordFailure(path: string, error: Error): void };
        }
      ).registry;
    registryOf(existing).recordFailure('/existing', new Error('HTTP 503 fetching chunk'));
    registryOf(future).recordFailure('/future', new Error('HTTP 503 fetching chunk'));

    expect(notify).toHaveBeenCalledTimes(2);
  });

  it('should handle getAllLoaders correctly', () => {
    const manager = SceneLoaderManager.getInstance();

    // Initially empty
    expect(manager.getAllLoaders().size).toBe(0);

    // Create multiple loaders
    manager.createLoader('loader1');
    manager.createLoader('loader2');
    manager.createLoader('loader3');

    const allLoaders = manager.getAllLoaders();
    expect(allLoaders.size).toBe(3);
    expect(allLoaders.has('loader1')).toBe(true);
    expect(allLoaders.has('loader2')).toBe(true);
    expect(allLoaders.has('loader3')).toBe(true);
  });

  it('should check loader existence with hasLoader', () => {
    const manager = SceneLoaderManager.getInstance();

    expect(manager.hasLoader('nonexistent')).toBe(false);

    manager.createLoader('exists');
    expect(manager.hasLoader('exists')).toBe(true);

    manager.destroyLoader('exists');
    expect(manager.hasLoader('exists')).toBe(false);
  });

  it('should handle destroying all loaders', () => {
    const manager = SceneLoaderManager.getInstance();

    // Create multiple loaders
    manager.createLoader('loader1');
    manager.createLoader('loader2');
    manager.createLoader('loader3');
    expect(manager.getLoaderCount()).toBe(3);

    // Destroy all
    manager.destroyAll();
    expect(manager.getLoaderCount()).toBe(0);
    expect(manager.getDefaultLoader()).toBeNull();
  });

  it('should handle default loader switching when default is destroyed', () => {
    const manager = SceneLoaderManager.getInstance();

    const loader1 = manager.createLoader('loader1', undefined, true);
    const loader2 = manager.createLoader('loader2', undefined, false);
    const loader3 = manager.createLoader('loader3', undefined, false);

    expect(manager.getDefaultLoader()).toBe(loader1);

    // Destroy the default loader
    manager.destroyLoader('loader1');

    // Should auto-select next available loader as default
    const newDefault = manager.getDefaultLoader();
    expect(newDefault).toBeTruthy();
    expect(newDefault === loader2 || newDefault === loader3).toBe(true);
    // Marking `loader3` as referenced — its presence ensures we exercise the
    // multi-loader election path even though only loader2 is the expected pick.
    void loader3;
  });

  // #1639 — the debug snapshot's `isLoading` flag is sourced from this
  // aggregate, so it has to answer for EVERY registered loader: a viewer can
  // hold several, and a snapshot reading only the default would report "idle"
  // while a sibling loader was still fetching/decoding.
  describe('isAnyLoadPassInProgress', () => {
    /** Force a loader's in-flight answer without driving a real updateView. */
    function setLoadPassInProgress(loader: unknown, inProgress: boolean): void {
      (loader as { isLoadPassInProgress: () => boolean }).isLoadPassInProgress = () => inProgress;
    }

    it('is false when no loader is registered', () => {
      const manager = SceneLoaderManager.getInstance();
      expect(manager.getLoaderCount()).toBe(0);
      expect(manager.isAnyLoadPassInProgress()).toBe(false);
    });

    it('is false when every registered loader is idle', () => {
      const manager = SceneLoaderManager.getInstance();
      // Freshly-created loaders have no sweep in flight — read them as-is
      // rather than stubbing, so the idle case exercises the real method.
      manager.createLoader('idle-a');
      manager.createLoader('idle-b', undefined, false);
      expect(manager.isAnyLoadPassInProgress()).toBe(false);
    });

    it('is true when ANY ONE of several loaders is mid-load-pass', () => {
      const manager = SceneLoaderManager.getInstance();
      manager.createLoader('first');
      const second = manager.createLoader('second', undefined, false);
      manager.createLoader('third', undefined, false);

      // The busy loader is deliberately NOT the default one: the aggregate
      // must not be satisfiable by consulting `getDefaultLoader()` alone.
      setLoadPassInProgress(second, true);
      expect(manager.getDefaultLoader()).not.toBe(second);
      expect(manager.isAnyLoadPassInProgress()).toBe(true);

      // …and it drops back to false once that sweep finishes.
      setLoadPassInProgress(second, false);
      expect(manager.isAnyLoadPassInProgress()).toBe(false);
    });
  });

  // #2508 — the debug snapshot's two memory-ceiling signals come from these two
  // aggregates, and `capture-readiness.ts` refuses a capture on either. Same
  // lesson as `isAnyLoadPassInProgress` above: the manager's contract admits
  // several loaders even though production registers one, so what it does with
  // more than one is the part worth pinning. The two answer that question
  // DIFFERENTLY on purpose, which is exactly why neither may go untested.
  describe('refinementResidencyStop', () => {
    let warn: ReturnType<typeof vi.spyOn>;

    beforeEach(async () => {
      // The reporter logs its first refusal; silence it so a merge test does
      // not print a wall of warnings.
      const { log } = await import('../../../utils/log');
      warn = vi.spyOn(log, 'warning').mockImplementation(() => undefined);
    });
    afterEach(() => warn.mockRestore());

    /**
     * Decline `paths` on a loader's REAL reporter rather than stubbing the
     * record: the merge rule is only interesting over records the reporter can
     * actually produce.
     */
    function decline(
      loader: unknown,
      paths: readonly string[],
      verdict: { reason: 'over-budget' | 'next-rung-would-exceed'; budgetBytes: number }
    ): void {
      const reporter = (loader as { refinementResidencyReporter: RefinementResidencyReporter })
        .refinementResidencyReporter;
      for (const path of paths) {
        reporter.reportOnce(path, {
          admitted: false,
          reason: verdict.reason,
          residentBytes: verdict.budgetBytes + 1,
          budgetBytes: verdict.budgetBytes,
        });
      }
    }

    it('is undefined when no loader is registered', () => {
      const manager = SceneLoaderManager.getInstance();
      expect(manager.getLoaderCount()).toBe(0);
      expect(manager.refinementResidencyStop()).toBeUndefined();
    });

    it('is undefined when no registered loader ever declined a rung', () => {
      // Freshly-created loaders have declined nothing — read them as-is so the
      // "never stopped" case exercises the real reporter. Absence is the whole
      // signal downstream: a synthesised zero record would refuse every capture.
      const manager = SceneLoaderManager.getInstance();
      manager.createLoader('quiet-a');
      manager.createLoader('quiet-b', undefined, false);
      expect(manager.refinementResidencyStop()).toBeUndefined();
    });

    it('takes the FIRST loader’s verdict but SUMS the declined count', () => {
      const manager = SceneLoaderManager.getInstance();
      const first = manager.createLoader('stop-first');
      const second = manager.createLoader('stop-second', undefined, false);

      // The second loader deliberately holds the LARGER count and a different
      // verdict: the four verdict fields describe one decision at one instant,
      // so "whichever loader stopped hardest" is not an answer — averaging or
      // last-write-wins would describe a moment that never happened.
      decline(first, ['/a', '/b'], { reason: 'over-budget', budgetBytes: 10 });
      decline(second, ['/c', '/d', '/e', '/f', '/g'], {
        reason: 'next-rung-would-exceed',
        budgetBytes: 999,
      });

      expect(manager.refinementResidencyStop()).toMatchObject({
        reason: 'over-budget',
        residentBytes: 11,
        budgetBytes: 10,
        firstPath: '/a',
        // Additive, because "how much of the scene stopped short" genuinely is.
        declinedPathCount: 7,
      });
    });

    it('re-truncates the merged path sample so it cannot grow with the loader count', () => {
      const manager = SceneLoaderManager.getInstance();
      const first = manager.createLoader('sample-first');
      const second = manager.createLoader('sample-second', undefined, false);

      decline(first, ['/a', '/b'], { reason: 'over-budget', budgetBytes: 10 });
      decline(
        second,
        Array.from({ length: 20 }, (_, i) => `/part_${i}`),
        { reason: 'over-budget', budgetBytes: 10 }
      );

      const merged = manager.refinementResidencyStop();
      // Each loader's own sample is already capped, so concatenating N of them
      // would ship N x the bound across the `page.evaluate` boundary.
      expect(merged?.declinedPaths).toHaveLength(RESIDENCY_DECLINED_PATH_SAMPLE);
      // First-seen order survives the merge: the first loader's paths lead.
      expect(merged?.declinedPaths[0]).toBe('/a');
      // …and the COUNT is untouched by the truncation — it is the measurement.
      expect(merged?.declinedPathCount).toBe(22);
    });
  });

  describe('gpuPoolStats', () => {
    /**
     * Swap a loader's pool for a stand-in reporting `byteBudgetEvictions`, or
     * `null` for "pooling disabled / not built yet". A real loader builds a real
     * pool in its constructor, so the disabled case has to be arranged.
     */
    function setPool(loader: unknown, byteBudgetEvictions: number | null): void {
      (loader as { _gpuBufferPool: unknown })._gpuBufferPool =
        byteBudgetEvictions === null
          ? null
          : {
              getStats: () => ({ byteBudgetEvictions }) as unknown as PoolStats,
              dispose: () => undefined,
            };
    }

    it('is undefined when no loader is registered', () => {
      const manager = SceneLoaderManager.getInstance();
      expect(manager.gpuPoolStats()).toBeUndefined();
    });

    it('is undefined when the default loader has no pool', () => {
      // Pooling disabled. Absent must read as "unknown", never as a
      // zero-eviction record — `debug-state.ts` omits the whole `gpuPool` field
      // in that case.
      const manager = SceneLoaderManager.getInstance();
      setPool(manager.createLoader('poolless'), null);
      expect(manager.gpuPoolStats()).toBeUndefined();
    });

    it('reads the default loader’s real pool when there is one', () => {
      const manager = SceneLoaderManager.getInstance();
      manager.createLoader('pooled');
      // A freshly-built pool has evicted nothing, which is the reading that
      // must NOT refuse a capture.
      expect(manager.gpuPoolStats()?.byteBudgetEvictions).toBe(0);
    });

    it('is scoped to the DEFAULT loader only, unlike refinementResidencyStop', () => {
      // The deliberate asymmetry: `PoolStats` carries a per-type breakdown and
      // a `largestPooledBytes` that do not sum, so a second loader's pool is
      // NOT represented. Assert the scope rather than leaving it to the
      // docstring — a future "helpful" aggregate would silently invent numbers.
      const manager = SceneLoaderManager.getInstance();
      const first = manager.createLoader('pool-default');
      const second = manager.createLoader('pool-other', undefined, false);

      setPool(first, null);
      setPool(second, 12);
      expect(manager.getDefaultLoader()).toBe(first);
      expect(manager.gpuPoolStats()).toBeUndefined();

      setPool(first, 3);
      expect(manager.gpuPoolStats()?.byteBudgetEvictions).toBe(3);
    });
  });

  // Regression: MED-37 — default election after destruction must follow
  // Map insertion order (oldest remaining loader wins). This is the
  // documented contract on `detachLoader` in scene-loader-manager.ts.
  it('should elect the oldest remaining loader as the new default (deterministic)', () => {
    const manager = SceneLoaderManager.getInstance();

    // Insert in known order; only the first is the default.
    manager.createLoader('a', undefined, true);
    const b = manager.createLoader('b', undefined, false);
    manager.createLoader('c', undefined, false);
    manager.createLoader('d', undefined, false);

    // Destroying the default ('a') must hand the crown to 'b' — the
    // oldest *remaining* loader, not 'c' or 'd'.
    manager.destroyLoader('a');
    expect(manager.getDefaultLoader()).toBe(b);

    // Destroying a non-default loader must not change the default.
    manager.destroyLoader('c');
    expect(manager.getDefaultLoader()).toBe(b);

    // Destroying the last remaining loader must null the default.
    manager.destroyLoader('b');
    manager.destroyLoader('d');
    expect(manager.getDefaultLoader()).toBeNull();
  });
});

describe('SceneLoaderManager — zarr-loader API integration', () => {
  beforeEach(() => {
    SceneLoaderManager.disposeInstance();
    DataMonitorManager.disposeInstance();
  });

  afterEach(() => {
    SceneLoaderManager.disposeInstance();
    DataMonitorManager.disposeInstance();
  });

  it('does not create global window variables (post-operation strict check)', () => {
    // architecture.md C1 fix: previously this asserted four hardcoded keys
    // are undefined BEFORE any operation runs — passes on a clean jsdom
    // regardless of what production code does. Now we perform an
    // operation first and assert window stays clean.
    const before = new Set(Object.keys(window));
    const manager = SceneLoaderManager.getInstance();
    manager.createLoader('integration-test');
    const after = Object.keys(window);
    const added = after.filter((k) => !before.has(k));
    expect(added).toEqual([]);
    // Belt-and-braces: the four legacy keys the original audit cared about.
    expect((window as any).__luxarSceneLoader).toBeUndefined();
    expect((window as any).__luxarLoader).toBeUndefined();
    expect((window as any).__luxarDataMonitor).toBeUndefined();
    expect((window as any).globalSceneLoader).toBeUndefined();
  });

  it('should support multiple independent loaders', () => {
    const manager = SceneLoaderManager.getInstance();

    // Create two independent loaders
    const loader1 = manager.createLoader('loader1');
    const loader2 = manager.createLoader('loader2');

    // They should be different instances
    expect(loader1).not.toBe(loader2);

    // Both should be accessible
    expect(manager.getLoader('loader1')).toBe(loader1);
    expect(manager.getLoader('loader2')).toBe(loader2);
  });

  it('should clean up properly with dispose', () => {
    const manager = SceneLoaderManager.getInstance();

    // Create some loaders
    manager.createLoader('test1');
    manager.createLoader('test2');
    expect(manager.getLoaderCount()).toBe(2);

    // Dispose a specific loader
    dispose('test1');
    expect(manager.getLoaderCount()).toBe(1);
    expect(manager.hasLoader('test1')).toBe(false);
    expect(manager.hasLoader('test2')).toBe(true);

    // Dispose all
    dispose();
    expect(manager.getLoaderCount()).toBe(0);
  });
});

describe('SceneLoaderManager — no global state pollution', () => {
  beforeEach(() => {
    SceneLoaderManager.disposeInstance();
    DataMonitorManager.disposeInstance();
  });

  afterEach(() => {
    SceneLoaderManager.disposeInstance();
    DataMonitorManager.disposeInstance();
  });

  it('does not add ANY new keys to window during normal operation (strict)', () => {
    // architecture.md C1/C2 fix: previous version filtered new keys by the
    // substring `luxar`/`loader`, so any pollution lacking those tokens
    // (e.g. `window.__sceneState`, `window.__monitor`, `window.profiler`)
    // would silently pass. Strict variant asserts NO new window keys
    // appear at all.
    const windowKeysBefore = new Set(Object.keys(window));

    const manager = SceneLoaderManager.getInstance();
    manager.createLoader('test');

    const newKeys = Object.keys(window).filter((key) => !windowKeysBefore.has(key));
    // Note: jsdom may add internal accessor properties; if a future jsdom
    // upgrade trips this, allow-list the specific runtime-internal key
    // here (not a substring match).
    expect(newKeys).toEqual([]);
  });
});
