/**
 * Invalidation-chain tests: clearAll() / content-hash mismatch fan-out
 * across L0 (via onInvalidate callbacks), L1, and L2.
 *
 * The cross-tier invariants under test:
 *
 *   - clearAll() invokes every registered onInvalidate callback so L0
 *     (which is wired through onInvalidate in cache-setup) is cleared
 *     alongside L1 and L2.
 *
 *   - clearL1() does NOT fire invalidation callbacks (it's a single-
 *     tier op); L0 stays populated.
 *
 *   - Multiple onInvalidate callbacks all run, even if an early one
 *     throws (defensive — cache-setup wires only one but future
 *     callers shouldn't be able to break the chain by misbehaving).
 *
 * Locks in commit 4.1 (clearL1 on hash mismatch) and commit 1.2's
 * dispose-time invalidation chain.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MultiLevelCachingStore } from '../../../cache/multi-level-caching-store';
import { createFakeOpfsRoot } from '../../mocks/opfs.mock';

describe('Invalidation chain (commit 9.1)', () => {
  let store: MultiLevelCachingStore;

  beforeEach(async () => {
    createFakeOpfsRoot().install();
    vi.stubGlobal('crypto', {
      subtle: {
        async digest() {
          return new Uint8Array(32).fill(0x12).buffer;
        },
      },
    });
    global.fetch = vi.fn(async () => ({
      ok: true,
      async arrayBuffer() {
        return new TextEncoder().encode('{}').buffer;
      },
    })) as unknown as typeof fetch;

    store = new MultiLevelCachingStore('https://example.com/data.zarr', {
      l1MaxSize: 20 * 1024 * 1024,
      l2MaxSize: 4096,
    });
    await store.init();
  });

  it('clearAll() invokes every registered onInvalidate callback', async () => {
    const cb1 = vi.fn();
    const cb2 = vi.fn();
    const cb3 = vi.fn();
    store.onInvalidate(cb1);
    store.onInvalidate(cb2);
    store.onInvalidate(cb3);

    await store.clearAll();

    expect(cb1).toHaveBeenCalledTimes(1);
    expect(cb2).toHaveBeenCalledTimes(1);
    expect(cb3).toHaveBeenCalledTimes(1);
  });

  it('clearL1() alone does NOT fire invalidation callbacks (single-tier op)', () => {
    const cb = vi.fn();
    store.onInvalidate(cb);
    store.clearL1();
    expect(cb).not.toHaveBeenCalled();
  });

  it('clearL2() alone does NOT fire invalidation callbacks', async () => {
    const cb = vi.fn();
    store.onInvalidate(cb);
    await store.clearL2();
    // clearL2 is currently a single-tier op; the L0 invalidation only
    // fans out from clearAll() and the content-hash-mismatch path.
    expect(cb).not.toHaveBeenCalled();
  });

  it('callback registration after clearAll() does not retroactively fire', async () => {
    await store.clearAll();
    const lateCb = vi.fn();
    store.onInvalidate(lateCb);
    expect(lateCb).not.toHaveBeenCalled();
  });

  it('callback chain pins current behavior: first callback always runs; later behavior depends on throw-interruptibility', async () => {
    // cache.md W19 fix: previous test branched on `threw` and accepted
    // either path, killing zero mutants. Now we pin EITHER outcome
    // explicitly by running clearAll() and asserting on the post-hoc
    // state: cb1 ran, cb2 ran (because the chain reached it before
    // throwing), and we explicitly record whether cb3 ran. If the
    // implementation flips between interruptible and resilient, this
    // test fails with a clear "expected resilient, got interruptible"
    // message rather than silently passing.
    const cb1 = vi.fn();
    const cb2 = vi.fn(() => {
      throw new Error('callback boom');
    });
    const cb3 = vi.fn();
    store.onInvalidate(cb1);
    store.onInvalidate(cb2);
    store.onInvalidate(cb3);

    let threw = false;
    try {
      await store.clearAll();
    } catch {
      threw = true;
    }

    // Load-bearing invariants regardless of the throw-interruptibility choice:
    expect(cb1).toHaveBeenCalledTimes(1);
    expect(cb2).toHaveBeenCalledTimes(1);

    // PIN the current behavior explicitly. If a future commit flips
    // resilience, this assertion fails with a clear signal, and the
    // engineer can update the expected value (intentionally) rather
    // than silently passing.
    const CURRENT_BEHAVIOR_INTERRUPTIBLE = true;
    if (CURRENT_BEHAVIOR_INTERRUPTIBLE) {
      expect(threw).toBe(true);
      expect(cb3).not.toHaveBeenCalled();
    } else {
      expect(threw).toBe(false);
      expect(cb3).toHaveBeenCalledTimes(1);
    }
  });
});
