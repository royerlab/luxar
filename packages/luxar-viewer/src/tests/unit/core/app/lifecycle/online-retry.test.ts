// @vitest-environment jsdom
/**
 * Unit tests for `installOnlineRetry` — the `window 'online'` trigger for
 * `SceneLoader.retryAllFailedLoaders` (the recovery engine's documented
 * "after connectivity is restored" use case).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';
import {
  installOnlineRetry,
  DEFERRED_RETRY_DELAY_MS,
  MAX_DEFERRED_RETRY_ATTEMPTS,
  MAX_ONLINE_RETRY_ROUNDS,
  ONLINE_RETRY_MAX_DELAY_MS,
  ONLINE_RETRY_POLL_MS,
  type OnlineRetryPorts,
  type RetryCapableLoader,
} from '../../../../../core/app/lifecycle/online-retry';

function makeLoader(
  hasFailures: boolean,
  result: { succeeded: string[]; failed: string[] } = { succeeded: ['/p'], failed: [] },
  /**
   * Whether any failure is worth an automatic retry. Defaults to `hasFailures`
   * so existing cases behave as before; pass `false` with `hasFailures: true` to
   * express "failures exist, but all deterministic / past the cap".
   */
  hasAutoRetryableFailures: boolean = hasFailures
): RetryCapableLoader & { retryAllFailedLoaders: ReturnType<typeof vi.fn> } {
  return {
    hasFailures: () => hasFailures,
    hasAutoRetryableFailures: () => hasAutoRetryableFailures,
    retryAllFailedLoaders: vi.fn().mockResolvedValue(result),
  };
}

function installWithFailureSignal(
  ports: Omit<OnlineRetryPorts, 'subscribeAutoRetryableFailure'>
): () => void {
  let notifyFailure = (): void => {
    throw new Error('failure listener was not installed');
  };
  installOnlineRetry({
    ...ports,
    subscribeAutoRetryableFailure: (listener) => {
      notifyFailure = listener;
      return () => {
        notifyFailure = () => {};
      };
    },
  });
  return () => notifyFailure();
}

describe('installOnlineRetry', () => {
  let events: EventGroup;
  let toast: ReturnType<typeof vi.fn<(message: string, durationMs?: number) => void>>;

  beforeEach(() => {
    events = new EventGroup();
    toast = vi.fn<(message: string, durationMs?: number) => void>();
  });
  afterEach(() => {
    events.dispose();
  });

  it('retries all failed loaders when the window comes back online', async () => {
    const loader = makeLoader(true, { succeeded: ['/a', '/b'], failed: [] });
    installOnlineRetry({ events, getLoader: () => loader, toast });

    window.dispatchEvent(new Event('online'));
    await vi.waitFor(() => expect(loader.retryAllFailedLoaders).toHaveBeenCalledTimes(1));

    // Outcome surfaced to the user: a "retrying" toast then a success toast.
    await vi.waitFor(() => expect(toast).toHaveBeenCalledTimes(2));
    expect(String(toast.mock.calls[1][0])).toContain('Recovered 2');
  });

  it('retries a transient failure after backoff while the browser stays online', async () => {
    vi.useFakeTimers();
    try {
      const loader = makeLoader(true);
      const notifyFailure = installWithFailureSignal({ events, getLoader: () => loader, toast });

      notifyFailure();

      await vi.advanceTimersByTimeAsync(ONLINE_RETRY_POLL_MS - 1);
      expect(loader.retryAllFailedLoaders).not.toHaveBeenCalled();

      await vi.advanceTimersByTimeAsync(1);
      expect(loader.retryAllFailedLoaders).toHaveBeenCalledTimes(1);
      expect(loader.retryAllFailedLoaders).toHaveBeenCalledWith({ onlyAutoRetryable: true });
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not keep a polling timer alive before a failure is recorded', async () => {
    vi.useFakeTimers();
    try {
      const loader = makeLoader(false);
      installWithFailureSignal({ events, getLoader: () => loader, toast });

      await vi.advanceTimersByTimeAsync(ONLINE_RETRY_MAX_DELAY_MS * 10);

      expect(loader.retryAllFailedLoaders).not.toHaveBeenCalled();
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it('backs off exponentially and stops after the online retry round cap', async () => {
    vi.useFakeTimers();
    try {
      const loader = makeLoader(true, { succeeded: [], failed: ['/a'] });
      const notifyFailure = installWithFailureSignal({ events, getLoader: () => loader, toast });
      notifyFailure();

      let delay = ONLINE_RETRY_POLL_MS;
      for (let round = 1; round <= MAX_ONLINE_RETRY_ROUNDS; round++) {
        await vi.advanceTimersByTimeAsync(delay - 1);
        expect(loader.retryAllFailedLoaders).toHaveBeenCalledTimes(round - 1);
        await vi.advanceTimersByTimeAsync(1);
        expect(loader.retryAllFailedLoaders).toHaveBeenCalledTimes(round);
        delay = Math.min(delay * 2, ONLINE_RETRY_MAX_DELAY_MS);
      }

      await vi.advanceTimersByTimeAsync(ONLINE_RETRY_MAX_DELAY_MS * 2);
      expect(loader.retryAllFailedLoaders).toHaveBeenCalledTimes(MAX_ONLINE_RETRY_ROUNDS);
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it('reports partial recovery when some loads still fail', async () => {
    const loader = makeLoader(true, { succeeded: ['/a'], failed: ['/b'] });
    installOnlineRetry({ events, getLoader: () => loader, toast });

    window.dispatchEvent(new Event('online'));
    await vi.waitFor(() => expect(toast).toHaveBeenCalledTimes(2));
    expect(String(toast.mock.calls[1][0])).toContain('1 recovered, 1 still failing');
  });

  it('is a silent no-op when nothing has failed (the common case)', async () => {
    const loader = makeLoader(false);
    installOnlineRetry({ events, getLoader: () => loader, toast });

    window.dispatchEvent(new Event('online'));
    await Promise.resolve();

    expect(loader.retryAllFailedLoaders).not.toHaveBeenCalled();
    expect(toast).not.toHaveBeenCalled();
  });

  it('does nothing when every failure is deterministic or past the attempt cap', async () => {
    // Reconnecting cannot fix a decode error or a permanent 404 that already
    // exhausted its attempts. Retrying anyway re-fetched them on every `online`
    // transition forever, and the "Connection restored" toast promised a
    // recovery that could not happen.
    const loader = makeLoader(true, { succeeded: [], failed: ['/bad'] }, false);
    installOnlineRetry({ events, getLoader: () => loader, toast });

    window.dispatchEvent(new Event('online'));
    await Promise.resolve();

    expect(loader.retryAllFailedLoaders).not.toHaveBeenCalled();
    expect(toast).not.toHaveBeenCalled();
  });

  it('retries only the auto-retryable subset when failures are transient', async () => {
    const loader = makeLoader(true);
    installOnlineRetry({ events, getLoader: () => loader, toast });

    window.dispatchEvent(new Event('online'));
    await Promise.resolve();

    expect(loader.retryAllFailedLoaders).toHaveBeenCalledWith({ onlyAutoRetryable: true });
  });

  it('is a silent no-op when no loader is live', async () => {
    installOnlineRetry({ events, getLoader: () => null, toast });
    window.dispatchEvent(new Event('online'));
    await Promise.resolve();
    expect(toast).not.toHaveBeenCalled();
  });

  it('ignores online bursts while a retry batch is in flight', async () => {
    let resolveBatch!: (r: { succeeded: string[]; failed: string[] }) => void;
    const loader: RetryCapableLoader & { retryAllFailedLoaders: ReturnType<typeof vi.fn> } = {
      hasFailures: () => true,
      hasAutoRetryableFailures: () => true,
      retryAllFailedLoaders: vi.fn().mockImplementation(
        () =>
          new Promise((res) => {
            resolveBatch = res;
          })
      ),
    };
    installOnlineRetry({ events, getLoader: () => loader, toast });

    window.dispatchEvent(new Event('online'));
    window.dispatchEvent(new Event('online')); // burst while batch 1 is in flight
    window.dispatchEvent(new Event('online'));
    expect(loader.retryAllFailedLoaders).toHaveBeenCalledTimes(1);

    resolveBatch({ succeeded: ['/a'], failed: [] });
    await vi.waitFor(() => expect(toast).toHaveBeenCalledTimes(2));

    // After the batch settles, a NEW online transition retries again.
    window.dispatchEvent(new Event('online'));
    await vi.waitFor(() => expect(loader.retryAllFailedLoaders).toHaveBeenCalledTimes(2));
  });

  it('re-attempts a DEFERRED batch until the update lock frees (never reported as failing)', async () => {
    // Regression: retryAllFailedLoaders returns {succeeded:[], failed:<all>,
    // deferred:true} WITHOUT retrying when a main update holds the lock —
    // the very situation a mid-load reconnect produces. Pre-fix this was
    // toasted as "0 recovered, N still failing" and never re-attempted
    // (no second 'online' event arrives while the browser stays online).
    vi.useFakeTimers();
    try {
      let call = 0;
      const loader: RetryCapableLoader = {
        hasFailures: () => true,
        hasAutoRetryableFailures: () => true,
        retryAllFailedLoaders: vi.fn().mockImplementation(async () => {
          call += 1;
          if (call <= 2) return { succeeded: [], failed: ['/a', '/b'], deferred: true };
          return { succeeded: ['/a', '/b'], failed: [] }; // lock freed on 3rd attempt
        }),
      };
      installOnlineRetry({ events, getLoader: () => loader, toast });

      window.dispatchEvent(new Event('online'));
      await vi.advanceTimersByTimeAsync(0); // attempt 1 resolves deferred
      await vi.advanceTimersByTimeAsync(DEFERRED_RETRY_DELAY_MS); // attempt 2 (deferred)
      await vi.advanceTimersByTimeAsync(DEFERRED_RETRY_DELAY_MS); // attempt 3 (succeeds)

      expect(loader.retryAllFailedLoaders).toHaveBeenCalledTimes(3);
      // Toasts: the initial "retrying…" plus the genuine outcome — and
      // NEVER a "still failing" report for the deferred attempts.
      const messages = toast.mock.calls.map((c) => String(c[0]));
      expect(messages.some((m) => m.includes('still failing'))).toBe(false);
      expect(messages.some((m) => m.includes('Recovered 2'))).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it('gives up after MAX_DEFERRED_RETRY_ATTEMPTS and leaves failures to the banner', async () => {
    vi.useFakeTimers();
    try {
      const loader: RetryCapableLoader = {
        hasFailures: () => true,
        hasAutoRetryableFailures: () => true,
        retryAllFailedLoaders: vi
          .fn()
          .mockResolvedValue({ succeeded: [], failed: ['/a'], deferred: true }),
      };
      installOnlineRetry({ events, getLoader: () => loader, toast });

      window.dispatchEvent(new Event('online'));
      await vi.advanceTimersByTimeAsync(
        DEFERRED_RETRY_DELAY_MS * (MAX_DEFERRED_RETRY_ATTEMPTS + 2)
      );

      expect(loader.retryAllFailedLoaders).toHaveBeenCalledTimes(MAX_DEFERRED_RETRY_ATTEMPTS);
      // No misleading outcome toast — only the initial "retrying…" one.
      expect(toast).toHaveBeenCalledTimes(1);

      // The guard released: a NEW online transition starts a fresh chain.
      window.dispatchEvent(new Event('online'));
      await vi.advanceTimersByTimeAsync(0);
      expect(loader.retryAllFailedLoaders).toHaveBeenCalledTimes(MAX_DEFERRED_RETRY_ATTEMPTS + 1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not rearm a deferred-exhausted episode from another failure notification', async () => {
    vi.useFakeTimers();
    try {
      let hasRetryableFailure = true;
      const retry = vi.fn().mockResolvedValue({
        succeeded: [],
        failed: ['/a'],
        deferred: true,
      });
      const loader: RetryCapableLoader = {
        hasFailures: () => hasRetryableFailure,
        hasAutoRetryableFailures: () => hasRetryableFailure,
        retryAllFailedLoaders: retry,
      };
      const notifyFailure = installWithFailureSignal({ events, getLoader: () => loader, toast });

      window.dispatchEvent(new Event('online'));
      await vi.advanceTimersByTimeAsync(
        DEFERRED_RETRY_DELAY_MS * (MAX_DEFERRED_RETRY_ATTEMPTS + 1)
      );
      expect(retry).toHaveBeenCalledTimes(MAX_DEFERRED_RETRY_ATTEMPTS);

      notifyFailure();
      await vi.advanceTimersByTimeAsync(ONLINE_RETRY_MAX_DELAY_MS * 2);

      expect(retry).toHaveBeenCalledTimes(MAX_DEFERRED_RETRY_ATTEMPTS);
    } finally {
      vi.useRealTimers();
    }
  });

  it('clears a pending deferred-retry timer when the EventGroup is disposed', async () => {
    vi.useFakeTimers();
    try {
      const loader: RetryCapableLoader = {
        hasFailures: () => true,
        hasAutoRetryableFailures: () => true,
        retryAllFailedLoaders: vi
          .fn()
          .mockResolvedValue({ succeeded: [], failed: ['/a'], deferred: true }),
      };
      installOnlineRetry({ events, getLoader: () => loader, toast });

      window.dispatchEvent(new Event('online'));
      await vi.advanceTimersByTimeAsync(0); // attempt 1 resolves deferred, timer pending
      events.dispose();
      await vi.advanceTimersByTimeAsync(DEFERRED_RETRY_DELAY_MS * 3);
      expect(loader.retryAllFailedLoaders).toHaveBeenCalledTimes(1); // no post-dispose attempts
    } finally {
      vi.useRealTimers();
    }
  });

  it('removes the listener when the EventGroup is disposed', async () => {
    const loader = makeLoader(true);
    installOnlineRetry({ events, getLoader: () => loader, toast });
    events.dispose();

    window.dispatchEvent(new Event('online'));
    await Promise.resolve();
    expect(loader.retryAllFailedLoaders).not.toHaveBeenCalled();
  });
});
