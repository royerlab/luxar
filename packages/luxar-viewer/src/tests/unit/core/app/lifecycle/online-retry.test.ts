/**
 * Unit tests for `installOnlineRetry` — the `window 'online'` trigger for
 * `SceneLoader.retryAllFailedLoaders` (the recovery engine's documented
 * "after connectivity is restored" use case).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';
import {
  installOnlineRetry,
  type RetryCapableLoader,
} from '../../../../../core/app/lifecycle/online-retry';

function makeLoader(
  hasFailures: boolean,
  result: { succeeded: string[]; failed: string[] } = { succeeded: ['/p'], failed: [] }
): RetryCapableLoader & { retryAllFailedLoaders: ReturnType<typeof vi.fn> } {
  return {
    hasFailures: () => hasFailures,
    retryAllFailedLoaders: vi.fn().mockResolvedValue(result),
  };
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

  it('removes the listener when the EventGroup is disposed', async () => {
    const loader = makeLoader(true);
    installOnlineRetry({ events, getLoader: () => loader, toast });
    events.dispose();

    window.dispatchEvent(new Event('online'));
    await Promise.resolve();
    expect(loader.retryAllFailedLoaders).not.toHaveBeenCalled();
  });
});
