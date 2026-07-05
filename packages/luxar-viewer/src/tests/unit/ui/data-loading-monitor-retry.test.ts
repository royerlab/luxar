/**
 * Unit tests for `DataLoadingMonitor.retryFailedLoads` — the Retry-button
 * orchestration behind the Overview tab's failed-loads banner: empty
 * early-return, double-click guard, deferred-vs-failed toasts, in-flight
 * reset, and the structureDirty marking that makes the banner actually
 * refresh on the incremental overview path.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { DataLoadingMonitor } from '../../../ui/data-loading-monitor';
import { notifier } from '../../../utils/cross-layer/notifier';
import type { FailedLoadsProviderPort } from '../../../data/scene-loader-monitor-port';

type MonitorInternals = {
  retryFailedLoads(): void;
  retryFailedLoadsInFlight: boolean;
  structureDirty: boolean;
  updateUI(): void;
};

beforeEach(() => {
  document.body.innerHTML = '<div id="test-container"></div>';
});
afterEach(() => {
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

type TestMonitor = MonitorInternals & {
  setFailedLoadsProvider(provider: FailedLoadsProviderPort | null): void;
  dispose(): void;
};

function makeMonitor(): TestMonitor {
  const container = document.getElementById('test-container')!;
  // Through `unknown`: the internals interface names private members, so a
  // direct intersection with the class type collapses to `never`.
  return new DataLoadingMonitor(container) as unknown as TestMonitor;
}

function makeProvider(
  paths: string[],
  result:
    | { succeeded: string[]; failed: string[]; deferred?: boolean }
    | Promise<{ succeeded: string[]; failed: string[]; deferred?: boolean }>
): FailedLoadsProviderPort & { retryAll: ReturnType<typeof vi.fn> } {
  return {
    getFailedPaths: () => paths,
    retryAll: vi
      .fn()
      .mockImplementation(() => (result instanceof Promise ? result : Promise.resolve(result))),
  };
}

describe('DataLoadingMonitor.retryFailedLoads', () => {
  it('does not call retryAll when nothing has failed', () => {
    const monitor = makeMonitor();
    const provider = makeProvider([], { succeeded: [], failed: [] });
    monitor.setFailedLoadsProvider(provider);

    monitor.retryFailedLoads();
    expect(provider.retryAll).not.toHaveBeenCalled();
    monitor.dispose();
  });

  it('guards against double-clicks while a batch is in flight, then resets', async () => {
    const monitor = makeMonitor();
    let resolveBatch!: (r: { succeeded: string[]; failed: string[] }) => void;
    const provider = makeProvider(
      ['/p'],
      new Promise((res) => {
        resolveBatch = res;
      })
    );
    monitor.setFailedLoadsProvider(provider);

    monitor.retryFailedLoads();
    monitor.retryFailedLoads(); // double-click while in flight
    expect(provider.retryAll).toHaveBeenCalledTimes(1);
    expect(monitor.retryFailedLoadsInFlight).toBe(true);

    resolveBatch({ succeeded: ['/p'], failed: [] });
    await vi.waitFor(() => expect(monitor.retryFailedLoadsInFlight).toBe(false));
    monitor.dispose();
  });

  it('marks the overview structure dirty so the banner/button actually refresh', async () => {
    // Regression: updateUI() only patches values incrementally — the banner
    // is part of the overview HTML, which is rebuilt only when
    // structureDirty. Pre-fix, clicking Retry neither disabled the button
    // nor dropped the banner after recovery.
    const monitor = makeMonitor();
    const provider = makeProvider(['/p'], { succeeded: ['/p'], failed: [] });
    monitor.setFailedLoadsProvider(provider);

    monitor.structureDirty = false;
    monitor.retryFailedLoads();
    expect(monitor.structureDirty).toBe(true); // disable-the-button rebuild

    await vi.waitFor(() => expect(monitor.retryFailedLoadsInFlight).toBe(false));
    expect(monitor.structureDirty).toBe(true); // completion rebuild
    monitor.dispose();
  });

  it('toasts the deferred message (never "still failing") when the batch was deferred', async () => {
    // Regression: a lock-deferred batch returns {succeeded:[], failed:<all>,
    // deferred:true} WITHOUT retrying; pre-fix this was toasted as
    // "0 recovered, N still failing".
    const monitor = makeMonitor();
    const toastSpy = vi.spyOn(notifier, 'toast').mockImplementation(() => undefined);
    const provider = makeProvider(['/a', '/b'], {
      succeeded: [],
      failed: ['/a', '/b'],
      deferred: true,
    });
    monitor.setFailedLoadsProvider(provider);

    monitor.retryFailedLoads();
    await vi.waitFor(() => expect(monitor.retryFailedLoadsInFlight).toBe(false));

    const messages = toastSpy.mock.calls.map((c) => String(c[0]));
    expect(messages.some((m) => m.includes('Retry deferred'))).toBe(true);
    expect(messages.some((m) => m.includes('still failing'))).toBe(false);
    monitor.dispose();
  });

  it('toasts the genuine outcome for a real partial failure', async () => {
    const monitor = makeMonitor();
    const toastSpy = vi.spyOn(notifier, 'toast').mockImplementation(() => undefined);
    const provider = makeProvider(['/a', '/b'], { succeeded: ['/a'], failed: ['/b'] });
    monitor.setFailedLoadsProvider(provider);

    monitor.retryFailedLoads();
    await vi.waitFor(() => expect(monitor.retryFailedLoadsInFlight).toBe(false));
    expect(
      toastSpy.mock.calls.some((c) => String(c[0]).includes('1 recovered, 1 still failing'))
    ).toBe(true);
    monitor.dispose();
  });
});
