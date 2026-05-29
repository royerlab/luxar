/**
 * Unit tests for `connectLoaderToMonitor` — the duck-type guard that wires a
 * loader to the data monitor only when it implements the full LoaderMonitor
 * surface. Regression: the progressive loaders now implement that surface, so
 * a progressive-shaped loader must connect (previously silently skipped).
 */

import { describe, it, expect, vi } from 'vitest';
import { connectLoaderToMonitor } from '../../../../../data/scene-loader/nodes/connect-loader-to-monitor';
import type { DataLoader } from '../../../../../data/data-loader-types';
import type { SceneLoaderMonitorPort } from '../../../../../data/scene-loader-monitor-port';

function makeMonitor() {
  return { connectLoader: vi.fn() } as unknown as SceneLoaderMonitorPort & {
    connectLoader: ReturnType<typeof vi.fn>;
  };
}

/** A loader exposing the full 4-method LoaderMonitor surface (as the
 *  progressive wrappers now do). */
function fullSurfaceLoader() {
  return {
    updateView: vi.fn(),
    loadPoints: vi.fn(),
    dispose: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    getMetrics: vi.fn(),
    getActiveQueries: vi.fn(),
  } as unknown as DataLoader;
}

describe('connectLoaderToMonitor', () => {
  it('connects a loader that implements the full LoaderMonitor surface', () => {
    const monitor = makeMonitor();
    connectLoaderToMonitor('/progressive_points', fullSurfaceLoader(), monitor);
    expect(monitor.connectLoader).toHaveBeenCalledTimes(1);
    expect(monitor.connectLoader).toHaveBeenCalledWith('/progressive_points', expect.anything());
  });

  it('skips a loader missing the monitor methods', () => {
    const monitor = makeMonitor();
    const bare = { updateView: vi.fn(), dispose: vi.fn() } as unknown as DataLoader;
    connectLoaderToMonitor('/bare', bare, monitor);
    expect(monitor.connectLoader).not.toHaveBeenCalled();
  });

  it('is a silent no-op when the monitor is null', () => {
    expect(() =>
      connectLoaderToMonitor('/x', fullSurfaceLoader(), null)
    ).not.toThrow();
  });
});
