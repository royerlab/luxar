/**
 * Tests for `loadLeafNode` — the partial-scene-resilience layer.
 *
 * Two contracts here. First, a failing leaf must not take the scene down: the
 * error is logged by kind and `null` returned so siblings still render. Second,
 * this layer does NOT notify the user — that is the end-of-load aggregate's job
 * (`loaders/failure-report.ts`).
 *
 * The per-node toast that used to live here could not work: leaves load
 * sequentially and the toast surface holds one message at a time, so N failures
 * showed a single toast naming the LAST path — the least useful one — while
 * `Network`-kind failures never toasted at all.
 */

import { describe, it, expect, vi, beforeEach, afterEach, type MockInstance } from 'vitest';

const notifierMocks = vi.hoisted(() => ({ toast: vi.fn(), error: vi.fn() }));
vi.mock('../../../../../utils/cross-layer/notifier', () => ({
  notifier: {
    toast: notifierMocks.toast,
    error: notifierMocks.error,
    showHelp: vi.fn(),
    hideHelp: vi.fn(),
    showLoading: vi.fn(),
    hideLoading: vi.fn(),
    clearError: vi.fn(),
  },
}));

import {
  loadLeafNode,
  LoaderError,
  classifyLoaderError,
} from '../../../../../data/scene-loader/nodes/load-leaf-error-dispatch';

describe('loadLeafNode', () => {
  let errorSpy: MockInstance;
  let warnSpy: MockInstance;

  beforeEach(() => {
    errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    notifierMocks.toast.mockClear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const failWith = (kind: 'Network' | 'Decode' | 'Validation' | 'Unexpected', path: string) =>
    loadLeafNode(() => {
      throw new LoaderError(kind, path, new Error('underlying cause'));
    }, path);

  it('logs every failing node but notifies for none of them', async () => {
    // The regression: two failures produced two toasts, of which the user saw
    // one — the second wiped the first well inside its 5s lifetime.
    expect(await failWith('Decode', '/a')).toBeNull();
    expect(await failWith('Decode', '/b')).toBeNull();

    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining('/a'));
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining('/b'));
    expect(notifierMocks.toast).not.toHaveBeenCalled();
  });

  it('logs a Network failure at warning level and does not notify', async () => {
    // Network never toasted even before this change, so the aggregate report is
    // what finally surfaces it.
    expect(await failWith('Network', '/net')).toBeNull();

    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Network error loading /net'));
    expect(notifierMocks.toast).not.toHaveBeenCalled();
  });

  it('returns the loaded value untouched on success', async () => {
    const node = { name: 'ok' } as unknown as import('three').Object3D;
    expect(await loadLeafNode(async () => node, '/ok')).toBe(node);
    expect(errorSpy).not.toHaveBeenCalled();
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('re-throws anything that is not a LoaderError', async () => {
    // Not this layer's call to swallow — the caller decides.
    const boom = new Error('not a loader error');
    await expect(
      loadLeafNode(() => {
        throw boom;
      }, '/x')
    ).rejects.toBe(boom);
  });

  it('logs the cause stack when one is available', async () => {
    await failWith('Unexpected', '/deep');
    expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining('Stack trace for /deep'),
      expect.anything()
    );
  });
});

describe('classifyLoaderError', () => {
  // Persisted into FailedLoaderInfo and consumed by retry policy, so the
  // mapping is now load-bearing rather than a logging detail.
  it.each([
    ['fetch failed', 'Network'],
    ['http 503 bad gateway', 'Network'],
    ['invalid chunk header', 'Decode'],
    ['failed to parse metadata', 'Decode'],
    ['validation failed', 'Validation'],
    ['something else entirely', 'Unexpected'],
  ])('classifies %j as %s', (message, kind) => {
    expect(classifyLoaderError(new Error(message))).toBe(kind);
  });

  it('classifies an AbortError as Network', () => {
    const e = new Error('aborted');
    e.name = 'AbortError';
    expect(classifyLoaderError(e)).toBe('Network');
  });

  it('classifies a non-Error as Unexpected', () => {
    expect(classifyLoaderError('just a string')).toBe('Unexpected');
  });
});
