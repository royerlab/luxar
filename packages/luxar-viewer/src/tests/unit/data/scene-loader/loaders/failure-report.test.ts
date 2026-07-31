/**
 * Tests for the end-of-load failure report.
 *
 * The regression these guard: `loadScene` logged "Scene loaded successfully"
 * unconditionally, so a scene whose every node failed produced a green log over
 * an empty viewport. `loadScene` structurally cannot throw on a failed node
 * (`loadLeafNode` swallows every `LoaderError` so the rest of the scene still
 * builds), which is why the outcome has to be reported explicitly.
 *
 * Asserted with the presence-AND-absence style of `tests/unit/config/validation.test.ts`:
 * "does not log success" is the whole point, so it needs its own assertion.
 */

import { describe, it, expect, vi, beforeEach, afterEach, type MockInstance } from 'vitest';

const notifierMocks = vi.hoisted(() => ({
  toast: vi.fn(),
  error: vi.fn(),
}));
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
  reportLoadOutcome,
  warnFailedLoaders,
} from '../../../../../data/scene-loader/loaders/failure-report';

describe('reportLoadOutcome', () => {
  // The log module writes through console, so spy there (validation.test.ts idiom).
  let logSpy: { success: MockInstance; error: MockInstance; warning: MockInstance };

  beforeEach(() => {
    logSpy = {
      success: vi.spyOn(console, 'log').mockImplementation(() => {}),
      error: vi.spyOn(console, 'error').mockImplementation(() => {}),
      warning: vi.spyOn(console, 'warn').mockImplementation(() => {}),
    };
    notifierMocks.toast.mockClear();
    notifierMocks.error.mockClear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('logs success and notifies nobody on a clean load', () => {
    expect(reportLoadOutcome([], ['/a', '/b', '/c'])).toBe('clean');

    expect(logSpy.success).toHaveBeenCalledWith(
      expect.stringContaining('Scene loaded successfully')
    );
    expect(logSpy.error).not.toHaveBeenCalled();
    expect(logSpy.warning).not.toHaveBeenCalled();
    expect(notifierMocks.toast).not.toHaveBeenCalled();
    expect(notifierMocks.error).not.toHaveBeenCalled();
  });

  it('warns without claiming success on a partial failure', () => {
    expect(reportLoadOutcome(['/a'], ['/a', '/b', '/c'])).toBe('partial');

    expect(logSpy.warning).toHaveBeenCalledWith(expect.stringContaining('1 node(s) failed'));
    expect(logSpy.warning).toHaveBeenCalledWith(expect.stringContaining('/a'));
    // The regression: success must NOT be claimed alongside a failure.
    expect(logSpy.success).not.toHaveBeenCalled();
    // Per-node failures already logged and the monitor banner is the standing
    // surface, so a partial load does not also interrupt with a notification.
    expect(notifierMocks.toast).not.toHaveBeenCalled();
  });

  it('escalates to an error plus one toast when every node failed', () => {
    expect(reportLoadOutcome(['/a', '/b', '/c'], ['/a', '/b', '/c'])).toBe('total');

    expect(logSpy.error).toHaveBeenCalledWith(expect.stringContaining('all 3 node(s) failed'));
    expect(logSpy.success).not.toHaveBeenCalled();
    expect(notifierMocks.toast).toHaveBeenCalledTimes(1);
    expect(notifierMocks.toast).toHaveBeenCalledWith(
      expect.stringContaining('Scene failed to load'),
      expect.any(Number)
    );
  });

  it('grades a rendered eager node plus a failed lazy level as partial, not total', () => {
    // A lazy substitutive LOD level can record a failure without registering a
    // loader. The one registered (eager) node rendered fine, so this is a
    // partial outcome — not "all nodes failed".
    expect(reportLoadOutcome(['/lazy'], ['/eager'])).toBe('partial');
    expect(logSpy.error).not.toHaveBeenCalled();
    expect(notifierMocks.toast).not.toHaveBeenCalled();
    expect(logSpy.success).not.toHaveBeenCalled();
  });

  it('still grades every registered node failing as total even with an extra lazy failure', () => {
    expect(reportLoadOutcome(['/a', '/lazy'], ['/a'])).toBe('total');
    expect(notifierMocks.toast).toHaveBeenCalledTimes(1);
    // The count matches the listed paths (the failed set), not the registered
    // subset — "all 1 node(s) failed: /a, /lazy" would read as a bug.
    expect(logSpy.error).toHaveBeenCalledWith(
      expect.stringContaining('all 2 node(s) failed to load: /a, /lazy')
    );
  });

  it('reports partial rather than total when the attempted count is unknown', () => {
    // Guard against an empty registered set being read as "everything failed".
    expect(reportLoadOutcome(['/a'], [])).toBe('partial');
    expect(notifierMocks.toast).not.toHaveBeenCalled();
  });
});

describe('warnFailedLoaders', () => {
  let warnSpy: MockInstance;

  beforeEach(() => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('keeps the extracted wording and names every path', () => {
    // Pins the extraction from SceneLoader.updateView: an existing test asserts
    // on the 'Some data could not be loaded' text.
    warnFailedLoaders(['/a', '/b']);

    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Some data could not be loaded'));
    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('/a, /b'));
    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('2 loader(s) failed'));
  });

  it('says nothing when there are no failures', () => {
    warnFailedLoaders([]);
    expect(warnSpy).not.toHaveBeenCalled();
  });
});
