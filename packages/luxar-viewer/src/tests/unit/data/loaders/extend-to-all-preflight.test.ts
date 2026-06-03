/**
 * Unit tests for the shared extend_to_all preflight loggers.
 *
 * Pure logging — we spy on console.warn / console.log to verify which
 * messages fire under which inputs. The project log utility writes
 * through console under the hood.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { warnExtendToAllNoDimensions, announceExtendToAllOnce } from '../../../../data/loaders';
import { Modules } from '../../../../utils/log';

describe('warnExtendToAllNoDimensions', () => {
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('is silent when extendDims is empty', () => {
    warnExtendToAllNoDimensions({
      extendDims: [],
      hasResolvedDimensions: false,
      nodePath: '/n',
      logModule: Modules.SPATIAL_INDEX_LOADER,
    });
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('is silent when dimensions are resolved', () => {
    warnExtendToAllNoDimensions({
      extendDims: ['time'],
      hasResolvedDimensions: true,
      nodePath: '/n',
      logModule: Modules.LINES_LOADER,
    });
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('warns when extendDims is non-empty and dimensions are unresolved', () => {
    warnExtendToAllNoDimensions({
      extendDims: ['time'],
      hasResolvedDimensions: false,
      nodePath: '/MyNode',
      logModule: Modules.GSPLATS_SPATIAL_INDEX_LOADER,
    });
    expect(warnSpy).toHaveBeenCalledTimes(1);
    const [msg] = warnSpy.mock.calls[0] as string[];
    expect(msg).toContain('extend_to_all=[time]');
    expect(msg).toContain('/MyNode');
    expect(msg).toContain('view state has no resolved scene dimensions');
  });

  it('joins multiple extendDims with comma+space', () => {
    warnExtendToAllNoDimensions({
      extendDims: ['time', 'channel', 'batch'],
      hasResolvedDimensions: false,
      nodePath: '/n',
      logModule: Modules.SPATIAL_INDEX_LOADER,
    });
    const [msg] = warnSpy.mock.calls[0] as string[];
    expect(msg).toContain('[time, channel, batch]');
  });
});

describe('announceExtendToAllOnce', () => {
  let logSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);
  });

  afterEach(() => {
    logSpy.mockRestore();
  });

  it('is silent when extendDims is empty', () => {
    announceExtendToAllOnce({
      extendDims: [],
      nodePath: '/n',
      logModule: Modules.LINES_LOADER,
    });
    expect(logSpy).not.toHaveBeenCalled();
  });

  it('emits the broadcast announcement with the joined dims and node path', () => {
    announceExtendToAllOnce({
      extendDims: ['time', 'channel'],
      nodePath: '/MyNode',
      logModule: Modules.LINES_LOADER,
    });
    expect(logSpy).toHaveBeenCalledTimes(1);
    const [msg] = logSpy.mock.calls[0] as string[];
    expect(msg).toContain('configured with extend_to_all: time, channel');
    expect(msg).toContain('/MyNode');
  });
});
