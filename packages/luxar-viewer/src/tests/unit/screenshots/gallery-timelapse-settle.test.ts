import { afterEach, describe, expect, it } from 'vitest';
import {
  isTimelapseSliceSettled,
  resolveTimelapseSettleMs,
} from '../../../tests/screenshots/gallery-timelapse-settle';

describe('gallery timelapse settling', () => {
  afterEach(() => {
    delete (globalThis as any).__luxarDebug;
  });

  it('waits through progressive refinement even after the load-pass signal clears', () => {
    (globalThis as any).__luxarDebug = {
      getState: () => ({ isLoading: false }),
      getSceneLoader: () => ({
        getDefaultLoader: () => ({ isUpdateInProgress: () => true }),
      }),
    };

    expect(isTimelapseSliceSettled()).toBe(false);
  });

  it('settles only when the active loader releases its full update lock', () => {
    (globalThis as any).__luxarDebug = {
      getSceneLoader: () => ({
        getDefaultLoader: () => ({ isUpdateInProgress: () => false }),
      }),
    };

    expect(isTimelapseSliceSettled()).toBe(true);
  });

  it('does not settle when the loader contract is unavailable', () => {
    (globalThis as any).__luxarDebug = { getSceneLoader: () => null };

    expect(isTimelapseSliceSettled()).toBe(false);
  });

  it.each([
    [undefined, 8000],
    ['', 8000],
    ['0', 8000],
    ['-1', 8000],
    ['NaN', 8000],
    ['Infinity', 8000],
    ['1500', 1500],
  ])('resolves settle timeout %j to %d ms', (raw, expected) => {
    expect(resolveTimelapseSettleMs(raw)).toBe(expected);
  });
});
