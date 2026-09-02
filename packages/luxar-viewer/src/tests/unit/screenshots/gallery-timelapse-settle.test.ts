import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  isTimelapseSliceSettled,
  resolveTimelapseSettleMs,
  waitForTimelapseSliceSettled,
} from '../../../tests/screenshots/gallery-timelapse-settle';

describe('gallery timelapse settling', () => {
  afterEach(() => {
    delete (globalThis as any).__luxarDebug;
    vi.restoreAllMocks();
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

  it('warns with the demo and timeout when refinement does not settle', async () => {
    const timeout = new Error('timeout');
    const page = { waitForFunction: vi.fn().mockRejectedValue(timeout) };
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    await waitForTimelapseSliceSettled(page, 'gsplats_4d_h2afva_timelapse', 1500);

    expect(page.waitForFunction).toHaveBeenCalledWith(isTimelapseSliceSettled, undefined, {
      timeout: 1500,
    });
    expect(warn).toHaveBeenCalledWith(
      '[gsplats_4d_h2afva_timelapse] timelapse slice did not settle within 1500 ms: Error: timeout'
    );
  });

  it('settles the timelapse still after its frame-point jump', () => {
    const specPath = path.join(
      path.dirname(fileURLToPath(import.meta.url)),
      '../../screenshots/generate-gallery.spec.ts'
    );
    const source = fs.readFileSync(specPath, 'utf-8');

    expect(source).toMatch(
      /await jumpTimeDimToFrac\(page, frac\);\s+await waitForTimelapseSliceSettled\(page, demo\.id, TL_SETTLE_MS\);/
    );
  });
});
