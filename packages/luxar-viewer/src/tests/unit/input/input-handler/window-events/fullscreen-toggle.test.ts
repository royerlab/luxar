/**
 * Unit tests for the fullscreen-toggle command body.
 *
 * Three branches:
 *
 *   1. Not currently fullscreen → requestFullscreen() on documentElement.
 *   2. requestFullscreen rejects → logged, swallowed, NO canvas fallback
 *      (a canvas-only fullscreen would hide every DOM overlay; the module
 *      header documents why the historical fallback was removed).
 *   3. Currently fullscreen → exitFullscreen().
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { toggleFullscreen } from '../../../../../input/input-handler/window-events/fullscreen-toggle';

function setFullscreen(fullscreen: boolean): void {
  Object.defineProperty(document, 'fullscreenElement', {
    configurable: true,
    get: () => (fullscreen ? document.body : null),
  });
}

describe('toggleFullscreen', () => {
  let requestFsSpy: ReturnType<typeof vi.fn>;
  let exitFsSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    requestFsSpy = vi.fn().mockResolvedValue(undefined);
    exitFsSpy = vi.fn().mockResolvedValue(undefined);
    // Patch document.documentElement.requestFullscreen + document.exitFullscreen.
    Object.defineProperty(document.documentElement, 'requestFullscreen', {
      configurable: true,
      value: requestFsSpy,
    });
    Object.defineProperty(document, 'exitFullscreen', {
      configurable: true,
      value: exitFsSpy,
    });
    setFullscreen(false);
  });

  it('not in fullscreen → calls document.documentElement.requestFullscreen', () => {
    toggleFullscreen();
    expect(requestFsSpy).toHaveBeenCalledTimes(1);
    expect(exitFsSpy).not.toHaveBeenCalled();
  });

  it('in fullscreen → calls document.exitFullscreen (not requestFullscreen)', () => {
    setFullscreen(true);
    toggleFullscreen();
    expect(exitFsSpy).toHaveBeenCalledTimes(1);
    expect(requestFsSpy).not.toHaveBeenCalled();
  });

  it('swallows a requestFullscreen rejection (logs only, no throw, no fallback)', async () => {
    requestFsSpy.mockRejectedValueOnce(new Error('not allowed'));

    expect(() => toggleFullscreen()).not.toThrow();

    // Let the promise chain resolve.
    await new Promise((r) => setTimeout(r, 0));

    // Exactly one attempt — the removed canvas fallback must not resurface
    // as a second requestFullscreen call on any element.
    expect(requestFsSpy).toHaveBeenCalledTimes(1);
    expect(exitFsSpy).not.toHaveBeenCalled();
  });

  it('swallows the exitFullscreen rejection (logs only, no throw)', async () => {
    setFullscreen(true);
    exitFsSpy.mockRejectedValueOnce(new Error('cannot exit'));
    expect(() => toggleFullscreen()).not.toThrow();
    await new Promise((r) => setTimeout(r, 0));
    expect(exitFsSpy).toHaveBeenCalled();
  });
});
