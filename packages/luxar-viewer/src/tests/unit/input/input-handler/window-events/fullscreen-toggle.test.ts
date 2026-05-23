/**
 * Unit tests for the fullscreen-toggle command body.
 *
 * input.md G2 fix: the source module (window-events/fullscreen-toggle.ts)
 * was entirely uncovered. Three branches:
 *
 *   1. Not currently fullscreen → requestFullscreen() on documentElement.
 *   2. requestFullscreen rejects → fall back to canvas.requestFullscreen().
 *   3. Currently fullscreen → exitFullscreen().
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { toggleFullscreen } from '../../../../../input/input-handler/window-events/fullscreen-toggle';
import type { SceneManager } from '../../../../../scene/scene-manager';

function makeSceneManager(canvas?: HTMLCanvasElement): SceneManager {
  return {
    renderer: { domElement: canvas ?? document.createElement('canvas') },
  } as unknown as SceneManager;
}

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
    toggleFullscreen({ sceneManager: makeSceneManager() });
    expect(requestFsSpy).toHaveBeenCalledTimes(1);
    expect(exitFsSpy).not.toHaveBeenCalled();
  });

  it('in fullscreen → calls document.exitFullscreen (not requestFullscreen)', () => {
    setFullscreen(true);
    toggleFullscreen({ sceneManager: makeSceneManager() });
    expect(exitFsSpy).toHaveBeenCalledTimes(1);
    expect(requestFsSpy).not.toHaveBeenCalled();
  });

  it('falls back to canvas.requestFullscreen when documentElement.requestFullscreen rejects', async () => {
    const canvas = document.createElement('canvas');
    const canvasRequestFs = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(canvas, 'requestFullscreen', {
      configurable: true,
      value: canvasRequestFs,
    });
    requestFsSpy.mockRejectedValueOnce(new Error('not allowed'));

    toggleFullscreen({ sceneManager: makeSceneManager(canvas) });

    // Let the promise chain resolve.
    await new Promise((r) => setTimeout(r, 0));

    expect(requestFsSpy).toHaveBeenCalledTimes(1);
    expect(canvasRequestFs).toHaveBeenCalledTimes(1);
  });

  it('swallows the canvas fallback rejection too (logs only, no throw)', async () => {
    const canvas = document.createElement('canvas');
    const canvasRequestFs = vi.fn().mockRejectedValue(new Error('also failed'));
    Object.defineProperty(canvas, 'requestFullscreen', {
      configurable: true,
      value: canvasRequestFs,
    });
    requestFsSpy.mockRejectedValueOnce(new Error('not allowed'));

    expect(() => toggleFullscreen({ sceneManager: makeSceneManager(canvas) })).not.toThrow();

    await new Promise((r) => setTimeout(r, 0));
    expect(canvasRequestFs).toHaveBeenCalled();
  });

  it('swallows the exitFullscreen rejection (logs only, no throw)', async () => {
    setFullscreen(true);
    exitFsSpy.mockRejectedValueOnce(new Error('cannot exit'));
    expect(() => toggleFullscreen({ sceneManager: makeSceneManager() })).not.toThrow();
    await new Promise((r) => setTimeout(r, 0));
    expect(exitFsSpy).toHaveBeenCalled();
  });
});
