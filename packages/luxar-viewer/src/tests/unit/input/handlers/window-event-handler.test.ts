/**
 * Unit tests for WindowEventHandler.
 *
 * Verifies the three window/document handlers + the cleanup-array
 * lifecycle. Each test wires a fresh fake SceneManager /
 * AnimationController, attaches the handler to a fresh cleanups
 * array, dispatches the relevant DOM event, and asserts on the
 * fake calls.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { WindowEventHandler } from '../../../../input/handlers/window-event-handler';
import type { SceneManager } from '../../../../scene/scene-manager';
import type { AnimationController } from '../../../../scene/animation-controller';
import type { RenderingControls } from '../../../../ui/rendering-controls';

function makeSceneManager(): {
  sceneManager: SceneManager;
  updateSize: ReturnType<typeof vi.fn>;
  updateFOV: ReturnType<typeof vi.fn>;
  canvas: HTMLCanvasElement;
} {
  const canvas = document.createElement('canvas');
  document.body.appendChild(canvas);
  const updateSize = vi.fn();
  const updateFOV = vi.fn();
  const sceneManager = {
    updateSize,
    updateFOV,
    renderer: { domElement: canvas },
  } as unknown as SceneManager;
  return { sceneManager, updateSize, updateFOV, canvas };
}

function makeAnimationController(): {
  animationController: AnimationController;
  startAnimation: ReturnType<typeof vi.fn>;
} {
  const startAnimation = vi.fn();
  const animationController = { startAnimation } as unknown as AnimationController;
  return { animationController, startAnimation };
}

function setFullscreen(fullscreen: boolean): void {
  Object.defineProperty(document, 'fullscreenElement', {
    configurable: true,
    get: () => (fullscreen ? document.body : null),
  });
}

describe('WindowEventHandler', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
    setFullscreen(false);
  });

  afterEach(() => {
    document.body.innerHTML = '';
  });

  describe('attach + cleanup lifecycle', () => {
    it('appends three cleanup thunks to the provided array', () => {
      const { sceneManager } = makeSceneManager();
      const { animationController } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      const cleanups: (() => void)[] = [];
      handler.attach(cleanups);
      expect(cleanups.length).toBe(3);
      cleanups.forEach((c) => expect(typeof c).toBe('function'));
    });

    it('cleanups remove the listeners (no calls after cleanup)', () => {
      const { sceneManager, updateSize } = makeSceneManager();
      const { animationController, startAnimation } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      const cleanups: (() => void)[] = [];
      handler.attach(cleanups);

      // Run cleanup first.
      cleanups.forEach((c) => c());

      // Now dispatching events should be a no-op.
      window.dispatchEvent(new Event('resize'));
      expect(updateSize).not.toHaveBeenCalled();
      expect(startAnimation).not.toHaveBeenCalled();
    });

    it('registers the wheel listener with passive: false (Phase 14 hygiene)', () => {
      // The Ctrl/Cmd+wheel FOV handler calls preventDefault(); browsers
      // can default wheel listeners on root targets to passive, in which
      // case preventDefault is ignored and the page zooms while the FOV
      // also changes. Verify the registration includes passive: false.
      const { sceneManager } = makeSceneManager();
      const { animationController } = makeAnimationController();
      const addSpy = vi.spyOn(window, 'addEventListener');

      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      const wheelCall = addSpy.mock.calls.find((c) => c[0] === 'wheel');
      expect(wheelCall).toBeDefined();
      // Third arg is the options object.
      expect(wheelCall![2]).toEqual(expect.objectContaining({ passive: false }));

      addSpy.mockRestore();
    });
  });

  describe('window resize', () => {
    it('forwards resize → updateSize + startAnimation', () => {
      const { sceneManager, updateSize } = makeSceneManager();
      const { animationController, startAnimation } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      window.dispatchEvent(new Event('resize'));
      expect(updateSize).toHaveBeenCalledTimes(1);
      expect(startAnimation).toHaveBeenCalledTimes(1);
    });
  });

  describe('wheel', () => {
    it('plain wheel: only kicks the animation loop (no FOV change)', () => {
      const { sceneManager, updateFOV } = makeSceneManager();
      const { animationController, startAnimation } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      window.dispatchEvent(new WheelEvent('wheel', { deltaY: 100 }));
      expect(startAnimation).toHaveBeenCalledTimes(1);
      expect(updateFOV).not.toHaveBeenCalled();
    });

    it('Ctrl+wheel: forwards delta to updateFOV', () => {
      const { sceneManager, updateFOV } = makeSceneManager();
      const { animationController } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      window.dispatchEvent(new WheelEvent('wheel', { deltaY: 75, ctrlKey: true }));
      expect(updateFOV).toHaveBeenCalledTimes(1);
      expect(updateFOV).toHaveBeenCalledWith(75);
    });

    it('Meta+wheel: forwards delta to updateFOV', () => {
      const { sceneManager, updateFOV } = makeSceneManager();
      const { animationController } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      window.dispatchEvent(new WheelEvent('wheel', { deltaY: -33, metaKey: true }));
      expect(updateFOV).toHaveBeenCalledWith(-33);
    });

    it('Ctrl+wheel: switches rendering-controls preset to Custom and syncs', () => {
      const { sceneManager } = makeSceneManager();
      const { animationController } = makeAnimationController();
      const settings = { fovPreset: '60° Standard' };
      const syncCurrentState = vi.fn();
      const renderingControls = {
        settings,
        syncCurrentState,
      } as unknown as RenderingControls;
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.setRenderingControls(renderingControls);
      handler.attach([]);

      window.dispatchEvent(new WheelEvent('wheel', { deltaY: 1, ctrlKey: true }));
      expect(settings.fovPreset).toBe('Custom');
      expect(syncCurrentState).toHaveBeenCalledTimes(1);
    });
  });

  describe('fullscreenchange', () => {
    it('entering fullscreen sets fixed/100vw/100vh inline styles', () => {
      const { sceneManager, canvas } = makeSceneManager();
      const { animationController } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      setFullscreen(true);
      document.dispatchEvent(new Event('fullscreenchange'));

      expect(canvas.style.width).toBe('100vw');
      expect(canvas.style.height).toBe('100vh');
      expect(canvas.style.position).toBe('fixed');
      expect(canvas.style.top).toBe('0px'); // jsdom normalizes "0" → "0px"
      expect(canvas.style.left).toBe('0px');
    });

    it('exiting fullscreen drops the entire style attribute', () => {
      const { sceneManager, canvas } = makeSceneManager();
      const { animationController } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);
      // Pre-set some inline styles to simulate post-fullscreen state.
      canvas.style.width = '800px';
      canvas.style.position = 'fixed';

      setFullscreen(false);
      document.dispatchEvent(new Event('fullscreenchange'));

      expect(canvas.hasAttribute('style')).toBe(false);
    });

    it('schedules updateSize + startAnimation via requestAnimationFrame', async () => {
      const { sceneManager, updateSize } = makeSceneManager();
      const { animationController, startAnimation } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      document.dispatchEvent(new Event('fullscreenchange'));

      // updateSize should NOT be called synchronously — it's deferred.
      expect(updateSize).not.toHaveBeenCalled();

      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

      expect(updateSize).toHaveBeenCalledTimes(1);
      expect(startAnimation).toHaveBeenCalledTimes(1);
    });
  });
});
