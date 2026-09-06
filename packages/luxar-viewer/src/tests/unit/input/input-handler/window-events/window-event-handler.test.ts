// @vitest-environment jsdom
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
import * as THREE from 'three';
import { WindowEventHandler } from '../../../../../input/input-handler/window-events/window-event-handler';
import type { SceneManager } from '../../../../../scene/scene-manager';
import type { AnimationController } from '../../../../../scene/animation/animation-controller';
import type { RenderingControls } from '../../../../../ui/rendering-controls';
import type { LuxarCamera } from '../../../../../utils/camera-utils';
import { resetViewerContainer, setViewerContainer } from '../../../../../utils/viewer-container';

function makeSceneManager(camera?: LuxarCamera): {
  sceneManager: SceneManager;
  updateSize: ReturnType<typeof vi.fn>;
  updateFOV: ReturnType<typeof vi.fn>;
  canvas: HTMLCanvasElement;
} {
  const canvas = document.createElement('canvas');
  document.body.appendChild(canvas);
  const updateSize = vi.fn();
  // Returns true like the real perspective-camera path.
  const updateFOV = vi.fn(() => true);
  // The onWheel FOV path is gated on a perspective camera; default to one so
  // the modifier-wheel tests exercise the FOV branch. Ortho tests pass an
  // OrthographicCamera to hit the early return.
  const sceneManager = {
    updateSize,
    updateFOV,
    camera: camera ?? new THREE.PerspectiveCamera(),
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

function dispatchWheel(target: EventTarget, init: WheelEventInit): WheelEvent {
  const event = new WheelEvent('wheel', {
    bubbles: true,
    cancelable: true,
    composed: true,
    ...init,
  });
  target.dispatchEvent(event);
  return event;
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
    resetViewerContainer();
    setFullscreen(false);
  });

  afterEach(() => {
    resetViewerContainer();
    document.body.innerHTML = '';
  });

  describe('attach + cleanup lifecycle', () => {
    it('appends four cleanup thunks to the provided array (resize, wheel, fullscreen×2)', () => {
      const { sceneManager } = makeSceneManager();
      const { animationController } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      const cleanups: (() => void)[] = [];
      handler.attach(cleanups);
      // resize + wheel + standard fullscreenchange + webkitfullscreenchange.
      expect(cleanups.length).toBe(4);
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

    it('registers the wheel listener with passive: false', () => {
      // input.md W7 fix: `addSpy.mock.calls.find(c => c[0] === 'wheel')`
      // returns the FIRST matching call and would silently miss a
      // mutation that registered two 'wheel' listeners (one with
      // passive:false, one passive). Switch to `.filter(...)` to
      // count the matches, assert exactly one wheel registration, and
      // then verify its options.
      const { sceneManager } = makeSceneManager();
      const { animationController } = makeAnimationController();
      const addSpy = vi.spyOn(window, 'addEventListener');

      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      const wheelCalls = addSpy.mock.calls.filter((c) => c[0] === 'wheel');
      expect(wheelCalls).toHaveLength(1); // exactly one wheel listener
      // Third arg is the options object.
      expect(wheelCalls[0][2]).toEqual(expect.objectContaining({ passive: false }));

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
    it('suppresses Ctrl+wheel from viewer UI without changing FOV or starting animation', () => {
      const { sceneManager, updateFOV } = makeSceneManager();
      const { animationController, startAnimation } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);
      const viewer = document.createElement('div');
      const control = document.createElement('input');
      viewer.appendChild(control);
      document.body.appendChild(viewer);
      setViewerContainer(viewer);
      const event = dispatchWheel(control, {
        ctrlKey: true,
        deltaY: 75,
      });

      expect(event.defaultPrevented).toBe(true);
      expect(startAnimation).not.toHaveBeenCalled();
      expect(updateFOV).not.toHaveBeenCalled();
    });

    it('leaves Ctrl+wheel from host-page UI outside the viewer untouched', () => {
      const { sceneManager, updateFOV } = makeSceneManager();
      const { animationController, startAnimation } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);
      const viewer = document.createElement('div');
      const hostControl = document.createElement('input');
      document.body.append(viewer, hostControl);
      setViewerContainer(viewer);

      const event = dispatchWheel(hostControl, { ctrlKey: true, deltaY: 75 });

      expect(event.defaultPrevented).toBe(false);
      expect(startAnimation).not.toHaveBeenCalled();
      expect(updateFOV).not.toHaveBeenCalled();
    });

    it('handles Ctrl+wheel from a canvas outside the configured viewer container', () => {
      const { sceneManager, updateFOV, canvas } = makeSceneManager();
      const { animationController, startAnimation } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);
      const viewer = document.createElement('div');
      document.body.appendChild(viewer);
      setViewerContainer(viewer);

      const event = dispatchWheel(canvas, { ctrlKey: true, deltaY: 75 });

      expect(event.defaultPrevented).toBe(true);
      expect(startAnimation).toHaveBeenCalledTimes(1);
      expect(updateFOV).toHaveBeenCalledWith(75);
    });

    it('handles Ctrl+wheel from the canvas across a shadow boundary', () => {
      const { sceneManager, updateFOV, canvas } = makeSceneManager();
      const { animationController, startAnimation } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);
      const host = document.createElement('div');
      const shadowRoot = host.attachShadow({ mode: 'open' });
      document.body.appendChild(host);
      shadowRoot.appendChild(canvas);

      const event = dispatchWheel(canvas, { ctrlKey: true, deltaY: 75 });

      expect(event.defaultPrevented).toBe(true);
      expect(startAnimation).toHaveBeenCalledTimes(1);
      expect(updateFOV).toHaveBeenCalledWith(75);
    });

    it('plain wheel: only kicks the animation loop (no FOV change)', () => {
      const { sceneManager, updateFOV, canvas } = makeSceneManager();
      const { animationController, startAnimation } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      dispatchWheel(canvas, { deltaY: 100 });
      expect(startAnimation).toHaveBeenCalledTimes(1);
      expect(updateFOV).not.toHaveBeenCalled();
    });

    it('Ctrl+wheel: forwards delta to updateFOV', () => {
      const { sceneManager, updateFOV, canvas } = makeSceneManager();
      const { animationController } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      const event = dispatchWheel(canvas, { deltaY: 75, ctrlKey: true });
      expect(event.defaultPrevented).toBe(true);
      expect(updateFOV).toHaveBeenCalledTimes(1);
      expect(updateFOV).toHaveBeenCalledWith(75);
    });

    it('#2531 Ctrl+wheel: forwards a NORMALIZED delta, not the raw line count', () => {
      // dispatchWheel leaves deltaMode at jsdom's default 0, so every test
      // above builds a pixel-mode event and none of them could see this.
      // Firefox reports 3 LINES per notch where Chromium reports 100 px; the
      // raw 3 made one notch a 0.15-degree FOV step instead of ~2.4 degrees.
      const { sceneManager, updateFOV, canvas } = makeSceneManager();
      const { animationController } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      // 48 is hard-coded rather than imported from `utils/wheel-delta`: the
      // point is that the raw 3 does NOT arrive, and deriving the expectation
      // from PIXELS_PER_LINE would move with it.
      dispatchWheel(canvas, { deltaY: 3, deltaMode: 1, ctrlKey: true });
      expect(updateFOV).toHaveBeenCalledWith(48); // 3 lines × 16 px/line
    });

    it('#2531 Ctrl+wheel: page mode uses the CANVAS height and is capped', () => {
      // The only call-site test that pins the `element` argument and the
      // clamp: dropping the element (or the clamp) at the call site is
      // invisible to every pixel/line-mode test above.
      const { sceneManager, updateFOV, canvas } = makeSceneManager();
      Object.defineProperty(canvas, 'clientHeight', { configurable: true, get: () => 600 });
      const { animationController } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      // A fractional page scales by the canvas's own 600 px — this is what
      // proves the element reached the helper (the 800 px nominal fallback
      // would give 80).
      dispatchWheel(canvas, { deltaY: 0.1, deltaMode: 2, ctrlKey: true });
      expect(updateFOV).toHaveBeenLastCalledWith(60);

      // A whole page (what "scroll one screen at a time" emits) saturates the
      // 200 px cap instead of handing the FOV a 600 px, 30-degree step.
      dispatchWheel(canvas, { deltaY: 1, deltaMode: 2, ctrlKey: true });
      expect(updateFOV).toHaveBeenLastCalledWith(200);
    });

    it('#2531 Ctrl+wheel: a pixel-mode delta is still forwarded verbatim', () => {
      // Bit-identity anchor: Chromium/WebKit FOV steps must not have moved.
      const { sceneManager, updateFOV, canvas } = makeSceneManager();
      const { animationController } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      dispatchWheel(canvas, { deltaY: 100, deltaMode: 0, ctrlKey: true });
      expect(updateFOV).toHaveBeenCalledWith(100);
    });

    it('Meta+wheel: forwards delta to updateFOV', () => {
      const { sceneManager, updateFOV, canvas } = makeSceneManager();
      const { animationController } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      dispatchWheel(canvas, { deltaY: -33, metaKey: true });
      expect(updateFOV).toHaveBeenCalledWith(-33);
    });

    // input.md G11 fix: Shift+wheel is NOT a FOV change — only Ctrl
    // and Meta gate the FOV path. Pin that Shift alone behaves like a
    // plain wheel event (animation kick, no updateFOV) so a mutation
    // that added `event.shiftKey` to the gate is caught.
    it('Shift+wheel: only kicks the animation loop (Shift is not a FOV modifier)', () => {
      const { sceneManager, updateFOV, canvas } = makeSceneManager();
      const { animationController, startAnimation } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      dispatchWheel(canvas, { deltaY: 50, shiftKey: true });
      expect(startAnimation).toHaveBeenCalledTimes(1);
      expect(updateFOV).not.toHaveBeenCalled();
    });

    it('Ctrl+wheel: switches rendering-controls preset to Custom and syncs', () => {
      const { sceneManager, canvas } = makeSceneManager();
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

      dispatchWheel(canvas, { deltaY: 1, ctrlKey: true });
      expect(settings.fovPreset).toBe('Custom');
      expect(syncCurrentState).toHaveBeenCalledTimes(1);
    });

    it('#774 Ctrl+wheel with an ortho camera: FOV path is gated, updateFOV NOT called', () => {
      // updateFOV now PERSISTS the perspective FOV stash even in ortho (for
      // deliberate reset/zarr/panel applies). The interactive wheel must be
      // gated to a perspective camera at the call site — otherwise a
      // pinch/ctrl-wheel zoom (browsers synthesize pinch as ctrlKey wheel)
      // in ortho would drag the stash and flip the preset to "Custom".
      const ortho = new THREE.OrthographicCamera(-10, 10, 5, -5, 0.1, 1000);
      const { sceneManager, updateFOV, canvas } = makeSceneManager(ortho);
      const { animationController } = makeAnimationController();
      const settings = { fovPreset: '60° Standard' };
      const syncCurrentState = vi.fn();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.setRenderingControls({ settings, syncCurrentState } as unknown as RenderingControls);
      handler.attach([]);

      dispatchWheel(canvas, { deltaY: 1, ctrlKey: true });
      // Gate short-circuits BEFORE updateFOV — stash cannot be corrupted.
      expect(updateFOV).not.toHaveBeenCalled();
      expect(settings.fovPreset).toBe('60° Standard');
      expect(syncCurrentState).not.toHaveBeenCalled();
    });

    it('#774 Meta+wheel (trackpad pinch) with an ortho camera: also gated, updateFOV NOT called', () => {
      // Symmetric macOS path: Cmd/Meta+wheel and trackpad pinch must not
      // reach updateFOV in ortho either.
      const ortho = new THREE.OrthographicCamera(-10, 10, 5, -5, 0.1, 1000);
      const { sceneManager, updateFOV, canvas } = makeSceneManager(ortho);
      const { animationController, startAnimation } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      dispatchWheel(canvas, { deltaY: -20, metaKey: true });
      // Animation loop still kicked (unconditional), but no FOV mutation.
      expect(startAnimation).toHaveBeenCalledTimes(1);
      expect(updateFOV).not.toHaveBeenCalled();
    });

    it('#774 perspective camera still runs the FOV path (control for the ortho gate)', () => {
      // The reciprocal of the ortho gate: with a perspective camera the wheel
      // path fires updateFOV and stamps the preset, so the gate is specific
      // to ortho — a mutant that gated ALL cameras would fail this.
      const { sceneManager, updateFOV, canvas } = makeSceneManager(); // perspective default
      const { animationController } = makeAnimationController();
      const settings = { fovPreset: '60° Standard' };
      const syncCurrentState = vi.fn();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.setRenderingControls({ settings, syncCurrentState } as unknown as RenderingControls);
      handler.attach([]);

      dispatchWheel(canvas, { deltaY: 1, ctrlKey: true });
      expect(updateFOV).toHaveBeenCalledTimes(1);
      expect(settings.fovPreset).toBe('Custom');
      expect(syncCurrentState).toHaveBeenCalledTimes(1);
    });

    it('[input.md G15] Ctrl+wheel without setRenderingControls: updateFOV fires but the if-guard prevents preset/sync', () => {
      // input.md G15[P5]: prior tests asserted updateFOV was called, but
      // did NOT prove the `if (this.renderingControls)` guard works. A
      // mutation that removed the guard would crash on `undefined.settings`
      // — but only when renderingControls is unset. Pin the no-crash
      // contract explicitly for the unset path.
      const { sceneManager, updateFOV, canvas } = makeSceneManager();
      const { animationController } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      // INTENTIONALLY skip setRenderingControls() — it remains undefined.
      handler.attach([]);

      expect(() => dispatchWheel(canvas, { deltaY: 75, ctrlKey: true })).not.toThrow();
      expect(updateFOV).toHaveBeenCalledTimes(1);
      expect(updateFOV).toHaveBeenCalledWith(75);
    });

    it('[input.md G15] Meta+wheel without setRenderingControls: same no-crash contract', () => {
      // Symmetric: macOS users on Cmd+wheel must also not crash when
      // renderingControls is unset. Pin both branches of the OR-gate.
      const { sceneManager, updateFOV, canvas } = makeSceneManager();
      const { animationController } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      expect(() => dispatchWheel(canvas, { deltaY: -33, metaKey: true })).not.toThrow();
      expect(updateFOV).toHaveBeenCalledWith(-33);
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

    it('entering webkit fullscreen (Safari <16.4) also fills the viewport', () => {
      // Regression: toggleFullscreen() enters via webkitRequestFullscreen on
      // Safari <16.4, which fires `webkitfullscreenchange` and sets
      // `webkitFullscreenElement` (not the standard ones). The handler must
      // still resize the canvas — else the viewer stays windowed in a black
      // fullscreen page.
      const { sceneManager, canvas } = makeSceneManager();
      const { animationController } = makeAnimationController();
      const handler = new WindowEventHandler(sceneManager, animationController);
      handler.attach([]);

      Object.defineProperty(document, 'webkitFullscreenElement', {
        configurable: true,
        get: () => document.body,
      });
      try {
        document.dispatchEvent(new Event('webkitfullscreenchange'));
        expect(canvas.style.width).toBe('100vw');
        expect(canvas.style.height).toBe('100vh');
        expect(canvas.style.position).toBe('fixed');
      } finally {
        Object.defineProperty(document, 'webkitFullscreenElement', {
          configurable: true,
          get: () => null,
        });
      }
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
