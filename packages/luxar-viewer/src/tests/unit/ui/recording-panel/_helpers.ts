/**
 * Shared mock factories for recording-panel tests.
 *
 * `vi.mock(...)` blocks can NOT live here — vitest's mock-hoisting works
 * per-test-file. Each test file must declare its own `vi.mock` for
 * `'../../../ui/gui'`, `'../../../ui/toast'`, etc. But the factory
 * functions that build mock SceneManager / AnimationController instances
 * (used by every test) are extracted here.
 *
 * Also exports the ImageData polyfill setup so each test file can call
 * it in a top-level call without duplicating the polyfill class.
 */

import { vi } from 'vitest';
import { LuxarOrbitControls } from '../../../../controls/luxar-orbit-controls';

// [ui.md/O3][P10] Removed unused exports `canvasToBlobOverride` and
// `installCanvasMock`: no test file in `recording-panel/` imports them.
// `screenshot-strategy.test.ts` declares its own local `canvasToBlobOverride`
// + inline createElement spy; the helper version was dead code.

/** Build the test double for SceneManager. */
export function createMockSceneManager(): any {
  const mockCanvas = document.createElement('canvas');
  mockCanvas.toBlob = vi.fn((callback: any) => {
    const blob = new Blob(['test'], { type: 'image/png' });
    callback(blob);
  });
  (mockCanvas as any).captureStream = vi.fn(() => new MediaStream());
  mockCanvas.focus = vi.fn();

  return {
    renderer: {
      domElement: mockCanvas,
      getSize: vi.fn().mockReturnValue({ x: 800, y: 600 }),
      setPixelRatio: vi.fn(),
      setSize: vi.fn(),
    },
    resizeLocked: false,
    postProcessing: {
      render: vi.fn(),
      renderToImageData: vi.fn().mockResolvedValue(new ImageData(4, 4)),
      captureHDRAsEXR: vi.fn().mockResolvedValue(new Uint8Array([0x76, 0x2f, 0x31, 0x01])),
      resize: vi.fn(),
      // SSAA off by default: the size handed to `resize` is the size the
      // render targets get. Tests that care override this.
      getEffectiveRenderScale: vi.fn().mockReturnValue(1),
      // The DISPLAY size — what `resize` was last given. It matches
      // `renderer.getSize()` here only because SSAA is off by default;
      // tests that raise the multiplier make the two diverge, since
      // the renderer gets the SSAA-multiplied size.
      getDisplaySize: vi.fn().mockReturnValue({ width: 800, height: 600 }),
    },
    camera: {
      position: { x: 10, y: 5, z: 10, distanceTo: vi.fn().mockReturnValue(15), clone: vi.fn() },
      lookAt: vi.fn(),
    },
    scene: {
      background: { clone: vi.fn() },
    },
    controls: {
      getControls: vi.fn().mockReturnValue(
        Object.assign(Object.create(LuxarOrbitControls.prototype), {
          target: { x: 0, y: 0, z: 0, clone: vi.fn().mockReturnValue({ x: 0, y: 0, z: 0 }) },
          applyOrbitRotation: vi.fn(),
          applyOrbitDolly: vi.fn(),
          // Off by default, matching the shipped default — a turntable test
          // that wants the baked dolly turns it on explicitly.
          autoDolly: false,
          autoDollyAmplitude: 0.15,
          autoDollyPeriod: 10,
        })
      ),
      getAutoRotate: vi.fn().mockReturnValue(false),
      setAutoRotate: vi.fn(),
      getAutoDolly: vi.fn().mockReturnValue(false),
      setAutoDolly: vi.fn(),
    },
    setAdaptivePixelRatio: vi.fn(),
  };
}

/**
 * Build the test double for AnimationController.
 *
 * Models the real controller's stopped-loop semantics: the rAF loop
 * starts STOPPED (the viewer idle-stops after ~2s of no interaction,
 * which is the normal state by the time a user has read the Recording
 * panel and confirmed a dialog), only `startAnimation()` flips it to
 * running, and per-frame callbacks fire only while it runs. Registering
 * a `continuous` callback KEEPS a running loop alive but never restarts
 * a stopped one — so a double that fires callbacks unconditionally is
 * structurally blind to the whole "capture emits N identical frames"
 * class of bug (a deleted `startAnimation()` would fail nothing).
 *
 * The options argument of each registration is recorded by `vi.fn()`
 * itself, so `{ continuous: true }` is assertable with
 * `toHaveBeenCalledWith`.
 *
 * Note the fidelity limit: this double runs a callback at REGISTRATION
 * time (once, if the loop is running), not on a simulated frame — its
 * consumers have no frame pump. `offline-capture-strategy.test.ts` has a
 * local double that instead QUEUES callbacks and runs them from a
 * `__tick()` driven by its `requestAnimationFrame` spy, which is what
 * catches "registered and removed with no tick in between". A consumer
 * that needs frame-tick fidelity should adopt that shape.
 */
export function createMockAnimationController({
  animating = false,
}: { animating?: boolean } = {}): any {
  let isAnimating = animating;
  const registered = new Set<string>();
  return {
    startAnimation: vi.fn(() => {
      isAnimating = true;
    }),
    stopAnimation: vi.fn(() => {
      isAnimating = false;
    }),
    // Invoke the callback once synchronously to simulate a single
    // rendered frame — but only while the loop is actually animating.
    addPerFrameCallback: vi.fn((id: string, cb?: () => void, _options?: unknown) => {
      registered.add(id);
      if (isAnimating) cb?.();
    }),
    // Returns a boolean, like the real `removePerFrameCallback`.
    removePerFrameCallback: vi.fn((id: string) => registered.delete(id)),
  };
}

// ── GUI module mock factories ──────────────────────────────────
// Used by each test file's `vi.mock('../../../ui/gui', ...)` call.
