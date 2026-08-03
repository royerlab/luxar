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
        })
      ),
      getAutoRotate: vi.fn().mockReturnValue(false),
      setAutoRotate: vi.fn(),
    },
    setAdaptivePixelRatio: vi.fn(),
  };
}

/** Build the test double for AnimationController. */
export function createMockAnimationController(): any {
  return {
    startAnimation: vi.fn(),
    stopAnimation: vi.fn(),
    addPerFrameCallback: vi.fn(),
    removePerFrameCallback: vi.fn(),
  };
}

// ── GUI module mock factories ──────────────────────────────────
// Used by each test file's `vi.mock('../../../ui/gui', ...)` call.
