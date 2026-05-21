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

/**
 * Polyfill ImageData for jsdom (not available there by default).
 * Each test file calls this once at module scope.
 */
export function ensureImageDataPolyfill(): void {
  if (typeof globalThis.ImageData !== 'undefined') return;
  (globalThis as any).ImageData = class ImageData {
    data: Uint8ClampedArray;
    width: number;
    height: number;
    constructor(widthOrData: number | Uint8ClampedArray, heightOrWidth: number, height?: number) {
      if (widthOrData instanceof Uint8ClampedArray) {
        this.data = widthOrData;
        this.width = heightOrWidth;
        this.height = height ?? widthOrData.length / (4 * heightOrWidth);
      } else {
        this.width = widthOrData;
        this.height = heightOrWidth;
        this.data = new Uint8ClampedArray(this.width * this.height * 4);
      }
    }
  };
}

/**
 * Mutable hook so individual tests can make `canvas.toBlob` return null
 * (simulates encoding failure). Reset to `null` at the end of the test.
 */
export const canvasToBlobOverride: { current: ((cb: any) => void) | null } = {
  current: null,
};

/**
 * Replace `document.createElement('canvas')` so toBlob and getContext('2d')
 * return jsdom-friendly mocks. Other tag names pass through unchanged.
 * Returns the spy so callers can restore it if needed.
 */
export function installCanvasMock(): () => void {
  const origCreateElement = document.createElement.bind(document);
  const spy = vi
    .spyOn(document, 'createElement')
    .mockImplementation((tag: string, options?: any) => {
      const el = origCreateElement(tag, options);
      if (tag === 'canvas') {
        const canvasEl = el as HTMLCanvasElement;
        const origGetContext = canvasEl.getContext.bind(canvasEl);
        (canvasEl as any).getContext = (type: string, ...args: any[]) => {
          if (type === '2d') {
            return { putImageData: vi.fn(), drawImage: vi.fn() };
          }
          return origGetContext(type, ...args);
        };
        (el as HTMLCanvasElement).toBlob = vi.fn((cb: any) => {
          if (canvasToBlobOverride.current) {
            canvasToBlobOverride.current(cb);
          } else {
            cb(new Blob(['test'], { type: 'image/png' }));
          }
        });
      }
      return el;
    });
  return () => spy.mockRestore();
}

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

export function createMockElement(): any {
  return {
    style: {},
    className: '',
    classList: { add: vi.fn(), remove: vi.fn() },
    closest: vi.fn().mockReturnValue({ classList: { add: vi.fn() }, setAttribute: vi.fn() }),
    appendChild: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    querySelector: vi.fn().mockReturnValue({ readOnly: false, style: {}, cursor: '' }),
  };
}

export function createMockController(): any {
  return {
    name: vi.fn().mockReturnThis(),
    onChange: vi.fn().mockReturnThis(),
    show: vi.fn().mockReturnThis(),
    hide: vi.fn().mockReturnThis(),
    updateDisplay: vi.fn().mockReturnThis(),
    domElement: createMockElement(),
  };
}

export function createMockFolder(): any {
  return {
    add: vi.fn().mockImplementation(() => createMockController()),
    addFolder: vi.fn().mockImplementation(() => createMockFolder()),
    close: vi.fn(),
  };
}

export function createMockGUI(): any {
  return {
    domElement: createMockElement(),
    add: vi.fn().mockImplementation(() => createMockController()),
    addFolder: vi.fn().mockImplementation(() => createMockFolder()),
    show: vi.fn(),
    hide: vi.fn(),
    destroy: vi.fn(),
  };
}
