/**
 * Unit tests for the InputHandler class.
 *
 * Utilities in input-handler-utils.ts are tested separately in
 * input-handler.test.ts. The constructor wires several heavy
 * components (SceneManager,
 * AnimationController, PerformanceMonitor, DebugConsole), but only
 * stores them — the lifecycle methods that don't call `init()` can be
 * exercised with structural stubs.
 *
 * What we cover:
 *   - Constructor wiring (no-throw, ManagerRegistry-style storage)
 *   - Optional setters (setRenderingControls / setScaleBar / etc.)
 *   - clearDimensionUI is a no-op when no dimension UI exists
 *   - init() idempotency
 *   - dispose() without init (no listeners to clean up)
 *
 * What we deliberately skip (needs WebGL or extensive DOM choreography):
 *   - init() side effects (window/canvas listener registration)
 *   - initDimensionSliders / showDimensionSliders / setDimensionPosition
 *   - keyboard binding dispatch
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { InputHandler } from '../../../input/input-handler';
import type { SceneManager } from '../../../scene/scene-manager';
import type { AnimationController } from '../../../scene/animation-controller';
import type { PerformanceMonitor } from '../../../ui/monitors/performance-monitor';
import type { DebugConsole } from '../../../ui/panels/debug-console';

function makeSceneManagerStub(): SceneManager {
  // The InputHandler constructor passes sceneManager to
  // WindowEventHandler, which only stores it. Listeners that touch
  // sceneManager.updateSize() / .renderer.domElement / .updateFOV()
  // only fire after init() — we don't exercise those.
  return {
    updateSize: vi.fn(),
    updateFOV: vi.fn(),
    renderer: { domElement: document.createElement('canvas') },
    controls: {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    },
    camera: {},
    postProcessing: {},
  } as unknown as SceneManager;
}

function makeAnimationControllerStub(): AnimationController {
  return {
    startAnimation: vi.fn(),
    stopAnimation: vi.fn(),
    dispose: vi.fn(),
    isActive: false,
  } as unknown as AnimationController;
}

function makePerformanceMonitorStub(): PerformanceMonitor {
  return {
    show: vi.fn(),
    hide: vi.fn(),
    toggle: vi.fn(),
    cyclePanels: vi.fn(),
    visible: false,
    dispose: vi.fn(),
  } as unknown as PerformanceMonitor;
}

function makeDebugConsoleStub(): DebugConsole {
  return {
    show: vi.fn(),
    hide: vi.fn(),
    toggle: vi.fn(),
    getIsVisible: vi.fn(() => false),
    dispose: vi.fn(),
  } as unknown as DebugConsole;
}

describe('InputHandler — construction', () => {
  let sceneManager: SceneManager;
  let animationController: AnimationController;
  let performanceMonitor: PerformanceMonitor;
  let debugConsole: DebugConsole;

  beforeEach(() => {
    sceneManager = makeSceneManagerStub();
    animationController = makeAnimationControllerStub();
    performanceMonitor = makePerformanceMonitorStub();
    debugConsole = makeDebugConsoleStub();
  });

  it('constructor accepts the four required deps and does not throw', () => {
    expect(() =>
      new InputHandler(sceneManager, animationController, performanceMonitor, debugConsole)
    ).not.toThrow();
  });

  it('constructor accepts an optional dimensionSlidersFactory', () => {
    const factory = vi.fn();
    expect(() =>
      new InputHandler(
        sceneManager,
        animationController,
        performanceMonitor,
        debugConsole,
        factory
      )
    ).not.toThrow();
  });
});

describe('InputHandler — optional setters', () => {
  let handler: InputHandler;

  beforeEach(() => {
    handler = new InputHandler(
      makeSceneManagerStub(),
      makeAnimationControllerStub(),
      makePerformanceMonitorStub(),
      makeDebugConsoleStub()
    );
  });

  it('setScaleBar accepts the overlay reference without throwing', () => {
    const scaleBar = { dispose: vi.fn() };
    expect(() => handler.setScaleBar(scaleBar as never)).not.toThrow();
  });

  it('setColormapLegend accepts the overlay reference without throwing', () => {
    const legend = { dispose: vi.fn() };
    expect(() => handler.setColormapLegend(legend as never)).not.toThrow();
  });

  it('setOverlayManager accepts the manager reference without throwing', () => {
    const manager = { dispose: vi.fn() };
    expect(() => handler.setOverlayManager(manager as never)).not.toThrow();
  });

  it('setLayersPanel accepts the panel reference without throwing', () => {
    const panel = { dispose: vi.fn() };
    expect(() => handler.setLayersPanel(panel as never)).not.toThrow();
  });

  it('setDatasetBrowser accepts undefined to clear the reference', () => {
    expect(() => handler.setDatasetBrowser(undefined)).not.toThrow();
    const browser = { close: vi.fn() };
    expect(() => handler.setDatasetBrowser(browser)).not.toThrow();
    expect(() => handler.setDatasetBrowser(undefined)).not.toThrow();
  });
});

describe('InputHandler.clearDimensionUI', () => {
  it('is a no-op when no dimension sliders have been initialized', () => {
    const handler = new InputHandler(
      makeSceneManagerStub(),
      makeAnimationControllerStub(),
      makePerformanceMonitorStub(),
      makeDebugConsoleStub()
    );
    expect(() => handler.clearDimensionUI()).not.toThrow();
  });
});

describe('InputHandler.init — idempotency', () => {
  it('init() returns silently on the second invocation', () => {
    const handler = new InputHandler(
      makeSceneManagerStub(),
      makeAnimationControllerStub(),
      makePerformanceMonitorStub(),
      makeDebugConsoleStub()
    );
    // First init: sets up listeners (uses real DOM under jsdom).
    handler.init();
    // Second init: must short-circuit, not double-bind.
    expect(() => handler.init()).not.toThrow();
    handler.dispose();
  });
});

describe('InputHandler.dispose', () => {
  it('disposes cleanly without prior init() (no listeners to clean up)', () => {
    const debugConsole = makeDebugConsoleStub();
    const handler = new InputHandler(
      makeSceneManagerStub(),
      makeAnimationControllerStub(),
      makePerformanceMonitorStub(),
      debugConsole
    );
    expect(() => handler.dispose()).not.toThrow();
    expect(debugConsole.dispose).toHaveBeenCalled();
  });

  it('init() then dispose() does not throw', () => {
    const handler = new InputHandler(
      makeSceneManagerStub(),
      makeAnimationControllerStub(),
      makePerformanceMonitorStub(),
      makeDebugConsoleStub()
    );
    handler.init();
    expect(() => handler.dispose()).not.toThrow();
  });

  // dispose() must remove the sceneDimsManager listener so it
  // doesn't outlive the InputHandler on the singleton. Driving the
  // listener-attached path requires a fully-populated scene
  // (initDimensionSliders bails when initFromScene returns false
  // against the stubbed scene), so this test only verifies the
  // missing-listener case runs cleanly.
  it('dispose() handles the missing-listener case cleanly', () => {
    const handler = new InputHandler(
      makeSceneManagerStub(),
      makeAnimationControllerStub(),
      makePerformanceMonitorStub(),
      makeDebugConsoleStub()
    );
    expect(() => handler.dispose()).not.toThrow();
  });
});
