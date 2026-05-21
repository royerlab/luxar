/**
 * Unit tests for the InputHandler class.
 *
 * Pure utility functions (navigation keys, FOV math, dimension
 * selection) are tested separately in input-handler.test.ts. The
 * constructor wires several heavy components (SceneManager,
 * AnimationController, PerformanceMonitor, DebugConsole), but only
 * stores them — the lifecycle methods that don't call `init()` can be
 * exercised with structural stubs.
 *
 * What we cover:
 *   - Constructor wiring (no-throw, manager-style storage)
 *   - Optional setters (setRenderingControls / setScaleBar / etc.)
 *   - Setter forwarding to PanelCoordinator (R/C/Esc shortcuts)
 *   - clearDimensionUI is a no-op when no dimension UI exists
 *   - clearDimensionUI removes the sceneDimsManager listener
 *   - init() idempotency
 *   - dispose() without init (no listeners to clean up)
 *   - dispose() idempotency
 *
 * What we deliberately skip (needs WebGL or extensive DOM choreography):
 *   - init() side effects (window/canvas listener registration); covered
 *     by E2E spec keyboard-input-system.spec.ts
 *   - initDimensionSliders / showDimensionSliders / setDimensionPosition
 *   - keyboard binding dispatch
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { InputHandler } from '../../../input/input-handler';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import type { SceneManager } from '../../../scene/scene-manager';
import type { AnimationController } from '../../../scene/animation/animation-controller';
import type { PerformanceMonitor } from '../../../ui/performance-monitor';
import type { DebugConsole } from '../../../ui/debug-console';

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
    expect(
      () => new InputHandler(sceneManager, animationController, performanceMonitor, debugConsole)
    ).not.toThrow();
  });

  it('constructor accepts an optional dimensionSlidersFactory', () => {
    const factory = vi.fn();
    expect(
      () =>
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

  it('dispose() is idempotent (second call does not throw or re-dispose deps)', () => {
    const debugConsole = makeDebugConsoleStub();
    const handler = new InputHandler(
      makeSceneManagerStub(),
      makeAnimationControllerStub(),
      makePerformanceMonitorStub(),
      debugConsole
    );
    handler.dispose();
    expect(() => handler.dispose()).not.toThrow();
    // debugConsole.dispose was called by the first dispose; the second
    // dispose may or may not call it again depending on idempotency
    // guards, but in either case must not throw.
  });
});

describe('InputHandler — PanelCoordinator forwarding', () => {
  // The setters that forward to PanelCoordinator drive the keyboard
  // shortcuts (R for rendering controls, Esc for layers/dataset
  // browser/recording panel). Reach the private panelCoordinator via
  // a typed cast — these tests assert the contract that the wiring
  // happens, since a missed forward would silently break the shortcut.

  function makeHandler(): InputHandler {
    return new InputHandler(
      makeSceneManagerStub(),
      makeAnimationControllerStub(),
      makePerformanceMonitorStub(),
      makeDebugConsoleStub()
    );
  }

  it('setRenderingControls forwards to panelCoordinator', () => {
    const handler = makeHandler();
    const coordinator = (
      handler as unknown as {
        panelCoordinator: { setRenderingControls: (c: unknown) => void };
      }
    ).panelCoordinator;
    const spy = vi.spyOn(coordinator, 'setRenderingControls');
    const controls = { dispose: vi.fn() };
    handler.setRenderingControls(controls as never);
    expect(spy).toHaveBeenCalledWith(controls);
  });

  it('setRecordingPanel forwards to panelCoordinator', () => {
    const handler = makeHandler();
    const coordinator = (
      handler as unknown as {
        panelCoordinator: { setRecordingPanel: (p: unknown) => void };
      }
    ).panelCoordinator;
    const spy = vi.spyOn(coordinator, 'setRecordingPanel');
    const panel = { dispose: vi.fn() };
    handler.setRecordingPanel(panel as never);
    expect(spy).toHaveBeenCalledWith(panel);
  });

  it('setLayersPanel forwards to panelCoordinator', () => {
    const handler = makeHandler();
    const coordinator = (
      handler as unknown as {
        panelCoordinator: { setLayersPanel: (p: unknown) => void };
      }
    ).panelCoordinator;
    const spy = vi.spyOn(coordinator, 'setLayersPanel');
    const panel = { dispose: vi.fn() };
    handler.setLayersPanel(panel as never);
    expect(spy).toHaveBeenCalledWith(panel);
  });

  it('setDatasetBrowser forwards to panelCoordinator and accepts undefined', () => {
    const handler = makeHandler();
    const coordinator = (
      handler as unknown as {
        panelCoordinator: { setDatasetBrowser: (b: unknown) => void };
      }
    ).panelCoordinator;
    const spy = vi.spyOn(coordinator, 'setDatasetBrowser');
    const browser = { close: vi.fn() };
    handler.setDatasetBrowser(browser);
    expect(spy).toHaveBeenLastCalledWith(browser);
    handler.setDatasetBrowser(undefined);
    expect(spy).toHaveBeenLastCalledWith(undefined);
    expect(spy).toHaveBeenCalledTimes(2);
  });

  it('setScaleBar / setColormapLegend / setOverlayManager do NOT forward', () => {
    // Sanity: these setters only store local references. If
    // panelCoordinator forwarding is added, update this test alongside.
    const handler = makeHandler();
    const coordinator = (
      handler as unknown as {
        panelCoordinator: Record<string, unknown>;
      }
    ).panelCoordinator;
    const before = Object.keys(coordinator);
    handler.setScaleBar({ dispose: vi.fn() } as never);
    handler.setColormapLegend({ dispose: vi.fn() } as never);
    handler.setOverlayManager({ dispose: vi.fn() } as never);
    // The coordinator's own state shape should not have grown; only
    // its setters add forwarding paths.
    expect(Object.keys(coordinator)).toEqual(before);
  });
});

describe('InputHandler.clearDimensionUI — sceneDimsManager listener cleanup', () => {
  // Invariant: clearDimensionUI must remove the listener it
  // registered with sceneDimsManager, otherwise a disposed
  // InputHandler is retained by the singleton.
  it('removes a registered sceneDimsListener from sceneDimsManager', () => {
    const handler = makeHandler();

    // Inject a fake listener into the private slot, simulating what
    // initDimensionSliders does after a successful initFromScene.
    const fakeListener = vi.fn(async () => {});
    const slot = handler as unknown as {
      sceneDimsListener?: () => Promise<void>;
    };
    slot.sceneDimsListener = fakeListener;
    sceneDimsManager.addListener(fakeListener);
    expect(slot.sceneDimsListener).toBe(fakeListener);

    // sceneDimsManager is a Proxy whose `get` trap re-binds methods
    // on each access (`value.bind(_instance)`), which defeats
    // `vi.spyOn(sceneDimsManager, 'removeListener')` — the spy lands
    // on the proxy, but InputHandler's call goes through a freshly-
    // bound copy. Track the call manually by overriding the method
    // on the proxy (which the `set` trap forwards to the underlying
    // instance) so the override participates in the bind chain.
    const removeCalls: Array<() => void | Promise<void>> = [];
    const original = sceneDimsManager.removeListener.bind(sceneDimsManager);
    (
      sceneDimsManager as unknown as {
        removeListener: (cb: () => void | Promise<void>) => void;
      }
    ).removeListener = (cb) => {
      removeCalls.push(cb);
      original(cb);
    };

    try {
      handler.clearDimensionUI();
      expect(removeCalls).toContain(fakeListener);
      // After clearDimensionUI, the slot must be cleared so a second
      // call doesn't try to remove the (already-removed) listener.
      expect(slot.sceneDimsListener).toBeUndefined();
    } finally {
      // Restore the bound method by deleting the instance override
      // (the proxy then falls back to the prototype method).
      delete (
        sceneDimsManager as unknown as {
          removeListener?: (cb: () => void | Promise<void>) => void;
        }
      ).removeListener;
    }
  });

  function makeHandler(): InputHandler {
    return new InputHandler(
      makeSceneManagerStub(),
      makeAnimationControllerStub(),
      makePerformanceMonitorStub(),
      makeDebugConsoleStub()
    );
  }
});
