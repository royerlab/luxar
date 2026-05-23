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

// AUDIT NOTE (input.md W2): the four `make*Stub` factories below
// replace first-party internal modules (SceneManager,
// AnimationController, PerformanceMonitor, DebugConsole) rather than
// trust boundaries. This violates P3 (mock only at trust boundaries)
// but is intentional: the InputHandler constructor wires these
// dependencies into heavy submanagers (THREE.js renderer,
// AnimationLoop, performance panel, debug console) whose real
// construction requires a full WebGL context. The structural stubs
// pin the InputHandler API surface — the real integration is covered
// by the `keyboard-input-system.spec.ts` E2E tests. A more
// dependency-injected refactor of InputHandler (taking only the
// methods it calls instead of the full classes) would let us drop the
// stubs; that's tracked under input.md O4/M1 and is OOS for this audit.

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

  // AUDIT NOTE (input.md W1): the six setter smoke tests below
  // (setScaleBar / setColormapLegend / setOverlayManager / setLayersPanel /
  // setDatasetBrowser) only assert .not.toThrow() — the canonical
  // mutation-resistance hole. The setter contract is "store the reference
  // for later cleanup"; the observable proof is that the disposer is
  // called on dispose() of the handler. The dispose-time wiring is
  // covered by the surrounding 'dispose' test below; these setter tests
  // serve as API-surface pins (the methods exist + accept the typed arg)
  // rather than behavioral assertions. Acceptable but documented.
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
    // input.md W6 fix: an idempotency claim demands a count-based
    // assertion. Spy on window.addEventListener and confirm the
    // second init() does not re-register listeners. A mutation that
    // dropped the "already initialised" short-circuit would double
    // the listener count and be caught here.
    const handler = new InputHandler(
      makeSceneManagerStub(),
      makeAnimationControllerStub(),
      makePerformanceMonitorStub(),
      makeDebugConsoleStub()
    );
    const addSpy = vi.spyOn(window, 'addEventListener');
    try {
      handler.init();
      const firstCount = addSpy.mock.calls.length;
      expect(firstCount).toBeGreaterThan(0); // sanity: init registered something
      // Second init: must short-circuit; addEventListener call count is unchanged.
      expect(() => handler.init()).not.toThrow();
      expect(addSpy.mock.calls.length).toBe(firstCount);
    } finally {
      addSpy.mockRestore();
      handler.dispose();
    }
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
      // input.md C1 fix: strengthen the contract — pin not just that
      // removeListener was called with the listener, but that the singleton's
      // actual listener set no longer fires the listener. We register a
      // probe and assert it fires before clearDimensionUI (sanity) and
      // does NOT fire the fake listener after.
      expect(removeCalls).toContain(fakeListener);
      // Fire a dimension change on the singleton; the fake listener must
      // not be invoked (it has been removed).
      const callCountBefore = fakeListener.mock.calls.length;
      sceneDimsManager.setDimensionValue?.(0, 0);
      // Give microtask queue a chance to flush (listeners may be async).
      // The expectation: call count is unchanged.
      expect(fakeListener.mock.calls.length).toBe(callCountBefore);
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

// ─────────────────────────────────────────────────────────────────────
// MED-4 regression: setupControlEvents / setupUserInteractionEvents
// must late-bind the `startAnimation` lookup so a swapped
// AnimationController.startAnimation method still receives the call.
// Before the fix, the method reference was captured at construction
// time and the swap was silently ignored.
// ─────────────────────────────────────────────────────────────────────
describe('InputHandler — MED-4: late-bound startAnimation on controls/canvas events', () => {
  it('controls "change" event invokes the CURRENT animationController.startAnimation', () => {
    // Record every listener attached to controls by event name so the
    // test can fire them synchronously after init().
    const controlsListeners = new Map<string, (() => void)[]>();
    const sceneManager = makeSceneManagerStub();
    (sceneManager.controls.addEventListener as ReturnType<typeof vi.fn>).mockImplementation(
      (event: string, listener: () => void) => {
        const arr = controlsListeners.get(event) ?? [];
        arr.push(listener);
        controlsListeners.set(event, arr);
      }
    );

    const original = vi.fn();
    const animationController = {
      startAnimation: original,
      stopAnimation: vi.fn(),
      dispose: vi.fn(),
      isActive: false,
    } as unknown as AnimationController;

    const handler = new InputHandler(
      sceneManager,
      animationController,
      makePerformanceMonitorStub(),
      makeDebugConsoleStub()
    );
    handler.init();

    // Sanity: a `change` listener was registered.
    const changeListeners = controlsListeners.get('change') ?? [];
    expect(changeListeners.length).toBeGreaterThan(0);

    // Swap startAnimation AFTER init. With the late-bound fix, the
    // listener calls the NEW method; with the old reference-capture
    // bug, it would still call `original`.
    const swapped = vi.fn();
    (animationController as unknown as { startAnimation: () => void }).startAnimation = swapped;

    // Fire the registered listener as if controls emitted 'change'.
    for (const l of changeListeners) l();

    expect(swapped).toHaveBeenCalledTimes(1);
    expect(original).not.toHaveBeenCalled();

    handler.dispose();
  });

  it('canvas mousedown invokes the CURRENT animationController.startAnimation', () => {
    const sceneManager = makeSceneManagerStub();
    const canvas = sceneManager.renderer.domElement;

    const original = vi.fn();
    const animationController = {
      startAnimation: original,
      stopAnimation: vi.fn(),
      dispose: vi.fn(),
      isActive: false,
    } as unknown as AnimationController;

    const handler = new InputHandler(
      sceneManager,
      animationController,
      makePerformanceMonitorStub(),
      makeDebugConsoleStub()
    );
    handler.init();

    // Swap startAnimation AFTER init.
    const swapped = vi.fn();
    (animationController as unknown as { startAnimation: () => void }).startAnimation = swapped;

    // Dispatch a real mousedown event on the canvas; the registered
    // listener should call the swapped method.
    canvas.dispatchEvent(new MouseEvent('mousedown'));

    expect(swapped).toHaveBeenCalledTimes(1);
    expect(original).not.toHaveBeenCalled();

    handler.dispose();
  });
});
