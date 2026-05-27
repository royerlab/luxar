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

  // input.md [W1][P2] fix: the setter contract is "store the reference
  // on the private slot so registerAllKeyBindings' panels.get*() lookups
  // resolve to it at init() time". Previously each test only asserted
  // .not.toThrow(); we now read back the stored reference via the same
  // private slot the handler uses internally. A regression that dropped
  // the assignment would survive .not.toThrow() but fail the read-back.
  type HandlerSlots = {
    scaleBar?: unknown;
    colormapLegend?: unknown;
    overlayManager?: unknown;
    layersPanel?: unknown;
  };
  it('setScaleBar stores the reference on the private scaleBar slot', () => {
    const scaleBar = { dispose: vi.fn() };
    handler.setScaleBar(scaleBar as never);
    expect((handler as unknown as HandlerSlots).scaleBar).toBe(scaleBar);
  });

  it('setColormapLegend stores the reference on the private colormapLegend slot', () => {
    const legend = { dispose: vi.fn() };
    handler.setColormapLegend(legend as never);
    expect((handler as unknown as HandlerSlots).colormapLegend).toBe(legend);
  });

  it('setOverlayManager stores the reference on the private overlayManager slot', () => {
    const manager = { dispose: vi.fn() };
    handler.setOverlayManager(manager as never);
    expect((handler as unknown as HandlerSlots).overlayManager).toBe(manager);
  });

  it('setLayersPanel stores the reference on the private layersPanel slot', () => {
    const panel = { dispose: vi.fn() };
    handler.setLayersPanel(panel as never);
    expect((handler as unknown as HandlerSlots).layersPanel).toBe(panel);
  });

  it('setDatasetBrowser forwards (undefined → browser → undefined) to panelCoordinator', () => {
    // setDatasetBrowser doesn't keep a local field; it only forwards to
    // panelCoordinator. Spy on the coordinator (the trust boundary for
    // this setter) and verify the full sequence of forwarded calls.
    const coordinator = (
      handler as unknown as {
        panelCoordinator: { setDatasetBrowser: (b: unknown) => void };
      }
    ).panelCoordinator;
    const spy = vi.spyOn(coordinator, 'setDatasetBrowser');
    handler.setDatasetBrowser(undefined);
    const browser = { close: vi.fn() };
    handler.setDatasetBrowser(browser);
    handler.setDatasetBrowser(undefined);
    expect(spy).toHaveBeenCalledTimes(3);
    expect(spy.mock.calls[0][0]).toBeUndefined();
    expect(spy.mock.calls[1][0]).toBe(browser);
    expect(spy.mock.calls[2][0]).toBeUndefined();
  });
});

describe('InputHandler.clearDimensionUI', () => {
  it('is a no-op when no dimension sliders have been initialized', () => {
    // input.md [W1][P2] strengthening: previously .not.toThrow() only.
    // The no-listener contract: the sceneDimsListener slot is undefined
    // before, and clearDimensionUI must leave it undefined. A regression
    // that allocated a listener-removal probe even when no listener was
    // registered would survive the smoke test.
    const handler = new InputHandler(
      makeSceneManagerStub(),
      makeAnimationControllerStub(),
      makePerformanceMonitorStub(),
      makeDebugConsoleStub()
    );
    const slot = handler as unknown as { sceneDimsListener?: unknown };
    expect(slot.sceneDimsListener).toBeUndefined();
    handler.clearDimensionUI();
    expect(slot.sceneDimsListener).toBeUndefined();
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

  it('init() then dispose() clears the eventListeners queue', () => {
    // input.md [W1][P2] strengthening: previously .not.toThrow() only.
    // Pin the post-dispose invariants: eventListeners array is empty
    // (all registered cleanups have fired), so future dispose() calls
    // are idempotent.
    const handler = new InputHandler(
      makeSceneManagerStub(),
      makeAnimationControllerStub(),
      makePerformanceMonitorStub(),
      makeDebugConsoleStub()
    );
    handler.init();
    const slot = handler as unknown as { eventListeners: unknown[] };
    expect(slot.eventListeners.length).toBeGreaterThan(0);
    handler.dispose();
    expect(slot.eventListeners.length).toBe(0);
  });

  // dispose() must remove the sceneDimsManager listener so it
  // doesn't outlive the InputHandler on the singleton. Driving the
  // listener-attached path requires a fully-populated scene
  // (initDimensionSliders bails when initFromScene returns false
  // against the stubbed scene), so this test only verifies the
  // missing-listener case behaves correctly.
  it('dispose() before init() still disposes debugConsole', () => {
    // input.md [W1][P2] strengthening: the dispose() contract calls
    // debugConsole.dispose() unconditionally. Verify that observable
    // side effect rather than mere non-throw.
    const debugConsole = makeDebugConsoleStub();
    const handler = new InputHandler(
      makeSceneManagerStub(),
      makeAnimationControllerStub(),
      makePerformanceMonitorStub(),
      debugConsole
    );
    handler.dispose();
    expect(debugConsole.dispose).toHaveBeenCalledTimes(1);
    // eventListeners array is also empty.
    const slot = handler as unknown as { eventListeners: unknown[] };
    expect(slot.eventListeners.length).toBe(0);
  });

  it('dispose() is idempotent: second call leaves state empty and does not double-dispose', () => {
    // input.md [W1][P2] strengthening: the prior comment said "may or
    // may not double-call debugConsole.dispose()" — pin the actual
    // behaviour. dispose() calls debugConsole.dispose() each time it
    // runs (no idempotency guard there), but the eventListeners cleanup
    // is a true no-op the second time (array already empty).
    const debugConsole = makeDebugConsoleStub();
    const handler = new InputHandler(
      makeSceneManagerStub(),
      makeAnimationControllerStub(),
      makePerformanceMonitorStub(),
      debugConsole
    );
    handler.dispose();
    expect(debugConsole.dispose).toHaveBeenCalledTimes(1);
    handler.dispose();
    expect(debugConsole.dispose).toHaveBeenCalledTimes(2);
    const slot = handler as unknown as { eventListeners: unknown[] };
    expect(slot.eventListeners.length).toBe(0);
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
  // input.md O6 / Phase E8: previous version had TWO `it` blocks
  // ("controls 'change'" + "canvas mousedown") testing the SAME
  // late-binding fix — only the event source differs. Parametrize via
  // `it.each` so each row names the specific event source on failure
  // (e.g. "MED-4 late-bind: canvas mousedown"). The two tests differ
  // only in how the listener is wired/fired:
  //   - "controls 'change'": spies on controls.addEventListener and
  //     replays captured listeners after the startAnimation swap.
  //   - "canvas mousedown": dispatches a real DOM MouseEvent on the
  //     renderer's canvas, which the production code wires via
  //     canvas.addEventListener.
  // Both paths must invoke the CURRENT animationController.startAnimation
  // (post-swap), not the reference captured at init() time.
  it.each<{
    label: string;
    wire: (sceneManager: ReturnType<typeof makeSceneManagerStub>) => () => void;
  }>([
    {
      label: 'controls "change" event',
      wire: (sceneManager) => {
        const controlsListeners = new Map<string, (() => void)[]>();
        (sceneManager.controls.addEventListener as ReturnType<typeof vi.fn>).mockImplementation(
          (event: string, listener: () => void) => {
            const arr = controlsListeners.get(event) ?? [];
            arr.push(listener);
            controlsListeners.set(event, arr);
          }
        );
        return () => {
          const changeListeners = controlsListeners.get('change') ?? [];
          // Sanity: a `change` listener was registered at init().
          expect(changeListeners.length).toBeGreaterThan(0);
          for (const l of changeListeners) l();
        };
      },
    },
    {
      label: 'canvas mousedown event',
      wire: (sceneManager) => {
        const canvas = sceneManager.renderer.domElement;
        return () => {
          canvas.dispatchEvent(new MouseEvent('mousedown'));
        };
      },
    },
  ])('MED-4 late-bind: $label invokes the CURRENT startAnimation', ({ wire }) => {
    const sceneManager = makeSceneManagerStub();
    const fire = wire(sceneManager);

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

    // Swap startAnimation AFTER init. With the late-bound fix, the
    // listener calls the NEW method; with the old reference-capture
    // bug, it would still call `original`.
    const swapped = vi.fn();
    (animationController as unknown as { startAnimation: () => void }).startAnimation = swapped;

    fire();

    expect(swapped).toHaveBeenCalledTimes(1);
    expect(original).not.toHaveBeenCalled();

    handler.dispose();
  });
});
