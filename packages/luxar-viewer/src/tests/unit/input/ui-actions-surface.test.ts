// @vitest-environment jsdom
/**
 * Characterization tests for the UI-action surface `InputHandler` builds in
 * `registerAllKeyBindings()` and exposes through `getUiActions()`.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * That method is a wiring table: ~20 command names each mapped to a thunk
 * delegating to a handler method. Measured 2026-09, 27 of those thunks were
 * among the 44-of-96 uncovered functions in `input-handler.ts` — the whole
 * table is built during `init()` and, until now, nothing invoked any entry.
 *
 * The realistic defect here is not logic, it is MIS-WIRING: a copy-pasted
 * entry pointing at the neighbouring method, or the cached surface drifting
 * from the registered one. A test that simply restated the table name-by-name
 * would be a mirror of the source — it would change whenever the source
 * changed and tell us nothing. So these assertions are properties instead.
 *
 * WHAT IT PINS
 * ------------
 *   1. The commands object handed to the key-binding registrar is
 *      REFERENTIALLY the object `getUiActions()` returns. The source comment
 *      states the intent — "Cache the same surface so on-screen affordances
 *      can trigger identical actions without duplicating logic" — and nothing
 *      enforced it. Rebuilding the literal instead of caching it would let the
 *      keyboard and the on-screen control rail drift apart, which is a bug
 *      nobody would see until the two paths behaved differently.
 *   2. Every command survives being invoked with NO optional panels attached.
 *      That is the real condition early in startup, not a synthetic one.
 *   3. The two commands that reach the rest of the app by CustomEvent
 *      dispatch their exact event names, and the element-menu command
 *      suppresses the browser default.
 *   4. `getUiActions()` before `init()` throws rather than handing back a
 *      half-built surface.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  OPEN_DATASET_BROWSER_EVENT,
  OPEN_ELEMENT_MENU_EVENT,
} from '../../../core/app/interaction/canvas-actions';

import { InputHandler } from '../../../input';
import * as registerAll from '../../../input/input-handler/key-bindings/register-all';
import type { SceneManager } from '../../../scene/scene-manager';
import type { AnimationController } from '../../../scene/animation/animation-controller';
import type { PerformanceMonitor } from '../../../ui/performance-monitor';
import type { DebugConsole } from '../../../ui/debug-console';

// Structural stubs rather than real collaborators, for the reason already
// documented at the top of `input-handler-class.test.ts`: the constructor
// wires these into submanagers whose real construction needs a WebGL context.
// Real integration is covered by `keyboard-input-system.spec.ts`.
function makeSceneManagerStub(): SceneManager {
  return {
    updateSize: vi.fn(),
    updateFOV: vi.fn(),
    renderer: { domElement: document.createElement('canvas') },
    controls: {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      getFlyControls: vi.fn(() => undefined),
      // Present on the real ControlsManager. The equivalent stub in
      // `input-handler-class.test.ts` omits these two because no test there
      // ever reaches a command that calls them — which is exactly the drift
      // this file exists to catch, so they are spelled out here.
      getControlType: vi.fn(() => 'orbit'),
      setControlType: vi.fn(),
    },
    centerCameraOnScene: vi.fn(),
    setControlType: vi.fn(),
    getControlType: vi.fn(() => 'orbit'),
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
    cycleMode: vi.fn(),
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

function makeHandler(): InputHandler {
  return new InputHandler(
    makeSceneManagerStub(),
    makeAnimationControllerStub(),
    makePerformanceMonitorStub(),
    makeDebugConsoleStub()
  );
}

describe('InputHandler UI-action surface', () => {
  let handler: InputHandler | undefined;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    handler?.dispose();
    handler = undefined;
    vi.restoreAllMocks();
  });

  it('refuses to hand out a half-built surface before init()', () => {
    handler = makeHandler();
    expect(() => handler!.getUiActions()).toThrow(/before init/i);
  });

  it('caches the SAME commands object it registered, so keyboard and rail cannot drift', () => {
    const spy = vi.spyOn(registerAll, 'registerAllKeyBindings');
    handler = makeHandler();
    handler.init();

    expect(spy).toHaveBeenCalledTimes(1);
    const registered = spy.mock.calls[0][0];
    const exposed = handler.getUiActions();

    // Referential identity, not deep equality: rebuilding an identical-looking
    // literal for the control rail would satisfy `toEqual` and still be the
    // drift this guards against.
    expect(exposed.commands).toBe(registered.commands);
    expect(exposed.panels).toBe(registered.panels);
  });

  it('exposes every command as a callable', () => {
    handler = makeHandler();
    handler.init();
    const { commands } = handler.getUiActions();

    const names = Object.keys(commands) as Array<keyof typeof commands>;
    // Guards the invocation test below against passing vacuously if the
    // surface ever silently empties.
    expect(names.length).toBeGreaterThan(10);
    for (const name of names) {
      expect(typeof commands[name], `command ${String(name)}`).toBe('function');
    }
  });

  it('every command survives invocation with no optional panels attached', () => {
    // Bindings are registered during init(), but the scale bar, colormap
    // legend, overlay manager, recording panel and layers panel are all
    // attached later. Every entry dereferences one of those or a handler
    // method, so this is the state they genuinely run in.
    //
    // Scope note: this arm is a ROBUSTNESS check, not a behavioural one — it
    // proves no command explodes against absent panels, and deliberately
    // nothing more. Mutation testing confirmed it cannot stand alone (a thunk
    // returning `undefined` passes it), which is why the liveness and
    // CustomEvent assertions above and below carry the behavioural weight.
    handler = makeHandler();
    handler.init();
    const { commands } = handler.getUiActions();

    const failures: string[] = [];
    for (const [name, command] of Object.entries(commands)) {
      try {
        if (name === 'navigateDimension') {
          (command as (d: 'next' | 'previous') => void)('next');
        } else if (name === 'selectDimension') {
          (command as (i: number) => void)(0);
        } else if (name === 'setControlMode') {
          (command as (t: string) => void)('orbit');
        } else if (name === 'openElementMenu') {
          (command as (e: Event) => void)(new KeyboardEvent('keydown'));
        } else {
          (command as () => void)();
        }
      } catch (error) {
        failures.push(`${name}: ${(error as Error).message}`);
      }
    }

    expect(failures).toEqual([]);
  });

  it('exposes the panel getters as LIVE views, not snapshots taken at init()', () => {
    // Stryker found the weakness this replaces: the invocation arm below ran
    // every thunk but asserted nothing about what came back, so mutating
    // `getScaleBar: () => this.scaleBar` to `() => undefined` survived. Panels
    // are attached AFTER init(), so a getter that captured its value at
    // registration time would hand the key bindings `undefined` forever.
    const spy = vi.spyOn(registerAll, 'registerAllKeyBindings');
    handler = makeHandler();
    handler.init();
    const { panels } = spy.mock.calls[0][0];

    const scaleBar = { toggle: vi.fn(), setVisible: vi.fn(), isVisible: () => false };
    const legend = { toggle: vi.fn(), setVisible: vi.fn(), isVisible: () => false };
    const overlays = { toggle: vi.fn(), setVisible: vi.fn(), isVisible: () => false };
    const recording = { toggle: vi.fn(), setVisible: vi.fn(), isVisible: () => false };
    const layers = { toggle: vi.fn(), setVisible: vi.fn(), isVisible: () => false };

    // All five, not a sample: a getter left unasserted is a getter that can be
    // snapshotted without anything noticing.
    expect(panels.getScaleBar()).toBeUndefined();
    expect(panels.getColormapLegend()).toBeUndefined();
    expect(panels.getOverlayManager()).toBeUndefined();
    expect(panels.getRecordingPanel()).toBeUndefined();
    expect(panels.getLayersPanel()).toBeUndefined();

    handler.setScaleBar(scaleBar as never);
    handler.setColormapLegend(legend as never);
    handler.setOverlayManager(overlays as never);
    handler.setRecordingPanel(recording as never);
    handler.setLayersPanel(layers as never);

    // Same getter objects, captured before the setters ran.
    expect(panels.getScaleBar()).toBe(scaleBar);
    expect(panels.getColormapLegend()).toBe(legend);
    expect(panels.getOverlayManager()).toBe(overlays);
    expect(panels.getRecordingPanel()).toBe(recording);
    expect(panels.getLayersPanel()).toBe(layers);
  });

  it('removes the keydown/keyup listeners it added, by the same reference', () => {
    // Mutation testing found this gap: both
    // `() => window.removeEventListener('keydown', onKeyDown)` teardowns could
    // be replaced with `() => undefined` and every test still passed. That is
    // the leak class CLAUDE.md calls out under "Event Listener Memory Leaks",
    // and the subtle version of it is a reference mismatch — re-binding at
    // removal time produces a NEW function, so removeEventListener silently
    // removes nothing. Asserting the identity between the two calls is what
    // catches that; asserting "remove was called" would not.
    const added = new Map<string, EventListenerOrEventListenerObject>();
    const removed = new Map<string, EventListenerOrEventListenerObject>();
    const addSpy = vi.spyOn(window, 'addEventListener').mockImplementation((type, listener) => {
      if (type === 'keydown' || type === 'keyup') added.set(type, listener);
    });
    const removeSpy = vi
      .spyOn(window, 'removeEventListener')
      .mockImplementation((type, listener) => {
        if (type === 'keydown' || type === 'keyup') removed.set(type, listener);
      });

    handler = makeHandler();
    handler.init();
    expect([...added.keys()].sort()).toEqual(['keydown', 'keyup']);
    expect(removed.size).toBe(0);

    handler.dispose();
    handler = undefined;

    expect([...removed.keys()].sort()).toEqual(['keydown', 'keyup']);
    // The identity check — a re-bound handler would be a different function
    // and would leak while looking correctly cleaned up.
    expect(removed.get('keydown')).toBe(added.get('keydown'));
    expect(removed.get('keyup')).toBe(added.get('keyup'));

    addSpy.mockRestore();
    removeSpy.mockRestore();
  });

  it('dispatches the exact CustomEvents the rest of the app listens for', () => {
    handler = makeHandler();
    handler.init();
    const { commands } = handler.getUiActions();

    const seen: string[] = [];
    const listener = (event: Event) => seen.push(event.type);
    window.addEventListener(OPEN_DATASET_BROWSER_EVENT, listener);
    window.addEventListener(OPEN_ELEMENT_MENU_EVENT, listener);

    try {
      commands.toggleDatasetBrowser();

      const menuEvent = new KeyboardEvent('keydown', { cancelable: true });
      const preventDefault = vi.spyOn(menuEvent, 'preventDefault');
      commands.openElementMenu(menuEvent);

      expect(seen).toEqual([OPEN_DATASET_BROWSER_EVENT, OPEN_ELEMENT_MENU_EVENT]);
      // The element menu is bound to a key the browser also acts on, so
      // suppressing the default is part of the contract, not incidental.
      expect(preventDefault).toHaveBeenCalled();
    } finally {
      window.removeEventListener(OPEN_DATASET_BROWSER_EVENT, listener);
      window.removeEventListener(OPEN_ELEMENT_MENU_EVENT, listener);
    }
  });
});
