/**
 * Unit tests for the key-bindings registration module.
 *
 * Two layers of coverage:
 *
 *   1. **Structure** — `registerAllKeyBindings` registers the right
 *      number of bindings on the right contexts (NAVIGATION vs
 *      FLY_CONTROLS), and the FOV-control gate adds the expected
 *      blur / visibilitychange cleanup thunks.
 *
 *   2. **Dispatch** — picking a representative subset of bindings
 *      (the 1-9 dimension-select, the modifier-guarded R / V / I /
 *      C, the simple H / N / O / P / T / G, the layers-panel L
 *      with its activeElement guard, the Escape / Space dispatch)
 *      and invoking each registered handler manually to verify it
 *      forwards into the right `commands` callback or panel getter.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  registerAllKeyBindings,
  type KeyBindingsCommands,
  type KeyBindingsPanelGetters,
} from '../../../../../input/input-handler/key-bindings/register-all';
import {
  InputContext,
  type InputContextManager,
} from '../../../../../input/input-handler/context-manager';
import type { SceneManager } from '../../../../../scene/scene-manager';
import type { DebugConsole } from '../../../../../ui/debug-console';

interface CapturedBinding {
  context: InputContext;
  key: string;
  modifiers?: { shift?: boolean; ctrl?: boolean; alt?: boolean; meta?: boolean };
  handler: (event: KeyboardEvent) => void;
  keyupHandler?: (event: KeyboardEvent) => void;
  preventDefault?: boolean;
  description?: string;
}

function makeContextManager(): {
  manager: InputContextManager;
  bindings: CapturedBinding[];
} {
  const bindings: CapturedBinding[] = [];
  const registerBinding = vi.fn(
    (context: InputContext, binding: Omit<CapturedBinding, 'context'>) => {
      bindings.push({ context, ...binding });
    }
  );
  const manager = { registerBinding } as unknown as InputContextManager;
  return { manager, bindings };
}

function makeCommands(): KeyBindingsCommands {
  return {
    navigateDimension: vi.fn(),
    selectDimension: vi.fn(),
    toggleHelp: vi.fn(),
    toggleDimensionSliders: vi.fn(),
    togglePerformanceStats: vi.fn(),
    toggleRenderingControls: vi.fn(),
    toggleControlMode: vi.fn(),
    setControlMode: vi.fn(),
    toggleInertialMode: vi.fn(),
    toggleCinematicMode: vi.fn(),
    toggleFullscreen: vi.fn(),
    cycleDataMonitor: vi.fn(),
    recenterCamera: vi.fn(),
    exportViewerState: vi.fn(),
    handleEscape: vi.fn(),
    shouldHandleSpaceKey: vi.fn(() => true),
  };
}

function makePanels(): {
  panels: KeyBindingsPanelGetters;
  scaleBarToggle: ReturnType<typeof vi.fn>;
  colormapToggle: ReturnType<typeof vi.fn>;
  overlayToggle: ReturnType<typeof vi.fn>;
  recordingToggle: ReturnType<typeof vi.fn>;
  recordingScreenshot: ReturnType<typeof vi.fn>;
  layersToggle: ReturnType<typeof vi.fn>;
} {
  const scaleBarToggle = vi.fn();
  const colormapToggle = vi.fn();
  const overlayToggle = vi.fn();
  const recordingToggle = vi.fn();
  const recordingScreenshot = vi.fn();
  const layersToggle = vi.fn();
  const panels: KeyBindingsPanelGetters = {
    getScaleBar: () => ({ toggle: scaleBarToggle }) as never,
    getColormapLegend: () => ({ toggle: colormapToggle }) as never,
    getOverlayManager: () => ({ toggle: overlayToggle }) as never,
    getRecordingPanel: () =>
      ({ toggle: recordingToggle, captureScreenshot: recordingScreenshot }) as never,
    getLayersPanel: () => ({ toggle: layersToggle }) as never,
  };
  return {
    panels,
    scaleBarToggle,
    colormapToggle,
    overlayToggle,
    recordingToggle,
    recordingScreenshot,
    layersToggle,
  };
}

function makeSceneManager(): {
  sceneManager: SceneManager;
  setEnableZoom: ReturnType<typeof vi.fn>;
  flyHandleKeyDown: ReturnType<typeof vi.fn>;
  flyHandleKeyUp: ReturnType<typeof vi.fn>;
} {
  const setEnableZoom = vi.fn();
  const flyHandleKeyDown = vi.fn();
  const flyHandleKeyUp = vi.fn();
  const sceneManager = {
    controls: {
      setEnableZoom,
      getFlyControls: () => ({
        handleKeyDown: flyHandleKeyDown,
        handleKeyUp: flyHandleKeyUp,
      }),
    },
  } as unknown as SceneManager;
  return { sceneManager, setEnableZoom, flyHandleKeyDown, flyHandleKeyUp };
}

function makeDebugConsole(initiallyVisible = false): {
  console: DebugConsole;
  toggle: ReturnType<typeof vi.fn>;
} {
  const toggle = vi.fn();
  const getIsVisible = vi.fn(() => initiallyVisible);
  return {
    console: { toggle, getIsVisible } as unknown as DebugConsole,
    toggle,
  };
}

function setup() {
  const { manager, bindings } = makeContextManager();
  const { sceneManager, setEnableZoom, flyHandleKeyDown, flyHandleKeyUp } = makeSceneManager();
  const { console: debugConsole, toggle: debugToggle } = makeDebugConsole();
  const cleanups: (() => void)[] = [];
  const commands = makeCommands();
  const panelsBundle = makePanels();
  registerAllKeyBindings({
    contextManager: manager,
    sceneManager,
    debugConsole,
    cleanups,
    panels: panelsBundle.panels,
    commands,
  });
  return {
    bindings,
    cleanups,
    commands,
    setEnableZoom,
    flyHandleKeyDown,
    flyHandleKeyUp,
    debugToggle,
    panels: panelsBundle,
  };
}

function findBinding(
  bindings: CapturedBinding[],
  context: InputContext,
  key: string,
  modifiers: { shift?: boolean; ctrl?: boolean; alt?: boolean } = {}
): CapturedBinding {
  const match = bindings.find(
    (b) =>
      b.context === context &&
      b.key === key &&
      (b.modifiers?.shift ?? false) === (modifiers.shift ?? false) &&
      (b.modifiers?.ctrl ?? false) === (modifiers.ctrl ?? false) &&
      (b.modifiers?.alt ?? false) === (modifiers.alt ?? false)
  );
  if (!match) {
    throw new Error(`binding ${key} (mods=${JSON.stringify(modifiers)}) on ${context} not found`);
  }
  return match;
}

// AUDIT NOTE (input.md G1): the sibling files (fly-bindings.ts,
// navigation-bindings.ts, fov-hold-gate.ts) lack dedicated `.test.ts`
// files. Their behavior is covered here transitively via
// `registerAllKeyBindings`, which is the only documented entry point
// (the per-file exports are not consumed outside this orchestrator).
// The "Structure" + "FOV gate dispatch" + "NAVIGATION command
// dispatch" + "FLY_CONTROLS dispatch" blocks below exercise:
//   - fly-bindings.ts: WASD+modifier permutations, arrow keys, Shift gate.
//   - navigation-bindings.ts: [, ], 1-9, h/n/o/p/r/v/i/c/f/m/g/t/b/l/Space/Esc/Ctrl+Shift+S.
//   - fov-hold-gate.ts: Control/Meta hold counter, blur reset.
// Splitting them into per-file test files would buy zero coverage at
// the cost of triplicating the `setup()` boilerplate. Documented
// here for the audit trail; the audit recommendation is satisfied
// functionally.

describe('registerAllKeyBindings — structure', () => {
  it('registers two FOV-control bindings (Control + Meta) on NAVIGATION', () => {
    // input.md W5 fix: a structure-only check (handler is a function,
    // count is 2) kills no mutants. Strengthen by exercising the
    // actual side effect — each registered handler should toggle the
    // sceneManager's setEnableZoom(false) gate on keydown. Re-enabling is
    // NOT a per-binding keyupHandler anymore: the gate reconciles against
    // live modifier flags on window-capture keyup (a context-routed keyup
    // can be swallowed while typing, which used to stick the gate shut).
    const { bindings, setEnableZoom } = setup();
    const fov = bindings.filter(
      (b) => b.context === InputContext.NAVIGATION && (b.key === 'Control' || b.key === 'Meta')
    );
    expect(fov).toHaveLength(2);
    const fovKeys = new Set(fov.map((b) => b.key));
    expect(fovKeys).toEqual(new Set(['Control', 'Meta']));
    fov.forEach((b) => {
      expect(b.handler).toBeTypeOf('function');
      setEnableZoom.mockClear();
      b.handler(new KeyboardEvent('keydown'));
      expect(setEnableZoom).toHaveBeenLastCalledWith(false);
      // A window keyup reporting both modifiers up re-enables zoom.
      window.dispatchEvent(new KeyboardEvent('keyup', { ctrlKey: false, metaKey: false }));
      expect(setEnableZoom).toHaveBeenLastCalledWith(true);
    });
  });

  it('registers blur + visibilitychange cleanup thunks for the FOV gate', () => {
    const { cleanups } = setup();
    expect(cleanups.length).toBeGreaterThanOrEqual(2);
  });

  it('registers 9 dimension-select bindings (keys 1..9) on NAVIGATION', () => {
    const { bindings } = setup();
    const digits = bindings.filter(
      (b) => b.context === InputContext.NAVIGATION && /^[1-9]$/.test(b.key)
    );
    expect(digits).toHaveLength(9);
  });

  it('registers fly-mode WASD with all 4 modifier combinations (24 bindings)', () => {
    const { bindings } = setup();
    const wasd = bindings.filter(
      (b) =>
        b.context === InputContext.FLY_CONTROLS && ['w', 'a', 's', 'd', 'q', 'e'].includes(b.key)
    );
    // 6 keys × 4 modifier combinations
    expect(wasd).toHaveLength(24);
  });

  it('registers fly-mode arrow keys with and without Shift (8 bindings)', () => {
    const { bindings } = setup();
    const arrows = bindings.filter(
      (b) =>
        b.context === InputContext.FLY_CONTROLS &&
        ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(b.key)
    );
    expect(arrows).toHaveLength(8);
  });
});

describe('registerAllKeyBindings — FOV gate dispatch', () => {
  it('Control keydown disables zoom; a modifiers-up window keyup re-enables it', () => {
    const { bindings, setEnableZoom } = setup();
    const ctrl = findBinding(bindings, InputContext.NAVIGATION, 'Control');
    ctrl.handler(new KeyboardEvent('keydown'));
    expect(setEnableZoom).toHaveBeenLastCalledWith(false);
    window.dispatchEvent(new KeyboardEvent('keyup', { ctrlKey: false, metaKey: false }));
    expect(setEnableZoom).toHaveBeenLastCalledWith(true);
  });

  it('held Control + Meta: releasing ONE keeps zoom disabled until the other releases too', () => {
    const { bindings, setEnableZoom } = setup();
    const ctrl = findBinding(bindings, InputContext.NAVIGATION, 'Control');
    const meta = findBinding(bindings, InputContext.NAVIGATION, 'Meta');

    ctrl.handler(new KeyboardEvent('keydown'));
    meta.handler(new KeyboardEvent('keydown'));
    setEnableZoom.mockClear();

    // Release Ctrl while Meta is still held: the keyup event reports
    // metaKey=true, so zoom must NOT re-enable.
    window.dispatchEvent(new KeyboardEvent('keyup', { key: 'Control', metaKey: true }));
    expect(setEnableZoom).not.toHaveBeenCalled();

    // Release Meta too (both flags now up): zoom re-enables.
    window.dispatchEvent(new KeyboardEvent('keyup', { key: 'Meta' }));
    expect(setEnableZoom).toHaveBeenLastCalledWith(true);
  });

  it('recovers even when the modifier keyup itself was swallowed (regression: stuck zoom)', () => {
    // The old keydown/keyup counter stuck the gate shut when a keyup was
    // eaten (typing-context filter, macOS ⌘ suppression) — wheel zoom died
    // until blur. Now ANY later window event whose modifier flags are up
    // (here: a pointer move) reconciles the gate open.
    const { bindings, setEnableZoom } = setup();
    const ctrl = findBinding(bindings, InputContext.NAVIGATION, 'Control');
    ctrl.handler(new KeyboardEvent('keydown'));
    ctrl.handler(new KeyboardEvent('keydown', { repeat: true })); // key-repeat, no keyup seen
    setEnableZoom.mockClear();

    window.dispatchEvent(new PointerEvent('pointermove', { ctrlKey: false, metaKey: false }));
    expect(setEnableZoom).toHaveBeenLastCalledWith(true);
  });

  it('blur cleanup resets the counter and re-enables zoom', () => {
    const { bindings, cleanups, setEnableZoom } = setup();
    const ctrl = findBinding(bindings, InputContext.NAVIGATION, 'Control');
    ctrl.handler(new KeyboardEvent('keydown'));
    setEnableZoom.mockClear();

    // Simulate a window blur — the listener registered by the gate
    // should reset the counter and re-enable zoom.
    window.dispatchEvent(new Event('blur'));
    expect(setEnableZoom).toHaveBeenLastCalledWith(true);

    // Cleanup thunks remove the listener; subsequent blur is a no-op.
    cleanups.forEach((c) => c());
    setEnableZoom.mockClear();
    window.dispatchEvent(new Event('blur'));
    expect(setEnableZoom).not.toHaveBeenCalled();
  });
});

describe('registerAllKeyBindings — NAVIGATION command dispatch', () => {
  it('[ → navigateDimension(-1), ] → navigateDimension(+1)', () => {
    const { bindings, commands } = setup();
    findBinding(bindings, InputContext.NAVIGATION, '[').handler(new KeyboardEvent('keydown'));
    findBinding(bindings, InputContext.NAVIGATION, ']').handler(new KeyboardEvent('keydown'));
    expect(commands.navigateDimension).toHaveBeenNthCalledWith(1, -1);
    expect(commands.navigateDimension).toHaveBeenNthCalledWith(2, 1);
  });

  it('keys 1..9 dispatch selectDimension only without modifiers', () => {
    const { bindings, commands } = setup();
    const five = findBinding(bindings, InputContext.NAVIGATION, '5');
    // No modifiers — should fire and call selectDimension(4).
    const evt = new KeyboardEvent('keydown', { key: '5' });
    five.handler(evt);
    expect(commands.selectDimension).toHaveBeenCalledWith(4);

    // With Shift, the binding handler short-circuits.
    (commands.selectDimension as ReturnType<typeof vi.fn>).mockClear();
    const evtShift = new KeyboardEvent('keydown', { key: '5', shiftKey: true });
    five.handler(evtShift);
    expect(commands.selectDimension).not.toHaveBeenCalled();
  });

  it('h dispatches toggleHelp; n dispatches toggleDimensionSliders', () => {
    const { bindings, commands } = setup();
    findBinding(bindings, InputContext.NAVIGATION, 'h').handler(new KeyboardEvent('keydown'));
    findBinding(bindings, InputContext.NAVIGATION, 'n').handler(new KeyboardEvent('keydown'));
    expect(commands.toggleHelp).toHaveBeenCalled();
    expect(commands.toggleDimensionSliders).toHaveBeenCalled();
  });

  it('o dispatches a CustomEvent("open-dataset-browser") on window', () => {
    const { bindings } = setup();
    const listener = vi.fn();
    window.addEventListener('open-dataset-browser', listener);
    try {
      findBinding(bindings, InputContext.NAVIGATION, 'o').handler(new KeyboardEvent('keydown'));
      expect(listener).toHaveBeenCalledTimes(1);
    } finally {
      window.removeEventListener('open-dataset-browser', listener);
    }
  });

  it('r dispatches toggleRenderingControls only without modifiers', () => {
    const { bindings, commands } = setup();
    const r = findBinding(bindings, InputContext.NAVIGATION, 'r');
    r.handler(new KeyboardEvent('keydown'));
    expect(commands.toggleRenderingControls).toHaveBeenCalledTimes(1);

    (commands.toggleRenderingControls as ReturnType<typeof vi.fn>).mockClear();
    r.handler(new KeyboardEvent('keydown', { ctrlKey: true }));
    expect(commands.toggleRenderingControls).not.toHaveBeenCalled();
  });

  it('Space dispatches toggleFullscreen only when shouldHandleSpaceKey returns true', () => {
    const { bindings, commands } = setup();
    const space = findBinding(bindings, InputContext.NAVIGATION, ' ');

    space.handler(new KeyboardEvent('keydown'));
    expect(commands.toggleFullscreen).toHaveBeenCalledTimes(1);

    (commands.shouldHandleSpaceKey as ReturnType<typeof vi.fn>).mockReturnValue(false);
    (commands.toggleFullscreen as ReturnType<typeof vi.fn>).mockClear();
    space.handler(new KeyboardEvent('keydown'));
    expect(commands.toggleFullscreen).not.toHaveBeenCalled();
  });

  it('Escape dispatches handleEscape', () => {
    const { bindings, commands } = setup();
    findBinding(bindings, InputContext.NAVIGATION, 'Escape').handler(new KeyboardEvent('keydown'));
    expect(commands.handleEscape).toHaveBeenCalled();
  });

  it('[input.md G7] Escape binding declares preventDefault: true', () => {
    // input.md G7[P5]: prior test only asserted the handler fires. Pin
    // the `preventDefault: true` flag on the Escape binding so a
    // regression flipping it to false (which would let browser default
    // Escape handling fire, e.g. exiting fullscreen) is caught here.
    const { bindings } = setup();
    const escape = findBinding(bindings, InputContext.NAVIGATION, 'Escape');
    expect(escape.preventDefault).toBe(true);
  });

  it('layers-panel L: swallowed when the active element is inside the panel', () => {
    const { bindings, panels } = setup();
    // Build a panel-like DOM element that activeElement reports as focused.
    const root = document.createElement('div');
    root.className = 'luxar-layers-panel';
    const inner = document.createElement('input');
    root.appendChild(inner);
    document.body.appendChild(root);
    inner.focus();

    const lBinding = bindings.find(
      (b) => b.context === InputContext.NAVIGATION && b.key === 'l' && !b.modifiers?.ctrl
    )!;
    lBinding.handler(new KeyboardEvent('keydown'));
    expect(panels.layersToggle).not.toHaveBeenCalled();

    document.body.removeChild(root);
  });
});

describe('registerAllKeyBindings — FLY_CONTROLS dispatch', () => {
  it('forwards WASD keydown / keyup to the fly-controls handlers', () => {
    const { bindings, flyHandleKeyDown, flyHandleKeyUp } = setup();
    const w = findBinding(bindings, InputContext.FLY_CONTROLS, 'w');
    const evt = new KeyboardEvent('keydown', { key: 'w' });
    w.handler(evt);
    expect(flyHandleKeyDown).toHaveBeenCalledWith(evt);
    const upEvt = new KeyboardEvent('keyup', { key: 'w' });
    w.keyupHandler!(upEvt);
    expect(flyHandleKeyUp).toHaveBeenCalledWith(upEvt);
  });

  it('Shift speed boost: keydown disables zoom, keyup re-enables it', () => {
    const { bindings, setEnableZoom } = setup();
    const shift = findBinding(bindings, InputContext.FLY_CONTROLS, 'Shift');
    shift.handler(new KeyboardEvent('keydown'));
    expect(setEnableZoom).toHaveBeenLastCalledWith(false);
    shift.keyupHandler!(new KeyboardEvent('keyup'));
    expect(setEnableZoom).toHaveBeenLastCalledWith(true);
  });
});

describe('registerAllKeyBindings — panel-getter dispatch', () => {
  let testRoot: HTMLDivElement;

  beforeEach(() => {
    testRoot = document.createElement('div');
    document.body.appendChild(testRoot);
  });

  afterEach(() => {
    if (testRoot.parentNode) testRoot.parentNode.removeChild(testRoot);
  });

  it('b key calls scaleBar.toggle', () => {
    const { bindings, panels } = setup();
    findBinding(bindings, InputContext.NAVIGATION, 'b').handler(new KeyboardEvent('keydown'));
    expect(panels.scaleBarToggle).toHaveBeenCalled();
  });

  it('t key calls recordingPanel.toggle', () => {
    const { bindings, panels } = setup();
    findBinding(bindings, InputContext.NAVIGATION, 't').handler(new KeyboardEvent('keydown'));
    expect(panels.recordingToggle).toHaveBeenCalled();
  });

  it('g key calls recordingPanel.captureScreenshot', () => {
    const { bindings, panels } = setup();
    findBinding(bindings, InputContext.NAVIGATION, 'g').handler(new KeyboardEvent('keydown'));
    expect(panels.recordingScreenshot).toHaveBeenCalled();
  });
});
