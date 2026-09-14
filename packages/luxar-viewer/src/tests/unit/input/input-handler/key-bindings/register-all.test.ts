// @vitest-environment jsdom
/**
 * Unit tests for the key-bindings registration module.
 *
 * Two layers of coverage:
 *
 *   1. **Structure** — `registerAllKeyBindings` registers the right
 *      number of bindings on the right contexts (NAVIGATION vs
 *      FLY_CONTROLS).
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
  InputContextManager,
} from '../../../../../input/input-handler/context-manager';
import type { SceneManager } from '../../../../../scene/scene-manager';
import type { DebugConsole } from '../../../../../ui/debug-console';
import type { ShortcutHelpMetadata } from '../../../../../types/shortcut-help';
import { KeyAction } from '../../../../../input/input-handler/key-bindings/actions';

interface CapturedBinding {
  context: InputContext;
  actionId: string;
  actionParameter?: string | number;
  key: string;
  modifiers?: { shift?: boolean; ctrl?: boolean; alt?: boolean; meta?: boolean };
  handler: (event: KeyboardEvent) => void;
  keyupHandler?: (event: KeyboardEvent) => void;
  preventDefault?: boolean;
  description: string;
  help: ShortcutHelpMetadata | false;
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
    toggleDatasetBrowser: vi.fn(),
    openElementMenu: vi.fn((event: KeyboardEvent) => event.preventDefault()),
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
    closeAllPanels: vi.fn(),
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

function makeSceneManager(hasFlyControls = true): {
  sceneManager: SceneManager;
  canvas: HTMLCanvasElement;
  flyHandleKeyDown: ReturnType<typeof vi.fn>;
  flyHandleKeyUp: ReturnType<typeof vi.fn>;
} {
  const flyHandleKeyDown = vi.fn();
  const flyHandleKeyUp = vi.fn();
  const canvas = document.createElement('canvas');
  const sceneManager = {
    renderer: { domElement: canvas },
    controls: {
      getFlyControls: () =>
        hasFlyControls
          ? {
              handleKeyDown: flyHandleKeyDown,
              handleKeyUp: flyHandleKeyUp,
            }
          : null,
    },
  } as unknown as SceneManager;
  return { sceneManager, canvas, flyHandleKeyDown, flyHandleKeyUp };
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

function setup(options: { hasFlyControls?: boolean } = {}) {
  const { manager, bindings } = makeContextManager();
  const { sceneManager, canvas, flyHandleKeyDown, flyHandleKeyUp } = makeSceneManager(
    options.hasFlyControls
  );
  const { console: debugConsole, toggle: debugToggle } = makeDebugConsole();
  const commands = makeCommands();
  const panelsBundle = makePanels();
  registerAllKeyBindings({
    contextManager: manager,
    sceneManager,
    debugConsole,
    animationShortcuts: {
      getSelectedDimension: () => -1,
      getAnimationManager: () => undefined,
    },
    panels: panelsBundle.panels,
    commands,
  });
  return {
    bindings,
    commands,
    canvas,
    flyHandleKeyDown,
    flyHandleKeyUp,
    debugToggle,
    panels: panelsBundle,
  };
}

function setupRealContextManager() {
  const contextManager = new InputContextManager();
  const { sceneManager, flyHandleKeyDown, flyHandleKeyUp } = makeSceneManager();
  const { console: debugConsole } = makeDebugConsole();
  const commands = makeCommands();
  const { panels } = makePanels();
  registerAllKeyBindings({
    contextManager,
    sceneManager,
    debugConsole,
    animationShortcuts: {
      getSelectedDimension: () => -1,
      getAnimationManager: () => undefined,
    },
    panels,
    commands,
  });
  return { contextManager, commands, flyHandleKeyDown, flyHandleKeyUp };
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
// navigation-bindings.ts) lack dedicated `.test.ts` files. Their
// behavior is covered here transitively via `registerAllKeyBindings`,
// which is the only documented entry point (the per-file exports are
// not consumed outside this orchestrator).
// The "Structure" + "NAVIGATION command dispatch" + "FLY_CONTROLS
// dispatch" blocks below exercise:
//   - fly-bindings.ts: WASD+modifier permutations, arrow keys, Shift boost.
//   - navigation-bindings.ts: [, ], 1-9, h/n/o/p/r/v/i/c/f/m/g/t/b/l/Space/Esc/Ctrl+Shift+S.
// Splitting them into per-file test files would buy zero coverage at
// the cost of duplicating the `setup()` boilerplate. Documented
// here for the audit trail; the audit recommendation is satisfied
// functionally.
//
// Ctrl/⌘+wheel FOV-vs-zoom exclusivity is no longer a key binding:
// each wheel handler reads the event's own live modifier flags, so
// there is no Control/Meta hold state to register or reconcile here
// (covered in controls-manager.test.ts and the orbit pointer tests).

describe('registerAllKeyBindings — structure', () => {
  it('registers unique action identities and explicit help visibility', () => {
    const { bindings } = setup();
    const actionKeys = bindings.map(
      (binding) =>
        `${binding.context}:${binding.actionId}:${binding.actionParameter ?? '<default>'}`
    );

    expect(new Set(actionKeys).size).toBe(actionKeys.length);
    expect(bindings.every((binding) => binding.description.length > 0)).toBe(true);
    expect(
      bindings.every((binding) => binding.help === false || binding.help.group.length > 0)
    ).toBe(true);
  });

  it('keeps grouped help metadata and descriptions consistent', () => {
    const { bindings } = setup();
    const groups = new Map<string, { description: string; help: ShortcutHelpMetadata }>();
    for (const binding of bindings) {
      if (!binding.help) continue;
      const prior = groups.get(binding.help.group);
      if (prior) {
        expect({ description: binding.description, help: binding.help }).toEqual(prior);
      } else {
        groups.set(binding.help.group, { description: binding.description, help: binding.help });
      }
    }
  });

  it('registers every action used for a control-rail shortcut', () => {
    const { contextManager } = setupRealContextManager();
    const railActions = [
      KeyAction.toggleHelp,
      KeyAction.recenterCamera,
      KeyAction.toggleControlMode,
      KeyAction.toggleDimensions,
      KeyAction.toggleRendering,
      KeyAction.toggleLayers,
      KeyAction.cycleDataMonitor,
      KeyAction.toggleDatasetBrowser,
      KeyAction.toggleRecording,
      KeyAction.toggleDebugConsole,
      KeyAction.toggleScaleBar,
      KeyAction.toggleColormapLegend,
      KeyAction.toggleOverlays,
      KeyAction.toggleCinematicMode,
      KeyAction.toggleFullscreen,
      KeyAction.togglePerformance,
    ];

    for (const actionId of railActions) {
      expect(contextManager.getShortcutLabel(actionId), actionId).toBeDefined();
    }
  });

  it('registers no Control/Meta hold bindings (wheel routing is stateless)', () => {
    const { bindings } = setup();
    const modifierHolds = bindings.filter((b) => b.key === 'Control' || b.key === 'Meta');
    expect(modifierHolds).toHaveLength(0);
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

  it('o dispatches the dataset-browser command', () => {
    const { bindings, commands } = setup();
    findBinding(bindings, InputContext.NAVIGATION, 'o').handler(new KeyboardEvent('keydown'));
    expect(commands.toggleDatasetBrowser).toHaveBeenCalledOnce();
  });

  it('element-menu shortcuts dispatch only while focus is on the scene or body', () => {
    const { bindings, canvas, commands } = setup();
    const button = document.createElement('button');
    const contextMenu = findBinding(bindings, InputContext.NAVIGATION, 'ContextMenu');
    const shiftF10 = findBinding(bindings, InputContext.NAVIGATION, 'F10', { shift: true });
    document.body.appendChild(canvas);
    document.body.appendChild(button);
    try {
      expect(contextMenu.preventDefault).not.toBe(true);
      expect(shiftF10.preventDefault).not.toBe(true);

      document.body.focus();
      const bodyEvent = new KeyboardEvent('keydown', { cancelable: true });
      contextMenu.handler(bodyEvent);
      expect(commands.openElementMenu).toHaveBeenCalledWith(bodyEvent);
      expect(bodyEvent.defaultPrevented).toBe(true);

      canvas.tabIndex = 0;
      canvas.focus();
      const canvasEvent = new KeyboardEvent('keydown', { shiftKey: true, cancelable: true });
      shiftF10.handler(canvasEvent);
      expect(commands.openElementMenu).toHaveBeenCalledWith(canvasEvent);
      expect(canvasEvent.defaultPrevented).toBe(true);

      button.focus();
      const buttonEvent = new KeyboardEvent('keydown', { cancelable: true });
      contextMenu.handler(buttonEvent);
      expect(commands.openElementMenu).toHaveBeenCalledTimes(2);
      expect(buttonEvent.defaultPrevented).toBe(false);
    } finally {
      canvas.remove();
      button.remove();
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
  it('declines fly events when no fly controls are active', () => {
    const { bindings } = setup({ hasFlyControls: false });
    const w = findBinding(bindings, InputContext.FLY_CONTROLS, 'w');
    expect(w.handler(new KeyboardEvent('keydown', { key: 'w' }))).toBe(false);
    expect(w.keyupHandler!(new KeyboardEvent('keyup', { key: 'w' }))).toBe(false);
  });

  it.each([InputContext.NAVIGATION, InputContext.FLY_CONTROLS])(
    'routes Ctrl+Shift+S to viewer-state export from %s',
    (context) => {
      const { contextManager, commands } = setupRealContextManager();
      contextManager.setContext(context);

      const handled = contextManager.handleKeyEvent(
        new KeyboardEvent('keydown', { key: 's', ctrlKey: true, shiftKey: true }),
        'down'
      );

      expect(handled).toBe(true);
      expect(commands.exportViewerState).toHaveBeenCalledTimes(1);
    }
  );

  it('keeps Shift+ArrowUp on fly look controls while FLY_CONTROLS is active', () => {
    const { contextManager, flyHandleKeyDown } = setupRealContextManager();
    const navigationHandler = vi.fn();
    contextManager.registerBinding(InputContext.NAVIGATION, {
      actionId: 'test.navigation.arrow-up',
      key: 'ArrowUp',
      modifiers: { shift: true },
      handler: navigationHandler,
      description: 'Test navigation arrow',
      help: false,
    });
    contextManager.setContext(InputContext.FLY_CONTROLS);

    const event = new KeyboardEvent('keydown', { key: 'ArrowUp', shiftKey: true });
    expect(contextManager.handleKeyEvent(event, 'down')).toBe(true);
    expect(flyHandleKeyDown).toHaveBeenCalledWith(event);
    expect(navigationHandler).not.toHaveBeenCalled();
  });

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

  it('Shift speed boost: keydown / keyup forward to the fly-controls handlers', () => {
    const { bindings, flyHandleKeyDown, flyHandleKeyUp } = setup();
    const shift = findBinding(bindings, InputContext.FLY_CONTROLS, 'Shift');
    const down = new KeyboardEvent('keydown', { key: 'Shift' });
    shift.handler(down);
    expect(flyHandleKeyDown).toHaveBeenCalledWith(down);
    const up = new KeyboardEvent('keyup', { key: 'Shift' });
    shift.keyupHandler!(up);
    expect(flyHandleKeyUp).toHaveBeenCalledWith(up);
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
