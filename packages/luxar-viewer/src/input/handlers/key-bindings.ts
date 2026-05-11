/**
 * Keyboard binding registration concern extracted from
 * `input/input-handler.ts`.
 *
 * Owns the entire key→command table — both the NAVIGATION-context
 * application bindings (H, P, R, V, F, C, etc.) and the
 * FLY_CONTROLS-context fly-mode bindings (WASD, arrows, Shift).
 * Plus the small piece of state that comes with FOV control:
 * a counter that gates `controls.setEnableZoom(true|false)` so a
 * Ctrl-or-Meta release while the other is still held doesn't
 * re-enable wheel zoom mid-gesture.
 *
 * The function takes everything it needs as a deps object.
 * Optional panels (scale bar, colormap legend, overlay manager,
 * recording panel, layers panel) are passed as getter functions so
 * the bindings always read the live reference at dispatch time —
 * the InputHandler may wire them in after this function runs.
 *
 * Behavior is byte-for-byte identical to the inline original; this
 * is a pure relocation + parameterization. The InputHandler still
 * implements every command (toggleHelp, toggleControlMode, etc.) and
 * passes them in via the `commands` block.
 *
 * @module input/handlers/key-bindings
 */

import { config } from '../../config';
import { log, Modules } from '../../utils/log';
import {
  InputContext,
  type InputContextManager,
} from '../input-context-manager';
import type { SceneManager } from '../../scene/scene-manager';
import type { ScaleBar } from '../../ui/components/scale-bar';
import type { ColormapLegend } from '../../ui/components/colormap-legend';
import type { OverlayManager } from '../../ui/helpers/overlay-manager';
import type { RecordingPanel } from '../../ui/recording-panel';
import type { LayersPanel } from '../../ui/layers';
import type { DebugConsole } from '../../ui/panels/debug-console';

/**
 * Command implementations the InputHandler still owns. Every binding
 * eventually dispatches into one of these — the key-binding module
 * is purely the table; the implementations stay with the host
 * InputHandler so they can read the rest of its state.
 */
export interface KeyBindingsCommands {
  navigateDimension(direction: -1 | 1): void;
  selectDimension(index: number): void;
  toggleHelp(): void;
  toggleDimensionSliders(): void;
  togglePerformanceStats(): void;
  toggleRenderingControls(): void;
  toggleControlMode(): void;
  toggleInertialMode(): void;
  toggleCinematicMode(): void;
  toggleFullscreen(): void;
  cycleDataMonitor(): void;
  recenterCamera(): void;
  exportViewerState(): void;
  handleEscape(): void;
  shouldHandleSpaceKey(): boolean;
}

/**
 * Late-bound optional panels — a binding may fire before the
 * corresponding panel has been wired in by the InputHandler's
 * setRecordingPanel / setLayersPanel / etc. setters. The getters
 * are called at dispatch time so the latest reference always wins.
 */
export interface KeyBindingsPanelGetters {
  getScaleBar(): ScaleBar | undefined;
  getColormapLegend(): ColormapLegend | undefined;
  getOverlayManager(): OverlayManager | undefined;
  getRecordingPanel(): RecordingPanel | undefined;
  getLayersPanel(): LayersPanel | undefined;
}

/** Everything `registerAllKeyBindings` needs to wire up the table. */
export interface KeyBindingsDeps {
  /** The input context manager — owns the actual binding registration. */
  contextManager: InputContextManager;
  /** SceneManager — needed for FOV-control zoom-toggling and fly controls. */
  sceneManager: SceneManager;
  /** Debug console (always present from InputHandler ctor). */
  debugConsole: DebugConsole;
  /**
   * Cleanup array shared with the InputHandler. The FOV-control
   * `blur` / `visibilitychange` listeners are appended here so the
   * InputHandler's `dispose()` cleans them up alongside its own
   * keydown / keyup window listeners.
   */
  cleanups: (() => void)[];
  panels: KeyBindingsPanelGetters;
  commands: KeyBindingsCommands;
}

/**
 * Wire every keyboard binding onto the provided context manager.
 *
 * Two sub-flows live here, in order:
 *
 *   1. NAVIGATION bindings (default, orbit-mode UI shortcuts).
 *      Includes the FOV-control gate that tracks Ctrl/Meta hold
 *      state via a small counter so multi-modifier presses don't
 *      glitch the zoom-enabled flag.
 *
 *   2. FLY_CONTROLS bindings (active in fly mode). WASD movement
 *      keys with all 4 modifier combinations the fly controls
 *      respect (none / Shift / Alt / Shift+Alt), arrow look keys
 *      with optional Shift, and the Shift speed-boost binding.
 */
export function registerAllKeyBindings(deps: KeyBindingsDeps): void {
  registerFovHoldGate(deps);
  registerNavigationBindings(deps);
  registerFlyControlBindings(deps);
}

// -- NAVIGATION-context: FOV-control gate -----------------------------------

/**
 * Ctrl / Meta hold registration. Disables wheel zoom while either is
 * held so Ctrl+wheel only adjusts FOV (the wheel handler in
 * `WindowEventHandler` reads the same modifier state). A counter
 * gates the toggle so releasing one key while the other is still held
 * doesn't re-enable zoom prematurely. A `blur` and a
 * `visibilitychange` listener reset the counter when the page loses
 * focus — otherwise a key release that happens off-page would leave
 * zoom permanently disabled.
 */
function registerFovHoldGate(deps: KeyBindingsDeps): void {
  const { contextManager, sceneManager, cleanups } = deps;

  let fovKeyHeldCount = 0;
  const resetFovKeyState = (): void => {
    if (fovKeyHeldCount === 0) return;
    fovKeyHeldCount = 0;
    sceneManager.controls.setEnableZoom(true);
  };
  const resetFovKeyStateWhenHidden = (): void => {
    if (document.visibilityState === 'hidden') resetFovKeyState();
  };

  for (const key of ['Control', 'Meta']) {
    contextManager.registerBinding(InputContext.NAVIGATION, {
      key,
      handler: () => {
        fovKeyHeldCount++;
        sceneManager.controls.setEnableZoom(false);
      },
      keyupHandler: () => {
        fovKeyHeldCount = Math.max(0, fovKeyHeldCount - 1);
        if (fovKeyHeldCount === 0) {
          sceneManager.controls.setEnableZoom(true);
        }
      },
      description: 'FOV control (hold Ctrl/Cmd + scroll to adjust field of view)',
    });
  }

  window.addEventListener('blur', resetFovKeyState);
  document.addEventListener('visibilitychange', resetFovKeyStateWhenHidden);
  cleanups.push(
    () => window.removeEventListener('blur', resetFovKeyState),
    () => document.removeEventListener('visibilitychange', resetFovKeyStateWhenHidden)
  );
}

// -- NAVIGATION-context: application bindings -------------------------------

function registerNavigationBindings(deps: KeyBindingsDeps): void {
  const { contextManager, debugConsole, panels, commands } = deps;

  // Dimension navigation
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: '[',
    handler: () => commands.navigateDimension(-1),
    preventDefault: true,
    description: 'Navigate dimension backward',
  });
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: ']',
    handler: () => commands.navigateDimension(1),
    preventDefault: true,
    description: 'Navigate dimension forward',
  });

  // Dimension selection (keys 1-9, only without modifiers)
  for (let i = 1; i <= 9; i++) {
    contextManager.registerBinding(InputContext.NAVIGATION, {
      key: String(i),
      handler: (event) => {
        if (!event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) {
          event.preventDefault();
          commands.selectDimension(i - 1);
        }
      },
      preventDefault: false,
      description: `Select dimension ${i}`,
    });
  }

  // Help overlay
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 'h',
    handler: () => commands.toggleHelp(),
    preventDefault: true,
    description: 'Toggle help overlay',
  });

  // Dimension sliders
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 'n',
    handler: () => commands.toggleDimensionSliders(),
    preventDefault: true,
    description: 'Toggle dimension sliders',
  });

  // Dataset browser
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 'o',
    handler: () => window.dispatchEvent(new CustomEvent('open-dataset-browser')),
    preventDefault: true,
    description: 'Open dataset browser',
  });

  // Performance stats
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 'p',
    handler: () => commands.togglePerformanceStats(),
    preventDefault: true,
    description: 'Toggle performance stats',
  });

  // Rendering controls (only without modifiers)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 'r',
    handler: (event) => {
      if (!event.metaKey && !event.ctrlKey && !event.shiftKey) {
        event.preventDefault();
        commands.toggleRenderingControls();
      }
    },
    preventDefault: false,
    description: 'Toggle rendering controls',
  });

  // Scale bar overlay
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 'b',
    handler: () => panels.getScaleBar()?.toggle(),
    preventDefault: true,
    description: 'Toggle scale bar',
  });

  // Colormap legend overlay
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: config.input.keyboard.shortcuts.toggleColormapLegend,
    handler: () => panels.getColormapLegend()?.toggle(),
    preventDefault: true,
    description: 'Toggle colormap legend',
  });

  // Screen-space overlays toggle
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: config.input.keyboard.shortcuts.toggleOverlays,
    handler: () => panels.getOverlayManager()?.toggle(),
    preventDefault: true,
    description: 'Toggle overlays',
  });

  // Recording panel toggle
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 't',
    handler: () => panels.getRecordingPanel()?.toggle(),
    preventDefault: true,
    description: 'Toggle recording panel',
  });

  // Quick screenshot
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 'g',
    handler: () => panels.getRecordingPanel()?.captureScreenshot(),
    preventDefault: true,
    description: 'Quick screenshot',
  });

  // Layers panel (L key without modifiers).
  // If the focus is already inside the panel (e.g. on a range slider,
  // select, or bound-edit text input), swallow L so dragging sliders
  // doesn't accidentally close the panel.
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: config.input.keyboard.shortcuts.toggleLayers,
    handler: (event) => {
      if (event.metaKey || event.ctrlKey || event.shiftKey) return;
      const active = document.activeElement as HTMLElement | null;
      if (active && active.closest('.luxar-layers-panel')) return;
      event.preventDefault();
      panels.getLayersPanel()?.toggle();
    },
    preventDefault: false,
    description: 'Toggle layers panel',
  });

  // Debug console (Ctrl+L)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 'l',
    modifiers: { ctrl: true },
    handler: () => {
      debugConsole.toggle();
      log.info(
        Modules.DEBUG_CONSOLE,
        `Debug console ${debugConsole.getIsVisible() ? 'opened' : 'closed'}`
      );
    },
    preventDefault: true,
    description: 'Toggle debug console',
  });

  // Data loading monitor (M key, no modifiers)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 'm',
    handler: (event) => {
      if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
        event.preventDefault();
        commands.cycleDataMonitor();
      }
    },
    preventDefault: false,
    description: 'Cycle data loading monitor',
  });

  // Recenter camera (F key, no modifiers)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 'f',
    handler: (event) => {
      if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
        event.preventDefault();
        commands.recenterCamera();
      }
    },
    preventDefault: false,
    description: 'Recenter camera on scene',
  });

  // Toggle control mode (V key, no modifiers)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 'v',
    handler: (event) => {
      if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
        event.preventDefault();
        commands.toggleControlMode();
      }
    },
    preventDefault: false,
    description: 'Cycle control mode (orbit/fly/ortho)',
  });

  // Toggle inertial mode (I key, no modifiers)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 'i',
    handler: (event) => {
      if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
        event.preventDefault();
        commands.toggleInertialMode();
      }
    },
    preventDefault: false,
    description: 'Toggle inertial mode (fly controls)',
  });

  // Toggle cinematic mode (C key, no modifiers)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 'c',
    handler: (event) => {
      if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
        event.preventDefault();
        commands.toggleCinematicMode();
      }
    },
    preventDefault: false,
    description: 'Toggle cinematic mode',
  });

  // Fullscreen toggle (Space, context-aware)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: ' ',
    handler: (event) => {
      if (commands.shouldHandleSpaceKey()) {
        event.preventDefault();
        commands.toggleFullscreen();
      }
    },
    preventDefault: false,
    description: 'Toggle fullscreen',
  });

  // Escape key — context-aware panel closing
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 'Escape',
    handler: () => commands.handleEscape(),
    preventDefault: true,
    description: 'Close panels / Exit fullscreen',
  });

  // Export viewer state (Ctrl+Shift+S)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: 's',
    modifiers: { ctrl: true, shift: true },
    handler: (event) => {
      event.preventDefault();
      commands.exportViewerState();
    },
    preventDefault: true,
    description: 'Export viewer state to clipboard',
  });
}

// -- FLY_CONTROLS-context: fly-mode bindings --------------------------------

/**
 * Fly-mode bindings. Movement keys (WASD, Q, E) need both keydown
 * and keyup handlers — fly controls hold the key state itself and
 * integrate movement per-frame. We register every WASD key with all
 * four supported modifier combinations because the fly controls
 * differentiate behavior on Shift (speed boost) and Alt (vertical
 * for W/S only).
 */
function registerFlyControlBindings(deps: KeyBindingsDeps): void {
  const { contextManager, sceneManager } = deps;
  const getFlyControls = () => sceneManager.controls.getFlyControls();

  const flyMovementKeys = ['w', 'a', 's', 'd', 'q', 'e'];
  const modifierCombinations = [
    {}, // No modifiers
    { shift: true }, // Shift only (speed boost)
    { alt: true }, // Alt only (vertical for W/S)
    { shift: true, alt: true }, // Shift+Alt (fast vertical)
  ];

  for (const key of flyMovementKeys) {
    for (const modifiers of modifierCombinations) {
      contextManager.registerBinding(InputContext.FLY_CONTROLS, {
        key,
        modifiers: Object.keys(modifiers).length > 0 ? modifiers : undefined,
        handler: (event) => getFlyControls()?.handleKeyDown(event),
        keyupHandler: (event) => getFlyControls()?.handleKeyUp(event),
        description: `Fly: ${key.toUpperCase()}${
          modifiers.shift ? '+Shift' : ''
        }${modifiers.alt ? '+Alt' : ''}`,
      });
    }
  }

  // Arrow keys for look direction (with and without Shift)
  const arrowKeys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
  for (const key of arrowKeys) {
    contextManager.registerBinding(InputContext.FLY_CONTROLS, {
      key,
      handler: (event) => getFlyControls()?.handleKeyDown(event),
      keyupHandler: (event) => getFlyControls()?.handleKeyUp(event),
      description: `Fly look: ${key}`,
    });
    contextManager.registerBinding(InputContext.FLY_CONTROLS, {
      key,
      modifiers: { shift: true },
      handler: (event) => getFlyControls()?.handleKeyDown(event),
      keyupHandler: (event) => getFlyControls()?.handleKeyUp(event),
      description: `Fly look: ${key}+Shift`,
    });
  }

  // Shift speed boost
  contextManager.registerBinding(InputContext.FLY_CONTROLS, {
    key: 'Shift',
    handler: () => sceneManager.controls.setEnableZoom(false),
    keyupHandler: () => sceneManager.controls.setEnableZoom(true),
    description: 'Speed boost + zoom control',
  });
}
