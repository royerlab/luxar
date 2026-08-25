/**
 * NAVIGATION-context application bindings: all the orbit-mode UI
 * shortcuts (H, P, R, V, F, C, B, L, M, N, O, T, G, [, ], digits,
 * Escape, Space, Ctrl+L, Ctrl+Shift+S).
 */

import { config } from '../../../config';
import { log, Modules } from '../../../utils/log';
import { InputContext } from '../context-manager';
import { isFocusOnSceneCanvas } from '../../../utils/dom/focus';
import type { KeyBindingsDeps } from './register-all';
import { KeyAction } from './actions';

export function registerNavigationBindings(deps: KeyBindingsDeps): void {
  const { contextManager, debugConsole, panels, commands, sceneManager } = deps;

  // Dimension navigation
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.navigateDimension,
    actionParameter: -1,
    key: '[',
    handler: () => commands.navigateDimension(-1),
    preventDefault: true,
    description: 'Step along the selected dimension',
    help: { section: 'dimensions', group: 'dimension-step', keys: ['[', ']'], order: 20 },
  });
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.navigateDimension,
    actionParameter: 1,
    key: ']',
    handler: () => commands.navigateDimension(1),
    preventDefault: true,
    description: 'Step along the selected dimension',
    help: { section: 'dimensions', group: 'dimension-step', keys: ['[', ']'], order: 20 },
  });

  // Dimension selection (keys 1-9, only without modifiers)
  for (let i = 1; i <= 9; i++) {
    contextManager.registerBinding(InputContext.NAVIGATION, {
      actionId: KeyAction.selectDimension,
      actionParameter: i - 1,
      key: String(i),
      handler: (event) => {
        if (!event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) {
          event.preventDefault();
          commands.selectDimension(i - 1);
        }
      },
      preventDefault: false,
      description: 'Select a non-displayed dimension (panel header shows target)',
      help: {
        section: 'dimensions',
        group: 'dimension-select',
        keys: ['1 – 9'],
        order: 10,
      },
    });
  }

  // Help overlay
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.toggleHelp,
    key: config.input.keyboard.shortcuts.toggleHelp,
    handler: () => commands.toggleHelp(),
    preventDefault: true,
    description: 'Toggle help overlay',
    help: { section: 'basics', group: 'toggle-help', order: 50 },
  });

  // Dimension sliders
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.toggleDimensions,
    key: config.input.keyboard.shortcuts.toggleDimensions,
    handler: () => commands.toggleDimensionSliders(),
    preventDefault: true,
    description: 'Toggle dimension sliders',
    help: {
      section: 'dimensions',
      group: 'toggle-dimensions',
      order: 40,
    },
  });

  // Dataset browser
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.toggleDatasetBrowser,
    key: config.input.keyboard.shortcuts.toggleDatasetBrowser,
    handler: () => commands.toggleDatasetBrowser(),
    preventDefault: true,
    description: 'Toggle dataset browser',
    help: { section: 'basics', group: 'dataset-browser', order: 40 },
  });

  // Element context menu at the current hover (issue #1917) — the keyboard
  // path the UI design guide requires of every context menu (§7.8).
  //
  // Registered on NAVIGATION, not FLY_CONTROLS: that context uses an
  // `allowedKeys` whitelist which contains neither key, and reaches a
  // NAVIGATION binding only through passthrough — so registering here is what
  // makes the shortcut work in BOTH modes.
  //
  // The command surface owns the window event dispatch because the listener is
  // rebuilt on every dataset load while this binding lives for the app's lifetime.
  //
  // Gated on focus being on the scene itself. Layer rows contain these keys,
  // but the guard still matters when the panel is reached by mouse: a cached
  // scene pick may remain while focus sits on body, and `openContextMenu` is
  // module-global. Without the focus check, the keyboard shortcut could open
  // a canvas menu while the user is interacting with UI outside the canvas.
  // Same guard the other scene-scoped global keys use.
  const openElementMenu = (event: KeyboardEvent): void => {
    if (!isFocusOnSceneCanvas(document.activeElement, sceneManager.renderer?.domElement ?? null)) {
      return;
    }
    commands.openElementMenu(event);
  };
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.openElementMenu,
    actionParameter: 'shift-f10',
    key: 'F10',
    modifiers: { shift: true },
    handler: openElementMenu,
    description: 'Context menu for the hovered element',
    help: {
      section: 'panels',
      group: 'element-menu',
      keys: ['⇧', 'F10'],
      order: 140,
    },
  });
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.openElementMenu,
    actionParameter: 'context-menu',
    key: 'ContextMenu',
    handler: openElementMenu,
    description: 'Context menu for the hovered element',
    help: false,
  });

  // Performance stats
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.togglePerformance,
    key: config.input.keyboard.shortcuts.togglePerformance,
    handler: () => commands.togglePerformanceStats(),
    preventDefault: true,
    description: 'Toggle performance stats',
    help: { section: 'panels', group: 'performance', order: 50 },
  });

  // Rendering controls (only without modifiers)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.toggleRendering,
    key: config.input.keyboard.shortcuts.toggleRendering,
    handler: (event) => {
      if (!event.metaKey && !event.ctrlKey && !event.shiftKey) {
        event.preventDefault();
        commands.toggleRenderingControls();
      }
    },
    preventDefault: false,
    description: 'Toggle rendering controls',
    help: { section: 'panels', group: 'rendering', order: 10 },
  });

  // Scale bar overlay
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.toggleScaleBar,
    key: config.input.keyboard.shortcuts.toggleScaleBar,
    handler: () => panels.getScaleBar()?.toggle(),
    preventDefault: true,
    description: 'Toggle scale bar',
    help: { section: 'panels', group: 'scale-bar', order: 80 },
  });

  // Colormap legend overlay
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.toggleColormapLegend,
    key: config.input.keyboard.shortcuts.toggleColormapLegend,
    handler: () => panels.getColormapLegend()?.toggle(),
    preventDefault: true,
    description: 'Toggle colormap legend',
    help: { section: 'panels', group: 'colormap-legend', order: 90 },
  });

  // Screen-space overlays toggle
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.toggleOverlays,
    key: config.input.keyboard.shortcuts.toggleOverlays,
    handler: () => panels.getOverlayManager()?.toggle(),
    preventDefault: true,
    description: 'Toggle overlays',
    help: { section: 'panels', group: 'overlays', order: 100 },
  });

  // Recording panel toggle
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.toggleRecording,
    key: 't',
    handler: () => panels.getRecordingPanel()?.toggle(),
    preventDefault: true,
    description: 'Recording panel (screenshot / video)',
    help: { section: 'panels', group: 'recording', order: 60 },
  });

  // Quick screenshot
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.captureScreenshot,
    key: 'g',
    handler: () => panels.getRecordingPanel()?.captureScreenshot(),
    preventDefault: true,
    description: 'Quick screenshot',
    help: { section: 'panels', group: 'screenshot', order: 70 },
  });

  // Layers panel (L key without modifiers).
  // If the focus is already inside the panel (e.g. on a range slider,
  // select, or bound-edit text input), swallow L so dragging sliders
  // doesn't accidentally close the panel.
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.toggleLayers,
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
    help: { section: 'panels', group: 'layers', order: 20 },
  });

  // Debug console (Ctrl+L)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.toggleDebugConsole,
    key: config.input.keyboard.shortcuts.toggleDebugConsole,
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
    help: { section: 'panels', group: 'debug-console', order: 120 },
  });

  // Data loading monitor (M key, no modifiers)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.cycleDataMonitor,
    key: 'm',
    handler: (event) => {
      if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
        event.preventDefault();
        commands.cycleDataMonitor();
      }
    },
    preventDefault: false,
    description: 'Data monitor (mini / expanded / off)',
    help: { section: 'panels', group: 'data-monitor', order: 40 },
  });

  // Recenter camera (F key, no modifiers)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.recenterCamera,
    key: config.input.keyboard.shortcuts.recenterCamera,
    handler: (event) => {
      if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
        event.preventDefault();
        commands.recenterCamera();
      }
    },
    preventDefault: false,
    description: 'Recenter camera on scene',
    help: { section: 'basics', group: 'recenter-camera', order: 20 },
  });

  // Toggle control mode (V key, no modifiers)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.toggleControlMode,
    key: config.input.keyboard.shortcuts.toggleControlMode,
    handler: (event) => {
      if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
        event.preventDefault();
        commands.toggleControlMode();
      }
    },
    preventDefault: false,
    description: 'Cycle control mode (orbit/fly/ortho)',
    help: { section: 'basics', group: 'control-mode', order: 30 },
  });

  // Toggle inertial mode (I key, no modifiers)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.toggleInertialMode,
    key: config.input.keyboard.shortcuts.toggleInertialMode,
    handler: (event) => {
      if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
        event.preventDefault();
        commands.toggleInertialMode();
      }
    },
    preventDefault: false,
    description: 'Toggle inertial mode (fly controls)',
    help: { section: 'fly', group: 'inertial-mode', order: 60 },
  });

  // Toggle cinematic mode (C key, no modifiers)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.toggleCinematicMode,
    key: config.input.keyboard.shortcuts.toggleCinematicMode,
    handler: (event) => {
      if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
        event.preventDefault();
        commands.toggleCinematicMode();
      }
    },
    preventDefault: false,
    description: 'Cinematic mode (bloom / noise / vignette / lens)',
    help: { section: 'panels', group: 'cinematic', order: 110 },
  });

  // Fullscreen toggle (Space, context-aware)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.toggleFullscreen,
    key: config.input.keyboard.shortcuts.toggleFullscreen,
    handler: (event) => {
      if (commands.shouldHandleSpaceKey()) {
        event.preventDefault();
        commands.toggleFullscreen();
      }
    },
    preventDefault: false,
    description: 'Toggle fullscreen',
    help: { section: 'basics', group: 'fullscreen', order: 10 },
  });

  // Escape key — context-aware panel closing
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.handleEscape,
    key: 'Escape',
    handler: () => commands.handleEscape(),
    preventDefault: true,
    description: 'Close panels / Exit fullscreen',
    help: { section: 'basics', group: 'escape', order: 60 },
  });

  // Export viewer state (Ctrl+Shift+S)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    actionId: KeyAction.exportViewerState,
    key: 's',
    modifiers: { ctrl: true, shift: true },
    handler: (event) => {
      event.preventDefault();
      commands.exportViewerState();
    },
    preventDefault: true,
    description: 'Export viewer state to clipboard',
    help: {
      section: 'panels',
      group: 'export-viewer-state',
      order: 130,
    },
  });
}
