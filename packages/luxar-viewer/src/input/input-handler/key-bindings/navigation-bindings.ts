/**
 * NAVIGATION-context application bindings: all the orbit-mode UI
 * shortcuts (H, P, R, V, F, C, B, L, M, N, O, T, G, [, ], digits,
 * Escape, Space, Ctrl+L, Ctrl+Shift+S).
 */

import { config } from '../../../config';
import { log, Modules } from '../../../utils/log';
import { InputContext } from '../context-manager';
import type { KeyBindingsDeps } from './register-all';

export function registerNavigationBindings(deps: KeyBindingsDeps): void {
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
    key: config.input.keyboard.shortcuts.toggleHelp,
    handler: () => commands.toggleHelp(),
    preventDefault: true,
    description: 'Toggle help overlay',
  });

  // Dimension sliders
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: config.input.keyboard.shortcuts.toggleDimensions,
    handler: () => commands.toggleDimensionSliders(),
    preventDefault: true,
    description: 'Toggle dimension sliders',
  });

  // Dataset browser
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: config.input.keyboard.shortcuts.toggleDatasetBrowser,
    handler: () => window.dispatchEvent(new CustomEvent('open-dataset-browser')),
    preventDefault: true,
    description: 'Open dataset browser',
  });

  // Performance stats
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: config.input.keyboard.shortcuts.togglePerformance,
    handler: () => commands.togglePerformanceStats(),
    preventDefault: true,
    description: 'Toggle performance stats',
  });

  // Rendering controls (only without modifiers)
  contextManager.registerBinding(InputContext.NAVIGATION, {
    key: config.input.keyboard.shortcuts.toggleRendering,
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
    key: config.input.keyboard.shortcuts.toggleScaleBar,
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
    key: config.input.keyboard.shortcuts.recenterCamera,
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
    key: config.input.keyboard.shortcuts.toggleControlMode,
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
    key: config.input.keyboard.shortcuts.toggleInertialMode,
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
    key: config.input.keyboard.shortcuts.toggleCinematicMode,
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
    key: config.input.keyboard.shortcuts.toggleFullscreen,
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
