/**
 * Assemble the control-rail item descriptors.
 *
 * Extracted from the init pipeline so the rail's wiring lives in one focused,
 * independently-testable place rather than inline in the orchestrator. Each
 * button fires the SAME command as its keyboard shortcut (via
 * `inputHandler.getUiActions()`), so on-screen and keyboard behaviour never
 * drift. Rich controls open rail popovers (see `ui/rail-panels/`).
 *
 * @module core/app/init/build-rail-items
 */

import { RAIL_ICONS, type ControlRailItem } from '../../../ui/control-rail';
import { buildSettingsPopover } from '../../../ui/rail-panels/settings-popover';
import { buildNavigationPopover } from '../../../ui/rail-panels/navigation-popover';
import { buildPerformancePopover } from '../../../ui/rail-panels/performance-popover';
import { nextControlType } from '../../../input/input-handler/commands/control-mode';
import type { InputHandler } from '../../../input/input-handler';
import type { SceneManager } from '../../../scene/scene-manager';
import type { RenderingControls } from '../../../ui/rendering-controls';
import type { AnimationController } from '../../../scene/animation/animation-controller';
import type { AdaptiveDPRManager } from '../../../rendering/adaptive-dpr-manager';
import type { PerformanceMonitor } from '../../../ui/performance-monitor';
import type { LayersPanel } from '../../../ui/layers';
import type { DebugConsole } from '../../../ui/debug-console';
import type { RecordingPanel } from '../../../ui/recording-panel';

/** Everything the rail item closures reference (all constructed by the pipeline). */
export interface RailItemsDeps {
  /** The command + panel action surface shared with the keyboard bindings. */
  ui: ReturnType<InputHandler['getUiActions']>;
  sceneManager: SceneManager;
  renderingControls: RenderingControls;
  animationController: AnimationController;
  adaptiveDPRManager: AdaptiveDPRManager;
  performanceMonitor: PerformanceMonitor;
  layersPanel: LayersPanel;
  debugConsole: DebugConsole;
  recordingPanel: RecordingPanel;
}

/**
 * Build the ordered list of control-rail items. Pure assembly of object
 * literals — the closures only run later, on user interaction.
 */
export function buildRailItems(deps: RailItemsDeps): ControlRailItem[] {
  const {
    ui,
    sceneManager,
    renderingControls,
    animationController,
    adaptiveDPRManager,
    performanceMonitor,
    layersPanel,
    debugConsole,
    recordingPanel,
  } = deps;

  return [
    {
      id: 'help',
      title: 'Help & shortcuts',
      shortcut: 'H',
      icon: RAIL_ICONS.help,
      activate: () => ui.commands.toggleHelp(),
      openSelector: '#luxar-help-overlay',
    },
    {
      // Navigation: left-click cycles orbit → fly → ortho (reuses the exact V
      // shortcut command, incl. the input-context switch); right-click opens
      // the current mode's parameters. The icon mirrors the live control mode.
      id: 'nav',
      title: 'Navigation',
      shortcut: 'V',
      icon: RAIL_ICONS.navOrbit,
      activate: () => ui.commands.toggleControlMode(),
      render: (btn) => {
        const type = sceneManager.getControlType();
        if (btn.dataset.navMode === type) return;
        btn.dataset.navMode = type;
        const icon =
          type === 'fly'
            ? RAIL_ICONS.navFly
            : type === 'ortho'
              ? RAIL_ICONS.navOrtho
              : RAIL_ICONS.navOrbit;
        const svg = btn.querySelector('svg');
        if (svg) svg.outerHTML = icon;
        // `type` is a fixed enum (orbit|fly|ortho) — safe to interpolate.
        const modeLabel = type.charAt(0).toUpperCase() + type.slice(1);
        const next = nextControlType(type); // next mode a click would select
        btn.setAttribute(
          'aria-label',
          `Navigation: ${modeLabel} — click for ${next} (V), right-click for options`
        );
        // The button icon is cryptic on its own, so name the current mode in
        // the hover tooltip too: e.g. "Navigation · Orbit  V".
        const tip = btn.querySelector('.luxar-control-rail__tip');
        if (tip) tip.innerHTML = `Navigation · ${modeLabel}<kbd>V</kbd>`;
      },
      popover: {
        trigger: 'context',
        title: 'Navigation',
        build: (host) =>
          buildNavigationPopover(host, {
            settings: renderingControls.settings,
            sceneManager,
            animationController,
            saveSettings: () => renderingControls.saveSettings(),
            triggerAnimation: () => animationController.startAnimation(),
            setMode: (type) => ui.commands.setControlMode(type),
          }),
      },
    },
    {
      id: 'dims',
      title: 'Dimensions',
      shortcut: 'N',
      icon: RAIL_ICONS.dims,
      activate: () => ui.commands.toggleDimensionSliders(),
      openSelector: '.luxar-dimension-sliders',
    },
    {
      id: 'render',
      title: 'Rendering',
      shortcut: 'R',
      icon: RAIL_ICONS.render,
      activate: () => ui.commands.toggleRenderingControls(),
      isActive: () => renderingControls.isVisible(),
    },
    {
      id: 'layers',
      title: 'Layers',
      shortcut: 'L',
      icon: RAIL_ICONS.layers,
      activate: () => ui.panels.getLayersPanel()?.toggle(),
      isActive: () => layersPanel.isVisible(),
      // Grayed + non-clickable until the scene actually has layers (nodes with
      // layer=true). Refreshed on the 'luxar-layers-changed' event after load.
      disabled: () => layersPanel.layerState.count === 0,
    },
    {
      id: 'monitor',
      title: 'Data monitor',
      shortcut: 'M',
      icon: RAIL_ICONS.monitor,
      activate: () => ui.commands.cycleDataMonitor(),
      openSelector: '.luxar-data-monitor',
    },
    {
      id: 'data',
      title: 'Datasets',
      shortcut: 'O',
      icon: RAIL_ICONS.data,
      activate: () => window.dispatchEvent(new CustomEvent('open-dataset-browser')),
      openSelector: '.luxar-dataset-browser',
    },
    {
      id: 'recording',
      title: 'Recording',
      shortcut: 'T',
      icon: RAIL_ICONS.recording,
      activate: () => ui.panels.getRecordingPanel()?.toggle(),
      isActive: () => recordingPanel.isVisible(),
      separatorBefore: true,
    },
    {
      id: 'logs',
      title: 'Logs (console)',
      shortcut: 'Ctrl+L',
      icon: RAIL_ICONS.logs,
      activate: () => debugConsole.toggle(),
      isActive: () => debugConsole.getIsVisible(),
      separatorBefore: true,
    },
    {
      id: 'view',
      title: 'View options',
      icon: RAIL_ICONS.view,
      activate: () => {}, // unused — opens the flyout below
      separatorBefore: true,
      flyout: [
        {
          id: 'scalebar',
          title: 'Scale bar',
          shortcut: 'B',
          icon: RAIL_ICONS.scalebar,
          activate: () => ui.panels.getScaleBar()?.toggle(),
          openSelector: '.luxar-scale-bar',
        },
        {
          id: 'legend',
          title: 'Colormap legend',
          shortcut: 'J',
          icon: RAIL_ICONS.legend,
          activate: () => ui.panels.getColormapLegend()?.toggle(),
          openSelector: '.luxar-colormap-legend',
        },
        {
          id: 'overlays',
          title: 'Overlays',
          shortcut: 'U',
          icon: RAIL_ICONS.overlays,
          activate: () => ui.panels.getOverlayManager()?.toggle(),
          openSelector: '.luxar-overlay:not(.luxar-overlay--hidden)',
        },
        {
          id: 'cinematic',
          title: 'Cinematic mode',
          shortcut: 'C',
          icon: RAIL_ICONS.cinematic,
          activate: () => ui.commands.toggleCinematicMode(),
          isActive: () => renderingControls.settings.cinematicMode,
        },
      ],
    },
    {
      // Viewer-wide preferences (theme, and future settings) — kept off the top
      // level so the rail stays focused. Click opens the Settings popover.
      id: 'settings',
      title: 'Settings',
      icon: RAIL_ICONS.settings,
      activate: () => {}, // unused — opens the popover below
      popover: {
        trigger: 'click',
        title: 'Settings',
        build: (host) =>
          buildSettingsPopover(host, {
            triggerAnimation: () => animationController.startAnimation(),
          }),
      },
    },
    {
      // The gauge toggles the perf readout docked below (rail footer). Placed
      // just under the eye so the readout appears at the very bottom of the rail.
      // Right-click opens the adaptive-resolution (DPR) controls popover.
      id: 'perf',
      title: 'Performance',
      shortcut: 'P',
      icon: RAIL_ICONS.perf,
      activate: () => ui.commands.togglePerformanceStats(),
      isActive: () => performanceMonitor.visible,
      popover: {
        trigger: 'context',
        title: 'Performance',
        build: (host) =>
          buildPerformancePopover(host, {
            settings: renderingControls.settings,
            manager: adaptiveDPRManager,
            saveSettings: () => renderingControls.saveSettings(),
            triggerAnimation: () => animationController.startAnimation(),
          }),
      },
    },
  ];
}
