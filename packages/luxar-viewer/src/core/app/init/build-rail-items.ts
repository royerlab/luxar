/**
 * Assemble the control-rail item descriptors.
 *
 * Extracted from the init pipeline so the rail's wiring lives in one focused,
 * independently-testable place rather than inline in the orchestrator.
 * Buttons dispatch through `inputHandler.getUiActions()` so on-screen and
 * keyboard controls share one command surface. Rich controls open rail
 * popovers (see `ui/rail-panels/`).
 *
 * @module core/app/init/build-rail-items
 */

import { RAIL_ICONS, type ControlRailItem } from '../../../ui/control-rail';
import { isDocumentFullscreen } from '../../../utils/fullscreen';
import { buildSettingsPopover } from '../../../ui/rail-panels/settings-popover';
import { buildNavigationPopover } from '../../../ui/rail-panels/navigation-popover';
import { buildPerformancePopover } from '../../../ui/rail-panels/performance-popover';
import { buildHomePopover } from '../../../ui/rail-panels/home-popover';
import { buildAudioPopover } from '../../../ui/rail-panels/audio-popover';
import { getSceneLoader } from '../../../data/scene-loader-manager';
import type { InputHandler } from '../../../input';
import { nextControlType } from '../../../controls/types';
import type { SceneManager } from '../../../scene/scene-manager';
import type { SceneDimsManager } from '../../../scene/scene-dims-manager';
import type { RenderingControls } from '../../../ui/rendering-controls';
import type { DensityGuardControl } from '../../../ui/rendering-controls/types';
import type { AnimationController } from '../../../scene/animation/animation-controller';
import type { AdaptiveDPRManager } from '../../../rendering/adaptive-dpr-manager';
import type { PerformanceMonitor } from '../../../ui/performance-monitor';
import type { LayersPanel } from '../../../ui/layers';
import type { DebugConsole } from '../../../ui/debug-console';
import type { RecordingPanel } from '../../../ui/recording-panel';
import type { AudioEngine } from '../../../audio/audio-engine';
import { KeyAction, type KeyActionId } from '../../../input';
import { getInputProfile } from '../../../utils/input-capabilities';
import { notifier } from '../../../utils/cross-layer/notifier';
import { eventBus } from '../../../utils/cross-layer/event-bus';

/** Everything the rail item closures reference (all constructed by the pipeline). */
export interface RailItemsDeps {
  /** The command + panel action surface shared with the keyboard bindings. */
  ui: ReturnType<InputHandler['getUiActions']>;
  shortcutForAction(actionId: KeyActionId): string | undefined;
  sceneManager: SceneManager;
  /** Shared nD dimension state (the pipeline passes the module singleton). */
  sceneDims: SceneDimsManager;
  renderingControls: RenderingControls;
  animationController: AnimationController;
  adaptiveDPRManager: AdaptiveDPRManager;
  /** Runtime handle on the projected-density guard, for the Performance popover. */
  densityGuard: DensityGuardControl;
  performanceMonitor: PerformanceMonitor;
  layersPanel: LayersPanel;
  debugConsole: DebugConsole;
  recordingPanel: RecordingPanel;
  /** The sound layer; the Sound button exists only while it has nodes. */
  audioEngine: AudioEngine;
}

/**
 * Build the ordered list of control-rail items. Pure assembly of object
 * literals — the closures only run later, on user interaction.
 */
export function buildRailItems(deps: RailItemsDeps): ControlRailItem[] {
  const {
    ui,
    shortcutForAction,
    sceneManager,
    sceneDims,
    renderingControls,
    animationController,
    adaptiveDPRManager,
    densityGuard,
    performanceMonitor,
    layersPanel,
    debugConsole,
    recordingPanel,
    audioEngine,
  } = deps;
  const controlModeShortcut = shortcutForAction(KeyAction.toggleControlMode);
  const coarse = getInputProfile().coarsePointer;

  // Rendering, Layers, and Recording all dock at the same spot beside the
  // rail (left: 73px), so the rail opens ONE floating surface at a time —
  // activating a docked panel closes the other docked panels, and opening
  // any rail popover (Settings, Navigation, Performance, Home) closes the
  // docked panels too (the reverse is automatic: clicking a rail button is
  // an outside-click, which dismisses an open popover). Keyboard shortcuts
  // are deliberately not routed through this: power users may still stack
  // panels explicitly via R/L/T.
  type LeftSurface = 'help' | 'render' | 'layers' | 'monitor' | 'recording';
  const closeCoarseSurfaces = (except?: LeftSurface): void => {
    if (!coarse) return;
    if (except !== 'help') notifier.hideHelp();
    if (except !== 'monitor') eventBus.emit('panel-hide', { panelId: 'data-monitor' });
  };
  const closeOtherLeftPanels = (except?: LeftSurface): void => {
    closeCoarseSurfaces(except);
    if (except !== 'render' && renderingControls.isVisible()) {
      ui.commands.toggleRenderingControls();
    }
    if (except !== 'layers' && layersPanel.isVisible()) {
      ui.panels.getLayersPanel()?.toggle();
    }
    if (except !== 'recording' && recordingPanel.isVisible()) {
      ui.panels.getRecordingPanel()?.toggle();
    }
  };

  // Touch-first devices (a coarse primary pointer): a phone has room for ONE
  // floating surface, so the help overlay and the data monitor join the
  // docked panels' exclusivity, and a "Hide panels" button provides an
  // unconditional close action. Desktop keeps stacking and no new item.
  const closeOthersOnCoarse = (except: 'help' | 'monitor'): void => {
    if (coarse) closeOtherLeftPanels(except);
  };
  // The Fullscreen API is absent on iPhone Safari (`fullscreenEnabled` is
  // false and there is no webkit fallback on the document), so the chip would
  // be dead there; gate on the real capability rather than on the device.
  const fullscreenAvailable =
    typeof document === 'undefined' ||
    !!(
      document.fullscreenEnabled ||
      (document as Document & { webkitFullscreenEnabled?: boolean }).webkitFullscreenEnabled
    );

  return [
    {
      id: 'help',
      title: 'Help & shortcuts',
      shortcut: shortcutForAction(KeyAction.toggleHelp),
      icon: RAIL_ICONS.help,
      activate: () => {
        closeOthersOnCoarse('help');
        ui.commands.toggleHelp();
      },
      openSelector: '#luxar-help-overlay',
    },
    {
      // Home: left-click reframes the camera to fit the whole scene (the F
      // shortcut); right-click opens deeper resets (origin / dimensions /
      // rendering). Momentary — a one-shot action, never shows active state.
      id: 'home',
      title: 'Home · fit scene',
      shortcut: shortcutForAction(KeyAction.recenterCamera),
      icon: RAIL_ICONS.home,
      momentary: true,
      activate: () => ui.commands.recenterCamera(),
      popover: {
        trigger: 'context',
        title: 'Home',
        build: (host) => {
          closeOtherLeftPanels();
          return buildHomePopover(host, {
            fitScene: () => ui.commands.recenterCamera(),
            centerOnOrigin: () => sceneManager.centerOnOrigin(),
            resetDimensions: () => sceneDims.resetPositions(),
            hasDimensionSliders: () => sceneDims.hasNonDisplayedDimensions(),
            resetRendering: () => renderingControls.resetToDefaults(),
            resetLayers: () => layersPanel.resetAllLayers(),
            hasLayers: () => layersPanel.layerState.count > 0,
            triggerAnimation: () => animationController.startAnimation(),
          });
        },
      },
    },
    {
      // Navigation: left-click cycles orbit → fly → ortho (reuses the exact V
      // shortcut command, incl. the input-context switch); right-click opens
      // the current mode's parameters. The icon mirrors the live control mode.
      id: 'nav',
      title: 'Navigation',
      shortcut: controlModeShortcut,
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
        const next = nextControlType(type);
        const shortcutLabel = controlModeShortcut ? ` (${controlModeShortcut})` : '';
        btn.setAttribute(
          'aria-label',
          `Navigation: ${modeLabel} — click for ${next}${shortcutLabel}, right-click or hold for options`
        );
        // The button icon is cryptic on its own, so name the current mode in
        // the hover tooltip too: e.g. "Navigation · Orbit  V".
        const tip = btn.querySelector('.luxar-control-rail__tip');
        if (tip) {
          tip.textContent = `Navigation · ${modeLabel}`;
          if (controlModeShortcut) {
            const shortcut = document.createElement('kbd');
            shortcut.textContent = controlModeShortcut;
            tip.appendChild(shortcut);
          }
        }
      },
      popover: {
        trigger: 'context',
        title: 'Navigation',
        build: (host) => {
          closeOtherLeftPanels();
          return buildNavigationPopover(host, {
            settings: renderingControls.settings,
            sceneManager,
            animationController,
            saveSettings: () => renderingControls.saveSettings(),
            triggerAnimation: () => animationController.startAnimation(),
            setMode: (type) => ui.commands.setControlMode(type),
          });
        },
      },
    },
    {
      id: 'dims',
      title: 'Dimensions',
      shortcut: shortcutForAction(KeyAction.toggleDimensions),
      icon: RAIL_ICONS.dims,
      activate: () => ui.commands.toggleDimensionSliders(),
      openSelector: '.luxar-dimension-sliders',
    },
    {
      id: 'render',
      title: 'Rendering',
      shortcut: shortcutForAction(KeyAction.toggleRendering),
      icon: RAIL_ICONS.render,
      activate: () => {
        if (!renderingControls.isVisible()) closeOtherLeftPanels('render');
        ui.commands.toggleRenderingControls();
      },
      isActive: () => renderingControls.isVisible(),
    },
    {
      id: 'layers',
      title: 'Layers',
      shortcut: shortcutForAction(KeyAction.toggleLayers),
      icon: RAIL_ICONS.layers,
      activate: () => {
        if (!layersPanel.isVisible()) closeOtherLeftPanels('layers');
        ui.panels.getLayersPanel()?.toggle();
      },
      isActive: () => layersPanel.isVisible(),
      // Grayed + non-clickable until the scene actually has layers (nodes with
      // layer=true). Refreshed on the 'luxar-layers-changed' event after load.
      disabled: () => layersPanel.layerState.count === 0,
    },
    {
      id: 'monitor',
      title: 'Data monitor',
      shortcut: shortcutForAction(KeyAction.cycleDataMonitor),
      icon: RAIL_ICONS.monitor,
      activate: () => {
        closeOthersOnCoarse('monitor');
        ui.commands.cycleDataMonitor();
      },
      openSelector: '.luxar-data-monitor',
    },
    {
      id: 'data',
      title: 'Datasets',
      shortcut: shortcutForAction(KeyAction.toggleDatasetBrowser),
      icon: RAIL_ICONS.data,
      activate: () => ui.commands.toggleDatasetBrowser(),
      openSelector: '.luxar-dataset-browser',
    },
    {
      // Sound: left-click toggles the mute, right-click opens the mixer. Hidden
      // (not grayed) on a scene without sound nodes — the layer is invisible
      // where it has nothing to say. Refreshed on 'luxar-audio-changed'.
      id: 'audio',
      title: 'Sound',
      icon: RAIL_ICONS.audio,
      hidden: () => !audioEngine.hasSoundNodes(),
      activate: () => audioEngine.setMuted(!audioEngine.isMuted()),
      isActive: () => audioEngine.isMuted(),
      render: (btn) => {
        const muted = audioEngine.isMuted();
        const key = muted ? 'muted' : 'on';
        if (btn.dataset.audioState === key) return;
        btn.dataset.audioState = key;
        const svg = btn.querySelector('svg');
        if (svg) svg.outerHTML = muted ? RAIL_ICONS.audioMuted : RAIL_ICONS.audio;
        btn.setAttribute(
          'aria-label',
          muted
            ? 'Sound: muted — click to unmute, right-click or hold for the mixer'
            : 'Sound: on — click to mute, right-click or hold for the mixer'
        );
        const tip = btn.querySelector('.luxar-control-rail__tip');
        if (tip) tip.textContent = muted ? 'Sound · Muted' : 'Sound · On';
      },
      popover: {
        trigger: 'context',
        title: 'Sound',
        build: (host) => {
          closeOtherLeftPanels();
          return buildAudioPopover(host, {
            getState: () => audioEngine.getState(),
            setAudio: (patch) => audioEngine.setAudio(patch),
          });
        },
      },
    },
    {
      id: 'recording',
      title: 'Recording',
      shortcut: shortcutForAction(KeyAction.toggleRecording),
      icon: RAIL_ICONS.recording,
      activate: () => {
        if (!recordingPanel.isVisible()) closeOtherLeftPanels('recording');
        ui.panels.getRecordingPanel()?.toggle();
      },
      isActive: () => recordingPanel.isVisible(),
      separatorBefore: true,
    },
    {
      id: 'logs',
      title: 'Logs (console)',
      shortcut: shortcutForAction(KeyAction.toggleDebugConsole),
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
          shortcut: shortcutForAction(KeyAction.toggleScaleBar),
          icon: RAIL_ICONS.scalebar,
          activate: () => ui.panels.getScaleBar()?.toggle(),
          openSelector: '.luxar-scale-bar',
        },
        {
          id: 'legend',
          title: 'Colormap legend',
          shortcut: shortcutForAction(KeyAction.toggleColormapLegend),
          icon: RAIL_ICONS.legend,
          activate: () => ui.panels.getColormapLegend()?.toggle(),
          openSelector: '.luxar-colormap-legend',
        },
        {
          id: 'overlays',
          title: 'Overlays',
          shortcut: shortcutForAction(KeyAction.toggleOverlays),
          icon: RAIL_ICONS.overlays,
          activate: () => ui.panels.getOverlayManager()?.toggle(),
          openSelector: '.luxar-overlay:not(.luxar-overlay--hidden)',
        },
        {
          id: 'cinematic',
          title: 'Cinematic mode',
          shortcut: shortcutForAction(KeyAction.toggleCinematicMode),
          icon: RAIL_ICONS.cinematic,
          activate: () => ui.commands.toggleCinematicMode(),
          isActive: () => renderingControls.settings.cinematicMode,
        },
        ...(fullscreenAvailable
          ? [
              {
                // Fullscreen is otherwise reachable only through the focus-gated
                // Space shortcut (dead whenever focus sits in a panel control), so
                // this chip is the discoverable affordance. Active-state tracks the
                // live fullscreen element; the rail refreshes on fullscreenchange
                // (see ControlRail.syncFullscreen), which also covers exits via
                // Escape or browser UI. Omitted where the API is absent (iPhone).
                id: 'fullscreen',
                title: 'Fullscreen',
                shortcut: shortcutForAction(KeyAction.toggleFullscreen),
                icon: RAIL_ICONS.fullscreen,
                activate: () => ui.commands.toggleFullscreen(),
                isActive: () => isDocumentFullscreen(),
                // Fullscreen is a session-long ambient state — lighting the View
                // button for the whole session would read as noise (and the rail
                // is hidden in fullscreen anyway). The chip itself still shows
                // active inside the flyout.
                excludeFromParentActive: true,
              },
            ]
          : []),
      ],
    },
    ...(coarse
      ? [
          {
            // Touch only: closes every open panel and popover, including while
            // fullscreen (where keyboard Escape is reserved for the browser).
            id: 'hide-panels',
            title: 'Hide panels',
            icon: RAIL_ICONS.hidePanels,
            momentary: true,
            activate: () => ui.commands.closeAllPanels(),
          },
        ]
      : []),
    {
      // Viewer-wide preferences (theme + persisted user settings) — kept off
      // the top level so the rail stays focused. Click opens the popover.
      id: 'settings',
      title: 'Settings',
      icon: RAIL_ICONS.settings,
      activate: () => {}, // unused — opens the popover below
      popover: {
        trigger: 'click',
        title: 'Settings',
        build: (host) => {
          closeOtherLeftPanels();
          return buildSettingsPopover(host, {
            triggerAnimation: () => animationController.startAnimation(),
            // Lazy — the loader exists only after the first scene load.
            getSceneLoader: () => getSceneLoader(),
          });
        },
      },
    },
    {
      // The gauge toggles the perf readout docked below (rail footer). Placed
      // just under the eye so the readout appears at the very bottom of the rail.
      // Right-click opens the adaptive-resolution (DPR) controls popover.
      id: 'perf',
      title: 'Performance',
      shortcut: shortcutForAction(KeyAction.togglePerformance),
      icon: RAIL_ICONS.perf,
      activate: () => ui.commands.togglePerformanceStats(),
      isActive: () => performanceMonitor.visible,
      popover: {
        trigger: 'context',
        title: 'Performance',
        build: (host) => {
          closeOtherLeftPanels();
          return buildPerformancePopover(host, {
            settings: renderingControls.settings,
            manager: adaptiveDPRManager,
            densityGuard,
            saveSettings: () => renderingControls.saveSettings(),
            triggerAnimation: () => animationController.startAnimation(),
          });
        },
      },
    },
  ];
}
