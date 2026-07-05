/**
 * Keyboard binding registration concern extracted from
 * `input/input-handler.ts`. The entry point + types live here; the
 * per-context binding bodies live as siblings (`fov-hold-gate.ts`,
 * `navigation-bindings.ts`, `fly-bindings.ts`).
 *
 * The function takes everything it needs as a deps object.
 * Optional panels (scale bar, colormap legend, overlay manager,
 * recording panel, layers panel) are passed as getter functions so
 * the bindings always read the live reference at dispatch time —
 * the InputHandler may wire them in after this function runs.
 *
 * @module input/input-handler/key-bindings/register-all
 */

import type { InputContextManager } from '../context-manager';
import type { SceneManager } from '../../../scene/scene-manager';
import type { ScaleBar } from '../../../ui/scale-bar';
import type { ColormapLegend } from '../../../ui/colormap-legend';
import type { OverlayManager } from '../../../ui/overlay-manager';
import type { RecordingPanel } from '../../../ui/recording-panel';
import type { LayersPanel } from '../../../ui/layers';
import type { DebugConsole } from '../../../ui/debug-console';
import { registerFovHoldGate } from './fov-hold-gate';
import { registerNavigationBindings } from './navigation-bindings';
import { registerFlyControlBindings } from './fly-bindings';

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
  /** Switch directly to a specific camera control mode (rail Navigation popover). */
  setControlMode(type: 'orbit' | 'fly' | 'ortho'): void;
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
