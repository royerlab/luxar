/**
 * Performance / adaptive-DPR setup for the rendering-controls panel.
 *
 * Builds the "Performance" folder with the Allow High DPR toggle, the
 * Adaptive Resolution toggle, a Manual DPR slider (visible when adaptive
 * is OFF), and read-only Current DPR / Current FPS display rows (visible
 * when adaptive is ON).
 *
 * Allow High DPR shows in BOTH modes, because it caps both: it is the
 * adaptive scale-up ceiling and the top of the manual slider's range.
 *
 * Returns:
 *  - the controllers for `adaptiveDPREnabled` and `allowHighDPR` so the
 *    facade can store them,
 *  - a visibility callback used by the persistence layer to re-sync
 *    the panel after settings load,
 *  - a cleanup function for the FPS-display polling interval.
 */

import type { Controller } from '../../gui';
import type GUI from '../../gui';
import type { RenderingSettings } from '../../../config';
import { FOLDER_ICONS } from '../folder-icons';
import type { AdaptiveDPRManager } from '../../../rendering/adaptive-dpr-manager';
import { getMaxPixelRatio } from '../../../rendering/pixel-ratio-cap';
import { formatFPSReading } from '../../performance-monitor';
import { log, Modules } from '../../../utils/log';
import type { DensityGuardControl } from '../types';

/**
 * `min`/`max`/`step` live on lil-gui's NumberController but not on the
 * base `Controller` type, so a structural cast targets just those three
 * fluent methods (same pattern as the fly-speed slider in
 * `ui/rendering-controls.ts`).
 */
type ChainableNumber = {
  min(v: number): ChainableNumber;
  max(v: number): ChainableNumber;
  step(v: number): ChainableNumber;
};

export interface PerformanceSetupContext {
  gui: GUI;
  settings: RenderingSettings;
  manager: AdaptiveDPRManager;
  /**
   * Runtime handle on the projected-density guard. Optional: without it the
   * folder has no Density Guard toggle or Thinning row (embeds without the
   * guard wiring, and the older tests).
   */
  densityGuard?: DensityGuardControl;
  saveSettings: () => void;
  triggerAnimation: () => void;
}

export interface PerformanceSetupResult {
  /** The Adaptive Resolution toggle controller. */
  adaptiveDPREnabled: Controller;
  /** The Allow High DPR toggle controller. */
  allowHighDPR: Controller;
  /** The Density Guard toggle controller (absent without `densityGuard` in the context). */
  densityGuardEnabled?: Controller;
  /** Re-applied by `loadSettings` after a stored adaptiveDPREnabled flips. */
  updateVisibility: (adaptiveEnabled: boolean) => void;
  /** Cleanup callback for the periodic FPS/DPR display update interval. */
  cleanup: () => void;
}

/** Build a read-only "Current DPR" / "Current FPS" display row. */
function createDisplayRow(label: string, tooltip: string): HTMLElement {
  const row = document.createElement('div');
  row.className = 'luxar-gui__controller';
  row.style.opacity = '0.7';
  row.setAttribute('title', tooltip);

  const nameEl = document.createElement('div');
  nameEl.className = 'luxar-gui__controller-name';
  nameEl.textContent = label;

  const valueEl = document.createElement('div');
  valueEl.className = 'luxar-gui__controller-widget';
  valueEl.style.textAlign = 'right';
  valueEl.style.paddingRight = '8px';
  valueEl.style.fontFamily = 'monospace';

  row.appendChild(nameEl);
  row.appendChild(valueEl);
  return row;
}

/** `1/8` for 0.125; `1` for an unthinned node. */
export function formatKeepFraction(keep: number): string {
  return keep >= 1 ? '1' : `1/${Math.round(1 / keep)}`;
}

/** Text for the Thinning row: what the guard is doing to the frame right now. */
export function formatThinning(control: DensityGuardControl): string {
  if (!control.isEnabled()) return 'off';
  const { nodes, minKeep } = control.thinning();
  if (nodes === 0) return 'none';
  return `${nodes} node${nodes === 1 ? '' : 's'} · keep ${formatKeepFraction(minKeep)}`;
}

interface DensityGuardControls {
  toggle: Controller;
  row: HTMLElement;
  /** Re-read the live thinning state into the row. */
  refresh(): void;
}

/**
 * The Density Guard toggle + Thinning readout. The toggle is bound to the
 * guard's LIVE state, not the stored flag: under `?no-density-guard` the
 * stored flag may say on while the guard is off, and the box must not lie.
 * The stored flag is only written when the session is not URL-disabled.
 */
function addDensityGuardControls(
  folder: Pick<GUI, 'add'>,
  context: PerformanceSetupContext,
  control: DensityGuardControl
): DensityGuardControls {
  const { settings, saveSettings, triggerAnimation } = context;
  const state = { enabled: control.isEnabled() };
  const toggle = folder.add(state, 'enabled').name('Density Guard');
  toggle.domElement.setAttribute(
    'title',
    'Density Guard: Thins over-dense nodes instead of drawing every element\n' +
      '• Measures elements per pixel for each node every frame\n' +
      '• Above a few elements per pixel, draws a hashed subset and brightens\n' +
      '  the rest to match — the composited brightness stays the same\n' +
      '• Also holds back detail levels the current framing cannot resolve\n' +
      '  (they load as you zoom in)\n' +
      '• Only for additive / luminous / volumetric blending; max, normal and\n' +
      '  opaque nodes are never thinned\n' +
      '• OFF: every element is drawn (the ?no-density-guard URL flag does the\n' +
      '  same for one session)'
  );
  const row = createDisplayRow(
    'Thinning',
    'What the Density Guard is doing right now\n' +
      '• none: every visible node is under its density cap\n' +
      '• N nodes · keep 1/K: N nodes are drawn at a 1/K subset (the densest\n' +
      '  one at 1/K), each brightened ×K to compensate\n' +
      '• Zooming in restores detail step by step'
  );
  const value = row.querySelector('.luxar-gui__controller-widget') as HTMLElement;
  const refresh = (): void => {
    value.textContent = formatThinning(control);
  };
  toggle.onChange((enabled: boolean) => {
    control.setEnabled(enabled);
    if (!control.sessionDisabled) {
      settings.densityGuardEnabled = enabled;
      saveSettings();
    }
    log.info(Modules.RENDERER, `Density guard ${enabled ? 'enabled' : 'disabled'}`);
    refresh();
    triggerAnimation();
  });
  refresh();
  return { toggle, row, refresh };
}

export function setupPerformanceControls(context: PerformanceSetupContext): PerformanceSetupResult {
  const { gui, settings, manager, saveSettings, triggerAnimation } = context;

  const performanceFolder = gui.addFolder('Performance', FOLDER_ICONS.performance);

  performanceFolder.domElement?.setAttribute(
    'title',
    'Performance: Controls that trade visual quality for speed\n\n' +
      "• Allow High DPR: Let the viewer render at your display's full pixel\n" +
      '  density instead of CSS resolution.\n' +
      '• Adaptive Resolution: Automatically lowers pixel ratio when FPS drops,\n' +
      '  then gradually restores quality when the GPU catches up.\n' +
      '• Manual DPR: Set a fixed pixel ratio (lower = faster but blurrier).\n' +
      '• Density Guard: Thin over-dense nodes (brightness-compensated) instead\n' +
      '  of drawing every element on top of the same pixels.\n\n' +
      'Useful for large datasets or lower-end GPUs where smooth interaction\n' +
      'matters more than pixel-perfect sharpness.'
  );

  const highDPRToggle = performanceFolder.add(settings, 'allowHighDPR').name('Allow High DPR');

  highDPRToggle.domElement.setAttribute(
    'title',
    "Allow High DPR: Render at your display's full pixel density\n" +
      `• Your display: ${manager.getNativeDPR().toFixed(2)}x\n` +
      '• OFF (default): the viewer never renders above 1.0 — the starting\n' +
      '  resolution, the adaptive ceiling, and the top of the Manual DPR slider\n' +
      `• ON: allows up to ${manager.getNativeDPR().toFixed(2)}x, which costs ` +
      `${(manager.getNativeDPR() ** 2).toFixed(0)}x the pixels\n` +
      '• Off by default: points and splats are soft-edged, so the extra pixels\n' +
      '  usually cost far more than they show\n' +
      '• Worth turning ON for scenes made of THIN LINES (river networks,\n' +
      '  tractograms, wiring diagrams), where CSS resolution blurs the strands\n' +
      '  together — and where it is cheap, since line scenes are not fill-bound'
  );

  const adaptiveToggle = performanceFolder
    .add(settings, 'adaptiveDPREnabled')
    .name('Adaptive Resolution');

  adaptiveToggle.domElement.setAttribute(
    'title',
    'Adaptive Resolution: Automatically adjusts rendering quality for smooth FPS\n' +
      '• Reduces pixel ratio when FPS drops below ~75% of your display rate\n' +
      '• Gradually restores quality once FPS stabilizes above ~90% of it\n' +
      '• Every reduction is probe-verified (reverted if it did not help)\n' +
      '• Minimum DPR: 0.5 (50% of native resolution)'
  );

  // Density Guard sits between the adaptive toggle and the manual slider:
  // both toggles are "trade quality for speed" switches, the slider is a knob.
  const density = context.densityGuard
    ? addDensityGuardControls(performanceFolder, context, context.densityGuard)
    : null;

  const nativeDPR = manager.getNativeDPR();
  // The slider tops out at the CEILING, not the display's DPR: while
  // high DPR is disallowed the toggle above is a hard cap on everything
  // interactive, this control included. `retargetManualDPRRange` widens
  // and narrows it when that toggle flips with the popover open.
  const manualDPRSettings = { dpr: getMaxPixelRatio() };

  const manualDPRControl = performanceFolder
    .add(manualDPRSettings, 'dpr', 0.25, getMaxPixelRatio(), 0.05)
    .name('Manual DPR')
    .onFinishChange((value: number) => {
      if (!settings.adaptiveDPREnabled) {
        manager.setManualDPR(value);
        triggerAnimation();
      }
    });

  manualDPRControl.domElement.setAttribute(
    'title',
    'Manual Device Pixel Ratio (when adaptive is off)\n' +
      `• Display: ${nativeDPR.toFixed(2)} — enable Allow High DPR to go above 1.00\n` +
      '• Lower values = better performance, less sharpness\n' +
      '• Applied when you release/commit the slider to avoid GPU resize thrash\n' +
      '• 1.0 = 100% resolution, 0.5 = 50% resolution'
  );

  /**
   * Point the slider at the current ceiling. `min`/`max`/`step` are
   * NumberController-only, so reach them through the same structural
   * cast the fly-speed slider uses.
   */
  const retargetManualDPRRange = (): void => {
    const ceiling = getMaxPixelRatio();
    (manualDPRControl as unknown as ChainableNumber).max(ceiling);
    if (manualDPRSettings.dpr > ceiling) manualDPRSettings.dpr = ceiling;
    manualDPRControl.updateDisplay();
  };

  const dprRow = createDisplayRow(
    'Current DPR',
    `Current Device Pixel Ratio\n• Display: ${nativeDPR.toFixed(2)}\n• Lower values = better performance, less sharpness`
  );
  const dprValue = dprRow.querySelector('.luxar-gui__controller-widget') as HTMLElement;

  const fpsRow = createDisplayRow(
    'Current FPS',
    'Current Frames Per Second\n' +
      '• Target: ~90% of your display refresh rate\n' +
      '• Scales down below ~75% of it'
  );
  const fpsValue = fpsRow.querySelector('.luxar-gui__controller-widget') as HTMLElement;

  const folderEl = performanceFolder.domElement;
  if (folderEl) {
    const childrenContainer = folderEl.querySelector('.luxar-gui__children');
    if (childrenContainer) {
      childrenContainer.appendChild(dprRow);
      childrenContainer.appendChild(fpsRow);
      if (density) childrenContainer.appendChild(density.row);
    }
  }

  // The manager clears its FPS window when the animation loop pauses
  // (notifyPaused), so currentFPS === 0 is an unambiguous "not
  // rendering" sentinel — display it as such instead of a frozen
  // last-window number pretending to be live.
  //
  // Everything above 0 is formatted by the SHARED reading formatter the
  // rail gauge uses, so the two readouts can never disagree about the
  // same rate: sub-1fps rates are real (the FPS window keeps a
  // two-sample minimum, so a software-rasterized scene reports 0.4fps
  // rather than 0) and must not round to the "0" that means the opposite.
  const formatFPS = (fps: number): string => (fps > 0 ? formatFPSReading(fps) : 'idle');

  const updateVisibility = (adaptiveEnabled: boolean): void => {
    if (adaptiveEnabled) {
      manualDPRControl.hide();
      dprRow.style.display = '';
      fpsRow.style.display = '';
      const state = manager.getState();
      dprValue.textContent = state.currentDPR.toFixed(2);
      fpsValue.textContent = formatFPS(state.currentFPS);
    } else {
      manualDPRControl.show();
      retargetManualDPRRange();
      manualDPRSettings.dpr = manager.getCurrentDPR() ?? getMaxPixelRatio();
      manualDPRControl.updateDisplay();
      dprRow.style.display = 'none';
      fpsRow.style.display = 'none';
    }
  };

  // Sync initial state from manager BEFORE setting visibility. A URL pin
  // is session-only, so keep the stored scene settings untouched while
  // pinned rather than letting the effective override leak into the next
  // saveSettings() call.
  if (!manager.isPinned()) {
    settings.adaptiveDPREnabled = manager.isActive();
    settings.allowHighDPR = manager.isHighDPRAllowed();
  }
  adaptiveToggle.updateDisplay();
  highDPRToggle.updateDisplay();
  updateVisibility(settings.adaptiveDPREnabled);

  highDPRToggle.onChange((allowed: boolean) => {
    manager.setHighDPRAllowed(allowed);
    saveSettings();
    log.info(Modules.RENDERER, `High DPR ${allowed ? 'allowed' : 'disallowed'}`);
    // The ceiling moved, so the manual slider's range and the readouts
    // must follow it in the same tick.
    updateVisibility(settings.adaptiveDPREnabled);
    triggerAnimation();
  });

  adaptiveToggle.onChange((enabled: boolean) => {
    manager.setEnabled(enabled);
    saveSettings();
    log.info(Modules.RENDERER, `Adaptive resolution ${enabled ? 'enabled' : 'disabled'}`);
    updateVisibility(enabled);
  });

  // Periodic display refresh — the DPR/FPS rows only while adaptive is
  // enabled (they are hidden otherwise); the Thinning row always.
  const updateInterval = setInterval(() => {
    if (settings.adaptiveDPREnabled) {
      const state = manager.getState();
      dprValue.textContent = state.currentDPR.toFixed(2);
      fpsValue.textContent = formatFPS(state.currentFPS);
    }
    density?.refresh();
  }, 500);

  // Open by default: the Performance rail popover shows these controls directly.
  performanceFolder.open();

  return {
    adaptiveDPREnabled: adaptiveToggle,
    allowHighDPR: highDPRToggle,
    densityGuardEnabled: density?.toggle,
    updateVisibility,
    cleanup: () => clearInterval(updateInterval),
  };
}
