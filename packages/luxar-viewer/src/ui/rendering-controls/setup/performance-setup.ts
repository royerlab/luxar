/**
 * Performance / adaptive-DPR setup for the rendering-controls panel.
 *
 * Builds the "⚡ Performance" folder with the Adaptive Resolution toggle,
 * a Manual DPR slider (visible when adaptive is OFF), and read-only
 * Current DPR / Current FPS display rows (visible when adaptive is ON).
 *
 * Returns:
 *  - the controller for `adaptiveDPREnabled` so the facade can store it,
 *  - a visibility callback used by the persistence layer to re-sync
 *    the panel after settings load,
 *  - a cleanup function for the FPS-display polling interval.
 */

import type { Controller } from '../../gui';
import type GUI from '../../gui';
import type { RenderingSettings } from '../../../config';
import { FOLDER_ICONS } from '../folder-icons';
import type { AdaptiveDPRManager } from '../../../rendering/adaptive-dpr-manager';
import { log, Modules } from '../../../utils/log';

export interface PerformanceSetupContext {
  gui: GUI;
  settings: RenderingSettings;
  manager: AdaptiveDPRManager;
  saveSettings: () => void;
  triggerAnimation: () => void;
}

export interface PerformanceSetupResult {
  /** The Adaptive Resolution toggle controller. */
  adaptiveDPREnabled: Controller;
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

export function setupPerformanceControls(context: PerformanceSetupContext): PerformanceSetupResult {
  const { gui, settings, manager, saveSettings, triggerAnimation } = context;

  const performanceFolder = gui.addFolder('Performance', FOLDER_ICONS.performance);

  performanceFolder.domElement?.setAttribute(
    'title',
    'Performance: Controls that trade visual quality for speed\n\n' +
      '• Adaptive Resolution: Automatically lowers pixel ratio when FPS drops,\n' +
      '  then gradually restores quality when the GPU catches up.\n' +
      '• Manual DPR: Set a fixed pixel ratio (lower = faster but blurrier).\n\n' +
      'Useful for large datasets or lower-end GPUs where smooth interaction\n' +
      'matters more than pixel-perfect sharpness.'
  );

  const adaptiveToggle = performanceFolder
    .add(settings, 'adaptiveDPREnabled')
    .name('Adaptive Resolution');

  adaptiveToggle.domElement.setAttribute(
    'title',
    'Adaptive Resolution: Automatically adjusts rendering quality for smooth FPS\n' +
      '• When FPS drops below 50, reduces pixel ratio\n' +
      '• Gradually restores quality when FPS stabilizes above 58\n' +
      '• Minimum DPR: 0.5 (50% of native resolution)'
  );

  const nativeDPR = manager.getNativeDPR();
  const manualDPRSettings = { dpr: nativeDPR };

  const manualDPRControl = performanceFolder
    .add(manualDPRSettings, 'dpr', 0.25, nativeDPR, 0.05)
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
      `• Native: ${nativeDPR.toFixed(2)}\n` +
      '• Lower values = better performance, less sharpness\n' +
      '• Applied when you release/commit the slider to avoid GPU resize thrash\n' +
      '• 1.0 = 100% resolution, 0.5 = 50% resolution'
  );

  const dprRow = createDisplayRow(
    'Current DPR',
    `Current Device Pixel Ratio\n• Native: ${nativeDPR.toFixed(2)}\n• Lower values = better performance, less sharpness`
  );
  const dprValue = dprRow.querySelector('.luxar-gui__controller-widget') as HTMLElement;

  const fpsRow = createDisplayRow(
    'Current FPS',
    'Current Frames Per Second\n• Target: 55-60 FPS\n• Scales down if below 50 FPS'
  );
  const fpsValue = fpsRow.querySelector('.luxar-gui__controller-widget') as HTMLElement;

  const folderEl = performanceFolder.domElement;
  if (folderEl) {
    const childrenContainer = folderEl.querySelector('.luxar-gui__children');
    if (childrenContainer) {
      childrenContainer.appendChild(dprRow);
      childrenContainer.appendChild(fpsRow);
    }
  }

  const updateVisibility = (adaptiveEnabled: boolean): void => {
    if (adaptiveEnabled) {
      manualDPRControl.hide();
      dprRow.style.display = '';
      fpsRow.style.display = '';
      const state = manager.getState();
      dprValue.textContent = state.currentDPR.toFixed(2);
      fpsValue.textContent = Math.round(state.currentFPS).toString();
    } else {
      manualDPRControl.show();
      manualDPRSettings.dpr = manager.getCurrentDPR() ?? nativeDPR;
      manualDPRControl.updateDisplay();
      dprRow.style.display = 'none';
      fpsRow.style.display = 'none';
    }
  };

  // Sync initial state from manager BEFORE setting visibility.
  settings.adaptiveDPREnabled = manager.isActive();
  adaptiveToggle.updateDisplay();
  updateVisibility(settings.adaptiveDPREnabled);

  adaptiveToggle.onChange((enabled: boolean) => {
    manager.setEnabled(enabled);
    saveSettings();
    log.info(Modules.RENDERER, `Adaptive resolution ${enabled ? 'enabled' : 'disabled'}`);
    updateVisibility(enabled);
  });

  // Periodic display refresh — only updates when adaptive is enabled.
  const updateInterval = setInterval(() => {
    if (settings.adaptiveDPREnabled) {
      const state = manager.getState();
      dprValue.textContent = state.currentDPR.toFixed(2);
      fpsValue.textContent = Math.round(state.currentFPS).toString();
    }
  }, 500);

  performanceFolder.close();

  return {
    adaptiveDPREnabled: adaptiveToggle,
    updateVisibility,
    cleanup: () => clearInterval(updateInterval),
  };
}
