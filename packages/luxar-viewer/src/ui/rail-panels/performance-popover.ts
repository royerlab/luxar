/**
 * Performance rail popover — adaptive-resolution (DPR) controls.
 *
 * Reached by right-clicking the Performance gauge in the rail (left-click still
 * toggles the docked FPS readout). Hosts the Adaptive Resolution toggle, the
 * Manual DPR slider, and the live DPR/FPS readout rows — moved here from the
 * Rendering Controls panel (performance is not a rendering concern).
 *
 * Rebuilt on each open, so it self-syncs from the AdaptiveDPRManager; the
 * teardown clears the setup's periodic refresh interval.
 *
 * @module ui/rail-panels/performance-popover
 */

import type { RenderingSettings } from '../../config';
import type { AdaptiveDPRManager } from '../../rendering/adaptive-dpr-manager';
import { setupPerformanceControls } from '../rendering-controls/setup/performance-setup';
import type { DensityGuardControl } from '../rendering-controls/types';
import { makePopoverGui } from './popover-gui';

export interface PerformancePopoverContext {
  /** Shared rendering settings (persisted by the rendering-controls panel). */
  settings: RenderingSettings;
  manager: AdaptiveDPRManager;
  /** Runtime handle on the projected-density guard (Density Guard toggle + thinning readout). */
  densityGuard?: DensityGuardControl;
  /** Persist the shared settings to localStorage. */
  saveSettings: () => void;
  /** Request a render so DPR changes take effect immediately. */
  triggerAnimation: () => void;
}

/**
 * Build the Performance popover into `host`. Returns a teardown that clears the
 * FPS-refresh interval and disposes the GUI when the popover closes.
 */
export function buildPerformancePopover(
  host: HTMLElement,
  ctx: PerformancePopoverContext
): () => void {
  const gui = makePopoverGui(host, 'Performance');
  const result = setupPerformanceControls({
    gui,
    settings: ctx.settings,
    manager: ctx.manager,
    densityGuard: ctx.densityGuard,
    saveSettings: ctx.saveSettings,
    triggerAnimation: ctx.triggerAnimation,
  });
  return () => {
    result.cleanup();
    gui.destroy();
  };
}
