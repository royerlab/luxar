/**
 * Settings rail popover — viewer-wide preferences reached from the gear icon.
 *
 * Currently hosts the theme picker (moved out of the Rendering Controls panel,
 * where it never belonged). Kept as a container so future viewer-wide prefs
 * slot in as sibling folders without adding more top-level rail buttons.
 *
 * @module ui/rail-panels/settings-popover
 */

import { setupThemeControls } from '../rendering-controls/setup/theme-setup';
import { makePopoverGui } from './popover-gui';

export interface SettingsPopoverContext {
  /** Re-render after a theme change so the new tokens paint immediately. */
  triggerAnimation: () => void;
}

/**
 * Build the Settings popover into `host`. Returns a teardown that disposes the
 * GUI when the popover closes.
 */
export function buildSettingsPopover(host: HTMLElement, ctx: SettingsPopoverContext): () => void {
  const gui = makePopoverGui(host, 'Settings');
  setupThemeControls({ gui, triggerAnimation: ctx.triggerAnimation });
  return () => gui.destroy();
}
