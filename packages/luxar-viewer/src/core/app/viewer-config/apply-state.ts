/**
 * Pure dispatch helper for applying `ZarrViewerConfig` state that lives
 * outside of the rendering-controls surface (panel visibility, theme,
 * dimension navigation). RenderingControls handles the rendering settings
 * on its own; this helper is for the rest.
 *
 * Extracted from `core/app.ts::applyViewerConfigState` so the dispatch
 * logic can be unit-tested without spinning up a full LuxarApp. The
 * caller (LuxarApp.init / LuxarApp.loadDataset) builds a `ViewerConfigPorts`
 * from its component fields and passes it through; this helper does
 * not reach out for any singletons it can't get from the ports.
 *
 * Only explicitly-set fields (`true` / `false` / non-undefined) are
 * applied — unset fields preserve the viewer's built-in defaults.
 *
 * @module core/app/viewer-config/apply-state
 */

import type { ZarrViewerConfig } from '../../../types/zarr';

/**
 * The surface this dispatcher needs from the parent LuxarApp. Each
 * field is optional because some panels are only built when their
 * underlying data is present (scaleBar requires dimension metadata,
 * overlays / layers require zarr `overlay_groups` / layer arrays, etc.).
 */
export interface ViewerConfigPorts {
  showHelp: () => void;
  renderingControls: { show: () => void; hide: () => void };
  performanceMonitor: { show: () => void };
  inputHandler: { showDimensionSliders: () => void };
  scaleBar?: { show: () => void; hide: () => void };
  layersPanel?: { show: () => void; hide: () => void };
  overlayManager?: { show: () => void; hide: () => void };
  setTheme: (themeId: string) => void;
  setDimensionValue: (dim: number, value: number) => void;
}

/**
 * Apply the `viewer_config` state subset that isn't owned by
 * RenderingControls. Mutates the ports' targets; returns nothing.
 *
 * The original `app.ts` call site passes `viewerConfig | undefined`
 * directly because the zarr scene metadata may be absent — this
 * helper preserves that contract and short-circuits on `undefined`.
 */
export function applyViewerConfigState(
  viewerConfig: ZarrViewerConfig | undefined,
  ports: ViewerConfigPorts
): void {
  if (!viewerConfig) return;

  // --- UI panel visibility ---
  const ui = viewerConfig.ui;
  if (ui) {
    if (ui.show_help === true) ports.showHelp();
    if (ui.show_rendering_controls === true) ports.renderingControls.show();
    if (ui.show_rendering_controls === false) ports.renderingControls.hide();
    if (ui.show_performance_monitor === true) ports.performanceMonitor.show();
    if (ui.show_dimensions === true) ports.inputHandler.showDimensionSliders();
    if (ui.show_scale_bar === true && ports.scaleBar) ports.scaleBar.show();
    if (ui.show_scale_bar === false && ports.scaleBar) ports.scaleBar.hide();
    if (ui.show_layers === true && ports.layersPanel) ports.layersPanel.show();
    if (ui.show_layers === false && ports.layersPanel) ports.layersPanel.hide();
    if (ui.show_overlays === true && ports.overlayManager) ports.overlayManager.show();
    if (ui.show_overlays === false && ports.overlayManager) ports.overlayManager.hide();
  }

  // --- Theme ---
  if (viewerConfig.theme) {
    ports.setTheme(viewerConfig.theme);
  }

  // --- Dimension navigation state ---
  if (viewerConfig.dimensions?.current_step) {
    for (let i = 0; i < viewerConfig.dimensions.current_step.length; i++) {
      ports.setDimensionValue(i, viewerConfig.dimensions.current_step[i]);
    }
  }
}
