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
  /** Set the browser tab title (document.title) from the scene's title. */
  setDocumentTitle: (title: string) => void;
  /**
   * Start playback on one dimension. Absent when the scene has no animatable
   * dimension (a purely 3D scene never builds an animation manager), which is
   * why the caller must tolerate it being undefined rather than assume a
   * no-op stub exists.
   */
  startDimensionAnimation?: (
    dim: number,
    options: {
      targetFPS?: number;
      loopMode?: string;
      direction?: string;
      /** Per-tick advance in the dimension's units; omitted means Auto. */
      stepSize?: number;
    }
  ) => void;
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

  // --- Browser tab title ---
  // Authored scene identity wins over the ?title= URL fallback applied at
  // bootstrap, so a tab always names the scene it actually shows. The
  // typeof guard is load-bearing: `viewerConfig` is untyped JSON straight
  // out of the scene's zarr attributes, and a non-string `title` would
  // otherwise throw here and abort the rest of the load (theme, dimension
  // state, and the render loop kicked off after this call).
  if (typeof viewerConfig.title === 'string' && viewerConfig.title.trim()) {
    ports.setDocumentTitle(viewerConfig.title.trim());
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

  // --- Animation state ---
  //
  // The `animation` block round-tripped through the scene file for a long time
  // without anything reading it back: `viewer-state-capture` wrote it on
  // Ctrl+Shift+S, the Python `ViewerConfig` exposed it, VIEWER_GUIDE.md
  // described it as restored, and on load it was silently dropped. A scene
  // could therefore say "open playing" in every representation except the one
  // that mattered.
  //
  // Applied AFTER `current_step` on purpose: playback starts from wherever the
  // dimension was left, so a scene that authors both opens at its chosen
  // timepoint and runs on from there rather than snapping back to the start.
  //
  // Only `playing === true` does anything. `false` is the viewer's own default
  // and re-asserting it would mean a captured-then-paused scene could never
  // simply inherit whatever the viewer does next.
  if (viewerConfig.animation && ports.startDimensionAnimation) {
    for (let dim = 0; dim < viewerConfig.animation.length; dim++) {
      const entry = viewerConfig.animation[dim];
      if (!entry || entry.playing !== true) continue;
      ports.startDimensionAnimation(dim, {
        targetFPS: entry.target_fps,
        loopMode: entry.loop,
        direction: entry.direction,
        stepSize: entry.step_size,
      });
    }
  }
}
