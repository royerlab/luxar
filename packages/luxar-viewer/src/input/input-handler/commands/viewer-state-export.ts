/**
 * Capture the full viewer state to JSON and copy to clipboard. Triggered
 * by Ctrl+Shift+S in the live keybindings; loadable in Python with
 * `luxar.ViewerConfig.from_json()`. Body extracted from
 * input-handler.ts.
 *
 * @module input/input-handler/commands/viewer-state-export
 */

import { notifier } from '../../../utils/cross-layer/notifier';
import { log, Modules } from '../../../utils/log';
import { captureViewerState } from '../../../config/zarr-bridge/viewer-state-capture';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import type { SceneManager } from '../../../scene/scene-manager';
import type { RenderingControlsHandle } from '../panel-capabilities';
import type { DimensionAnimationManager } from '../../../scene/animation/dimension-animation-manager';

export interface ViewerStateExportCtx {
  sceneManager: SceneManager;
  renderingControls: RenderingControlsHandle | undefined;
  animationManager: DimensionAnimationManager | undefined;
}

export function exportViewerState(ctx: ViewerStateExportCtx): void {
  if (!ctx.renderingControls) {
    log.warning(Modules.INPUT, 'Cannot export state: rendering controls not available');
    return;
  }

  const state = captureViewerState(
    ctx.sceneManager,
    ctx.renderingControls,
    sceneDimsManager,
    ctx.animationManager
  );

  const json = JSON.stringify(state, null, 2);

  // Copy to clipboard
  navigator.clipboard
    .writeText(json)
    .then(() => {
      notifier.toast('Viewer state copied to clipboard');
      log.info(Modules.INPUT, 'Viewer state exported to clipboard');
    })
    .catch((err) => {
      log.error(Modules.INPUT, 'Failed to copy state to clipboard:', err);
      notifier.toast('Failed to copy state to clipboard');
    });

  // Also store on debug interface for programmatic access
  if (window.__luxarDebug) {
    window.__luxarDebug.lastExportedState = state;
  }
}
