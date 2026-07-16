import { ScaleBar } from '../../../ui/scale-bar';
import { config } from '../../../config';
import type { SceneManager } from '../../../scene/scene-manager';
import type { AnimationController } from '../../../scene/animation/animation-controller';
import type { InputHandler } from '../../../input/input-handler';

/**
 * Build a fresh {@link ScaleBar}, register its per-frame update on the
 * animation controller, and wire it through the input handler so the
 * keyboard toggle keeps working. Disposes the previous instance + its
 * per-frame callback when present.
 *
 * Returns the new scale bar — the caller stores it on its own field
 * (also captured by the per-frame closure via the same reference).
 */
export interface InitScaleBarPorts {
  previous: ScaleBar | undefined;
  sceneManager: SceneManager;
  animationController: AnimationController;
  inputHandler: InputHandler;
}

export function initScaleBar(ports: InitScaleBarPorts): ScaleBar {
  // Dispose previous instance if reloading
  if (ports.previous) {
    ports.animationController.removePerFrameCallback('scale-bar');
    ports.previous.dispose();
  }

  const scaleBar = new ScaleBar({
    // Live accessor: the scene manager replaces the camera on
    // perspective ↔ ortho swaps, so the scale bar must re-read it per
    // update (see ScaleBarConfig.getCamera).
    getCamera: () => ports.sceneManager.camera,
    controls: ports.sceneManager.controls,
    canvas: ports.sceneManager.renderer.domElement,
    targetWidthPx: config.ui.scaleBar.targetWidthPx,
    position: config.ui.scaleBar.position,
  });

  // Register per-frame update for live camera tracking
  ports.animationController.addPerFrameCallback('scale-bar', () => {
    scaleBar.update();
  });

  // Wire to input handler for keyboard toggle
  ports.inputHandler.setScaleBar(scaleBar);
  return scaleBar;
}
