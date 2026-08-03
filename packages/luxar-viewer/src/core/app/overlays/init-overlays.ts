import * as THREE from 'three';
import { OverlayManager } from '../../../ui/overlay-manager';
import type { SceneManager } from '../../../scene/scene-manager';
import type { InputHandler } from '../../../input/input-handler';
import type { RecordingPanel } from '../../../ui/recording-panel';

/**
 * Build a fresh {@link OverlayManager}, optionally hydrate it from
 * zarr overlay metadata on the LuxarScene root, then wire it to the
 * input handler and (when present) recording panel. Idempotent — the
 * caller's {@link InitOverlaysPorts.disposePrevious} runs first so a
 * second call from another path won't leak the previous instance.
 *
 * Returns the new manager. The caller stores it on its own field.
 */
export interface InitOverlaysPorts {
  disposePrevious: () => void;
  sceneManager: SceneManager;
  inputHandler: InputHandler;
  recordingPanel: RecordingPanel | undefined;
}

export async function initOverlays(ports: InitOverlaysPorts): Promise<OverlayManager> {
  // Defensive: loadDataset() already disposes overlays upfront, but keep
  // this idempotent in case initOverlays() is called from another path.
  ports.disposePrevious();

  const root = ports.sceneManager.scene?.children?.find((c) => c.name === 'LuxarScene') as
    THREE.Group | undefined;

  const overlayConfigs = root?.userData?.overlayConfigs;
  const zarrBaseUrl = root?.userData?.zarrBaseUrl;

  const manager = new OverlayManager();
  if (overlayConfigs?.length > 0 && zarrBaseUrl) {
    await manager.loadOverlays(overlayConfigs, zarrBaseUrl);
  }
  ports.inputHandler.setOverlayManager(manager);
  ports.recordingPanel?.setOverlayManager(manager);
  return manager;
}
