/**
 * Post-processing-pipeline construction extracted from
 * SceneManager. Thin factory over `PostProcessingManager`'s
 * constructor that wires the resize callback through to the
 * caller's material-update hook.
 *
 * @module scene/scene-manager/render-pipeline/post-processing-setup
 */

import * as THREE from 'three';
import { PostProcessingManager } from '../../../rendering';
import type { Renderer, RendererCapabilities } from '../../../rendering/renderer-capabilities';
import { type LuxarCamera, updateCameraAspect } from '../../../utils/camera-utils';
import { log, Modules } from '../../../utils/log';

/** Options for `createPostProcessing`. */
export interface CreatePostProcessingOptions {
  renderer: Renderer;
  capabilities: RendererCapabilities;
  scene: THREE.Scene;
  camera: LuxarCamera;
  /**
   * Callback fired whenever the manager reallocates its
   * render-target pyramid (window resize, SSAA toggle, MSAA
   * toggle, DPR change). The factory restores the camera projection
   * from the logical display size before invoking this hook; hosts use
   * it to refresh material uniforms that cache
   * `renderer.getDrawingBufferSize()`.
   */
  onResize: () => void;
}

/**
 * Construct the HDR post-processing pipeline.
 *
 * Reads canvas dimensions from the renderer's domElement at
 * construction time and seeds the PostProcessingManager's
 * render-target pyramid with the initial size.
 */
export function createPostProcessing(options: CreatePostProcessingOptions): PostProcessingManager {
  const canvas = options.renderer.domElement;
  const width = canvas.clientWidth || window.innerWidth;
  const height = canvas.clientHeight || window.innerHeight;

  const postProcessing = new PostProcessingManager(
    options.renderer,
    options.capabilities,
    options.scene,
    options.camera,
    { width, height },
    (displaySize, camera) => {
      updateCameraAspect(camera, displaySize.width, displaySize.height);
      options.onResize();
    }
  );

  log.success(Modules.POST_PROCESSING, 'HDR pipeline initialized');
  return postProcessing;
}
