/**
 * WebGL context-loss / context-restoration concern extracted from
 * `scene/scene-manager.ts`.
 *
 * Owns the full lifecycle of a graceful WebGL recovery:
 *
 *   - **Loss**: a `webglcontextlost` event fires when the GPU driver
 *     crashes, the system hibernates, the browser hits its WebGL
 *     context limit, or GPU memory runs out. We `preventDefault` (the
 *     spec requires this for the browser to even attempt restoration),
 *     flip an `isContextLost` flag, and surface an error toast.
 *
 *   - **Restoration**: when `webglcontextrestored` fires, we walk a
 *     deterministic rebuild order — renderer.resetState →
 *     postProcessing.rebuildAfterContextRestore →
 *     materialManager.rebuildAfterContextRestore →
 *     mark-scene-resources-dirty → updateRendererSize. Then we
 *     dispatch `webgl-context-restored` so subscribers (SceneLoader,
 *     NodeFactory, picking) can re-register their per-context state,
 *     and `change` to trigger a render.
 *
 * The class is a `THREE.EventDispatcher` so it stays compatible with
 * the existing SceneManager pattern of dispatching `webgl-context-restored`
 * on its own surface — the SceneManager forwards the event after
 * receiving it from the recovery instance.
 *
 * Behavior is identical to the inline original: same guard messages,
 * same rebuild order, same dispatch-events-after-restore sequence.
 *
 * @module scene/scene-manager/render-pipeline/webgl-context-recovery
 */

import * as THREE from 'three';
import { log, Modules } from '../../../utils/log';
import { notifier } from '../../../utils/cross-layer/notifier';
import { materialManager } from '../../../rendering/material-manager';
import type { PostProcessingManager } from '../../../rendering';

/**
 * Side-effects the recovery class needs from the host scene. Passed
 * once at construction; the recovery class never reads SceneManager
 * fields directly, only this surface.
 */
export interface WebGLContextRecoveryDeps {
  canvas: HTMLCanvasElement;
  /**
   * Lookup function — the SceneManager calls
   * `setupContextLossHandling()` BEFORE `setupScene()`, so capturing
   * `this.scene` at construction would close over `undefined`. The
   * getter is invoked at restore time, by which point the scene is
   * fully initialized.
   */
  getScene(): THREE.Scene;
  renderer: THREE.WebGLRenderer;
  /**
   * Lookup function rather than a direct reference — the post-processing
   * manager IS preserved across rebuilds (its identity is stable so
   * downstream caches stay valid), but the host SceneManager initializes
   * it asynchronously after `setupContextLossHandling` runs. The
   * function returns `null` until init completes.
   */
  getPostProcessing(): PostProcessingManager | null;
  /**
   * Resize the renderer / DPR after restoration; identical to the
   * SceneManager's `updateRendererSize()` private. Wired in via this
   * callback rather than being re-implemented here because it touches
   * SceneManager-only resize-debouncing state.
   */
  updateRendererSize(): void;
  /**
   * Notify subscribers (SceneLoader, NodeFactory, etc.) that the
   * context has been restored. The SceneManager forwards this onto
   * its own `webgl-context-restored` event — the recovery instance
   * doesn't need to know about EventDispatcher subscribers itself.
   */
  onContextRestored(): void;
  /**
   * Notify the SceneManager to dispatch a `change` event so a frame
   * renders. Same forwarding pattern as `onContextRestored`.
   */
  triggerChange(): void;
  /**
   * Optional hook fired when the context is LOST. Used to shrink the
   * adaptive GPU byte budget (context loss is a strong out-of-VRAM
   * signal), so the post-restore scene holds less and doesn't lose the
   * context again. Forwarded via a callback rather than importing the
   * budget module here, keeping this concern dependency-free + testable.
   */
  onContextLost?(): void;
}

/**
 * Walk the scene graph and flip `needsUpdate` on every attribute and
 * material reachable from a Mesh / InstancedMesh. Three.js
 * re-creates buffers/programs lazily, but the explicit dirty flag is
 * the one knob that makes recovery deterministic for our custom
 * shader materials, instanced geometry, and pooled buffer attributes.
 *
 * Pure with respect to the scene — does NOT mutate the graph itself,
 * only the attribute / material flags.
 */
export function markSceneResourcesDirtyForContextRestore(scene: THREE.Object3D): void {
  scene.traverse((obj) => {
    if (obj instanceof THREE.Mesh || obj instanceof THREE.InstancedMesh) {
      const geometry = obj.geometry;
      if (geometry) {
        const attributes = geometry.attributes as Record<
          string,
          THREE.BufferAttribute | THREE.InterleavedBufferAttribute
        >;
        for (const attribute of Object.values(attributes)) {
          attribute.needsUpdate = true;
        }
        if (geometry.index) {
          geometry.index.needsUpdate = true;
        }
      }

      const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const material of materials) {
        if (material) {
          material.needsUpdate = true;
        }
      }
    }
  });
}

/**
 * Owns the canvas-side `webglcontextlost` / `webglcontextrestored`
 * listeners and the per-listener cleanup. `attach()` registers, and
 * `dispose()` unregisters. The `isContextLost` flag is read via
 * `getIsContextLost()`.
 */
export class WebGLContextRecovery {
  private isContextLost = false;
  private contextLostHandler: ((event: Event) => void) | null = null;
  private contextRestoredHandler: ((event: Event) => void) | null = null;

  constructor(private deps: WebGLContextRecoveryDeps) {}

  /** Whether the WebGL context is currently lost. */
  getIsContextLost(): boolean {
    return this.isContextLost;
  }

  /**
   * Register the two canvas listeners. Call once during scene-manager
   * init; idempotency is the caller's responsibility (we don't
   * double-bind here, but we do log once).
   */
  attach(): void {
    const canvas = this.deps.canvas;

    this.contextLostHandler = (event: Event) => {
      event.preventDefault(); // Required to allow context restoration
      this.isContextLost = true;
      log.error(
        Modules.SCENE_MANAGER,
        'WebGL context lost! This can happen due to GPU driver issues, system sleep, or memory pressure.'
      );
      notifier.error(
        'Graphics context lost - attempting to restore. This can happen if your GPU driver crashes or the system runs out of video memory. The app will try to recover automatically.'
      );
      // Shrink the GPU byte budget — loss is a strong OOM signal, so the
      // restored scene should hold less and avoid losing the context again.
      this.deps.onContextLost?.();
    };

    this.contextRestoredHandler = async (_event: Event) => {
      log.info(Modules.SCENE_MANAGER, 'WebGL context restored - recreating resources...');
      try {
        this.isContextLost = false;

        // Force renderer to recreate its internal state.
        this.deps.renderer.resetState();

        // Rebuild post-processing GPU-bound resources in place. Identity
        // is preserved across the rebuild so PickingSystem,
        // AnimationController, and RenderingControls keep their cached
        // references valid; user settings (bloom, exposure, tone mapping,
        // detector noise, etc.) are preserved end-to-end.
        const postProcessing = this.deps.getPostProcessing();
        if (postProcessing) {
          postProcessing.rebuildAfterContextRestore();
        }

        // Drop the material cache so the renderer re-compiles shaders
        // against the new context on the next render. The cached
        // materials' programs are invalid now; re-creation is lazy.
        materialManager.rebuildAfterContextRestore();
        markSceneResourcesDirtyForContextRestore(this.deps.getScene());
        this.deps.updateRendererSize();

        // Notify subscribers (e.g. SceneLoader) so they can re-register
        // their picking-system / GPU-pool resources against the new
        // context. Order is intentional: post-processing → materials →
        // subscribers (which include node-factory).
        this.deps.onContextRestored();

        // Trigger a render to force Three.js material/program resource recreation.
        this.deps.triggerChange();

        notifier.hideLoading();
        log.success(Modules.SCENE_MANAGER, 'WebGL context successfully restored');
      } catch (error) {
        log.error(Modules.SCENE_MANAGER, 'Failed to restore WebGL context:', error);
        notifier.error('Failed to restore graphics context. Please refresh the page to continue.');
      }
    };

    canvas.addEventListener('webglcontextlost', this.contextLostHandler, false);
    canvas.addEventListener('webglcontextrestored', this.contextRestoredHandler, false);

    log.info(Modules.SCENE_MANAGER, 'WebGL context loss handling initialized');
  }

  /**
   * Remove the canvas listeners. Idempotent — calling on an
   * already-disposed instance is a no-op. Called from
   * SceneManager.dispose().
   */
  dispose(): void {
    if (this.contextLostHandler) {
      this.deps.canvas.removeEventListener('webglcontextlost', this.contextLostHandler);
      this.contextLostHandler = null;
    }
    if (this.contextRestoredHandler) {
      this.deps.canvas.removeEventListener('webglcontextrestored', this.contextRestoredHandler);
      this.contextRestoredHandler = null;
    }
  }
}
