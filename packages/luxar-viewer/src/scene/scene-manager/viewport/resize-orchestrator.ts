/**
 * Resize orchestrator — owns the rAF-coalescing window-resize loop for
 * SceneManager.
 *
 * SceneManager holds one instance and delegates updateSize() to it.
 * The orchestrator carries its own rAF handle + pending dimensions
 * (the only state needed) and reads renderer / camera / postProcessing
 * references from the supplied ctx — these change over the
 * SceneManager's lifetime (camera type swap, post-processing late
 * init) so reading them on every resize keeps the orchestrator in
 * sync without re-construction.
 */

import { log, Modules } from '../../../utils/log';
import { type LuxarCamera, updateCameraAspect } from '../../../utils/camera-utils';
import { isDocumentFullscreen } from '../../../utils/fullscreen';
import type { PostProcessingManager } from '../../../rendering';
import type { Renderer } from '../../../rendering/renderer-capabilities';
import { getActivePixelRatio, syncPostProcessingDPRScale } from './dpr-policy';

export interface ResizeCtx {
  readonly renderer: Renderer;
  readonly camera: LuxarCamera | null;
  readonly postProcessing: PostProcessingManager | null;
  readonly pixelRatioOverride: number | null;
  /** Called after every resize to refresh world-space point sizing uniforms. */
  updateMaterialsForCurrentCamera(): void;
}

export class ResizeOrchestrator {
  private pendingResize: { width: number; height: number } | null = null;
  private resizeRAF: number | null = null;
  /** Suppress resize while a recording is in progress (resolution must stay locked). */
  resizeLocked = false;

  /**
   * Schedule a resize. Coalesces multiple synchronous resize events into a
   * single rAF callback so a burst of ResizeObserver fires doesn't trigger
   * N renderer resizes.
   *
   * Dimensions come from `getDims` (SceneManager passes its parent-first
   * canvas measurement so embedded viewers track their host container);
   * defaults to the window size for window-sized callers.
   */
  scheduleResize(
    getCtx: () => ResizeCtx,
    getDims: () => { width: number; height: number } = () => ({
      width: window.innerWidth,
      height: window.innerHeight,
    })
  ): void {
    if (this.resizeLocked) return;

    this.pendingResize = getDims();

    if (this.resizeRAF !== null) {
      cancelAnimationFrame(this.resizeRAF);
    }

    this.resizeRAF = requestAnimationFrame(() => {
      if (!this.pendingResize) return;
      this.doResize(this.pendingResize.width, this.pendingResize.height, getCtx());
      this.pendingResize = null;
      this.resizeRAF = null;
    });
  }

  /**
   * Apply a resize immediately (no rAF coalescing). Used by the manual
   * adaptive-DPR path which already runs at frame-level granularity.
   * Falls back to direct renderer sizing when post-processing hasn't
   * been initialised yet.
   */
  resizeNow(width: number, height: number, ctx: ResizeCtx): void {
    this.doResize(width, height, ctx);
  }

  /** Cancel any in-flight rAF and clear pending state. Called on SceneManager.dispose(). */
  dispose(): void {
    if (this.resizeRAF !== null) {
      cancelAnimationFrame(this.resizeRAF);
      this.resizeRAF = null;
    }
    this.pendingResize = null;
  }

  private doResize(width: number, height: number, ctx: ResizeCtx): void {
    if (isDocumentFullscreen()) {
      log.success(Modules.SCENE_MANAGER, `Using fullscreen dimensions: ${width}x${height}`);
    } else {
      log.success(Modules.SCENE_MANAGER, `Using windowed dimensions: ${width}x${height}`);
    }

    if (ctx.camera) {
      updateCameraAspect(ctx.camera, width, height);
    }

    // Ensure pixel ratio stays current. When adaptive/manual DPR is
    // active, preserve that explicit override across ordinary window
    // resizes; when no override is active, track ceiling changes
    // (including display changes when dragging between monitors).
    ctx.renderer.setPixelRatio(getActivePixelRatio(ctx.pixelRatioOverride));

    // PostProcessingManager owns renderer + composer sizing — it calls
    // renderer.setSize() and composer.setSize() internally via resize().
    // Only fall back to direct renderer sizing during early init before
    // PostProcessingManager has been created.
    if (ctx.postProcessing) {
      ctx.postProcessing.resize(width, height);
      syncPostProcessingDPRScale(ctx.postProcessing, ctx.pixelRatioOverride);
    } else {
      this.updateRendererSize(width, height, ctx);
    }

    if (ctx.camera) {
      ctx.updateMaterialsForCurrentCamera();
    }
  }

  private updateRendererSize(width: number, height: number, ctx: ResizeCtx): void {
    // Set pixel ratio BEFORE size for correct buffer calculations.
    ctx.renderer.setPixelRatio(getActivePixelRatio(ctx.pixelRatioOverride));
    ctx.renderer.setSize(width, height); // Allow Three.js to set CSS size.
    // Material refresh is handled by the doResize() epilogue — no need to
    // duplicate it here. Keeping this method narrowly focused on renderer
    // sizing avoids a per-resize call to materialManager.updateCameraParams.
  }
}
