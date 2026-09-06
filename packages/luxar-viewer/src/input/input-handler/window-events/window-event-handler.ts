/**
 * Window-event concern extracted from `input/input-handler.ts`.
 *
 * Owns the three window/document-level event handlers the viewer
 * registers on the global scope:
 *
 *   - `resize` (window) → forward to `SceneManager.updateSize()` and
 *     kick the animation loop.
 *   - `wheel` (window) → orbit-controls own the actual zoom math, so
 *     this handler only intercepts Ctrl/Meta+wheel for FOV control
 *     and pokes the rendering-controls UI to switch its preset to
 *     "Custom" so the slider value reflects the new FOV.
 *   - `fullscreenchange` (document) → toggle fullscreen-fitting
 *     inline styles on the canvas, then on the next frame run
 *     `updateSize()` once (modern browsers fire this event AFTER the
 *     viewport transition completes, so a single rAF is enough).
 *
 * The class registers its listeners via `attach(cleanups)` and
 * pushes the unregistration thunks onto the caller's cleanup array,
 * matching the InputHandler's existing `eventListeners` ownership
 * model. There's no separate `dispose()` — the InputHandler runs the
 * cleanup array on its own dispose path.
 *
 * The global wheel listener applies rendering behavior only to events from the
 * scene canvas, while still suppressing modifier-wheel page zoom over viewer UI.
 *
 * @module input/handlers/window-event-handler
 */

import type { SceneManager } from '../../../scene/scene-manager';
import type { AnimationController } from '../../../scene/animation/animation-controller';
import type { RenderingControlsHandle } from '../panel-capabilities';
import { isDocumentFullscreen } from '../../../utils/fullscreen';
import { isPerspectiveCamera } from '../../../utils/camera-utils';
import { getViewerContainer } from '../../../utils/viewer-container';
import { normalizeWheelDelta } from '../../../utils/wheel-delta';

export class WindowEventHandler {
  private renderingControls?: RenderingControlsHandle;

  /**
   * The canvas `style` attribute as it was just before we entered
   * fullscreen, so exiting restores exactly what the embedder authored
   * (e.g. `width:100%;height:100%` inside their container) instead of
   * wiping every inline style. `null` when not in fullscreen.
   */
  private savedCanvasStyle: string | null = null;

  constructor(
    private sceneManager: SceneManager,
    private animationController: AnimationController
  ) {}

  /**
   * Late-bind the rendering-controls reference. The InputHandler may
   * receive its `setRenderingControls(controls)` call after the
   * window listeners are already registered, so we accept the late
   * wiring instead of forcing the caller to re-attach.
   */
  setRenderingControls(rc: RenderingControlsHandle | undefined): void {
    this.renderingControls = rc;
  }

  /**
   * Register the three window/document listeners and append the
   * unregistration thunks to `cleanups`. Returns nothing — the
   * cleanups array is the caller's source of truth.
   */
  attach(cleanups: (() => void)[]): void {
    const onResize = () => this.onWindowResize();
    const onWheel = (event: WheelEvent) => this.onWheel(event);
    const onFullscreenChange = () => this.onFullscreenChange();

    window.addEventListener('resize', onResize);
    // Register the wheel listener with `passive: false` so the
    // Ctrl/Cmd+wheel FOV handler's `event.preventDefault()` reliably
    // suppresses page zoom. Browsers can default wheel listeners on
    // root targets (window/document) to passive in some configurations,
    // in which case `preventDefault()` is silently ignored and the page
    // zooms while the FOV also changes.
    window.addEventListener('wheel', onWheel, { passive: false });
    document.addEventListener('fullscreenchange', onFullscreenChange);
    // Safari < 16.4 exposes only the webkit-prefixed fullscreen API, so it
    // fires `webkitfullscreenchange` (not `fullscreenchange`). Since
    // toggleFullscreen() now enters fullscreen via `webkitRequestFullscreen`
    // there, we must listen for the webkit event too — otherwise the canvas
    // never gets resized to fill the viewport on those browsers.
    document.addEventListener('webkitfullscreenchange', onFullscreenChange);

    cleanups.push(
      () => window.removeEventListener('resize', onResize),
      () => window.removeEventListener('wheel', onWheel),
      () => document.removeEventListener('fullscreenchange', onFullscreenChange),
      () => document.removeEventListener('webkitfullscreenchange', onFullscreenChange)
    );
  }

  /**
   * Handle window resize: update canvas / camera dimensions and kick
   * the animation loop so the resized scene renders.
   */
  private onWindowResize(): void {
    this.sceneManager.updateSize();
    this.animationController.startAnimation();
  }

  /**
   * Handle fullscreen enter/exit. While in fullscreen, force the
   * canvas to fill the viewport via inline styles; on exit, drop the
   * `style` attribute entirely.
   *
   * The single rAF after the style change is intentional: modern
   * browsers fire `fullscreenchange` after the viewport transition
   * completes, so one frame is enough to capture final dimensions
   * before calling updateSize().
   */
  private onFullscreenChange(): void {
    const canvas = this.sceneManager.renderer.domElement;

    // Check both the standard and webkit fullscreen elements (Safari < 16.4).
    if (isDocumentFullscreen()) {
      // Entering fullscreen — save the embedder's inline styles first, then
      // make the canvas fill the entire screen. Saving only on a clean
      // enter (savedCanvasStyle === null) avoids clobbering the saved value
      // if fullscreenchange fires twice.
      if (this.savedCanvasStyle === null) {
        this.savedCanvasStyle = canvas.getAttribute('style') ?? '';
      }
      canvas.style.width = '100vw';
      canvas.style.height = '100vh';
      canvas.style.position = 'fixed';
      canvas.style.top = '0';
      canvas.style.left = '0';
      canvas.style.opacity = '1';
      canvas.style.filter = 'none';
    } else {
      // Exiting fullscreen — restore exactly the inline styles the canvas
      // had before, rather than wiping everything (which would drop an
      // embedder's own sizing/positioning rules).
      if (this.savedCanvasStyle) {
        canvas.setAttribute('style', this.savedCanvasStyle);
      } else {
        canvas.removeAttribute('style');
      }
      this.savedCanvasStyle = null;
    }

    requestAnimationFrame(() => {
      this.sceneManager.updateSize();
      this.animationController.startAnimation();
    });
  }

  /**
   * Handle mouse-wheel events. Orbit/ortho controls own the actual
   * zoom math (they listen for `wheel` on the canvas separately). All
   * we do here is:
   *
   *   - For canvas-originated events, poke the animation loop so the
   *     scene keeps rendering during continuous wheel input.
   *   - On Ctrl+wheel / Cmd+wheel, suppress page zoom across the viewer
   *     container, but apply FOV control only to canvas-originated events.
   *     The FOV wheel path is gated to a perspective camera: in
   *     ortho the orbit controls own modifier-wheel (and trackpad-pinch)
   *     zoom. `updateFOV` now persists the perspective FOV stash even in
   *     ortho for DELIBERATE reset/zarr/panel applies, so the interactive
   *     wheel must be gated here or a pinch would corrupt that stash. The
   *     delta handed to `updateFOV` is normalized to pixel-mode equivalent
   *     (`normalizeWheelDelta`), which puts a line-mode browser's notch in the
   *     same ballpark as a pixel-mode one — 2.4 vs 5.0 degrees, instead of
   *     0.15 vs 5.0 before. When the FOV actually changed, switch the
   *     rendering-controls preset to "Custom" so the panel value matches the
   *     slider.
   */
  private onWheel(event: WheelEvent): void {
    const eventPath = event.composedPath();
    const canvas = this.sceneManager.renderer.domElement;
    const isCanvasEvent = eventPath[0] === canvas;

    if (
      (event.ctrlKey || event.metaKey) &&
      (isCanvasEvent || eventPath.includes(getViewerContainer()))
    ) {
      // Suppress browser page zoom for modifier-wheel events over the viewer,
      // including panels. An embedder's host-page UI remains untouched.
      event.preventDefault();
    }

    if (!isCanvasEvent) return;

    this.animationController.startAnimation();

    if (event.ctrlKey || event.metaKey) {
      // FOV only applies to a perspective camera; in ortho the orbit controls
      // own modifier-wheel zoom. Gate the interactive wheel path here (the
      // deliberate reset/zarr/panel-apply paths still persist the stash via
      // updateFOV) so a pinch/ctrl-wheel zoom in ortho can't corrupt it.
      if (!isPerspectiveCamera(this.sceneManager.camera)) return;

      const fovChanged = this.sceneManager.updateFOV(normalizeWheelDelta(event, canvas));

      // Flip the preset when the FOV actually changed so the panel value
      // matches the slider.
      if (fovChanged && this.renderingControls) {
        this.renderingControls.settings.fovPreset = 'Custom';
        this.renderingControls.syncCurrentState();
      }
    }
  }
}
