// Animation loop management for the Luxar scene player
//
// This module handles the core animation loop that drives 3D rendering:
// - RequestAnimationFrame-based rendering loop for smooth 60fps
// - Intelligent pause/resume system to conserve CPU/GPU when idle
// - Performance monitoring integration for real-time metrics
// - Proper cleanup and resource management

import { ControlsManager } from '../../controls/controls-manager';
import { config } from '../../config';
import { PostProcessingManager } from '../../rendering';
import { AdaptiveDPRManager } from '../../rendering/adaptive-dpr-manager';
import { eventBus } from '../../utils/cross-layer/event-bus';

/**
 * AnimationController manages the main rendering loop and performance optimization
 *
 * Key Features:
 * - RequestAnimationFrame loop for browser-optimized rendering
 * - Automatic pause/resume based on user interaction (saves power)
 * - HDR post-processing pipeline with bloom effects
 * - Integrated performance monitoring with stats.js
 * - Proper frame timing and resource cleanup
 *
 * Technical Details:
 * - Uses requestAnimationFrame for 60fps synchronized with display refresh
 * - Pauses after 2 seconds of inactivity to reduce CPU/GPU usage
 * - Integrates Three.js controls.update() and HDR post-processing render
 * - Measures frame timing for performance analysis including post-processing
 */
export class AnimationController {
  /** Whether the animation loop is currently running */
  private isAnimating = false;

  /** RequestAnimationFrame ID for cancellation */
  private animationId: number = 0;

  /** Timeout ID for auto-pause functionality */
  private idleTimeout: ReturnType<typeof setTimeout> | null = null;

  /** Per-frame callbacks for additional updates (keyed by ID for safe add/remove) */
  private perFrameCallbacks: Map<string, { callback: () => void; continuous: boolean }> = new Map();

  /** Adaptive DPR manager for dynamic resolution scaling */
  private adaptiveDPRManager: AdaptiveDPRManager | null = null;

  /**
   * Predicate that returns true while the WebGL context is lost. When
   * set, the animation loop skips `postProcessing.render()` (and any
   * GPU-bound work) so we don't issue draw calls against a dead
   * context — those produce noisy GL errors and waste frame work
   * during the loss window. The renderer is rebuilt by SceneManager
   * on `webgl-context-restored`; until then we keep ticking
   * controls.update() and per-frame callbacks but skip rendering.
   */
  private isContextLost: (() => boolean) | null = null;

  /** Guard for the idle-pause DPR restore (null = always allowed). */
  private canRestoreAtIdle: (() => boolean) | null = null;

  /**
   * Create animation controller for rendering loop management.
   *
   * Sets up performance monitoring and prepares animation loop. Does not
   * start animation - call startAnimation() to begin rendering.
   *
   * @param controls - Controls manager for camera updates each frame
   * @param postProcessing - Post-processing manager for HDR rendering
   *
   * @example
   * ```typescript
   * const animController = new AnimationController(
   *   controlsManager,
   *   postProcessingManager
   * );
   * animController.startAnimation();  // Begin rendering loop
   * ```
   */
  constructor(
    private controls: ControlsManager,
    private postProcessing: PostProcessingManager
  ) {}

  /**
   * Add a per-frame callback with a unique identifier.
   *
   * Multiple callbacks can be registered simultaneously (unlike setPerFrameCallback).
   * Use unique IDs to allow safe removal without affecting other callbacks.
   *
   * Useful for operations that need to run every frame:
   * - Dynamic clipping plane adjustments
   * - Dimension animations
   * - Camera-based LOD updates
   * - Custom animations or effects
   *
   * The callback is executed after controls.update() but before rendering.
   *
   * @param id - Unique identifier for this callback (for later removal)
   * @param callback - Function to call each frame
   * @param options - Options controlling callback behavior
   * @param options.continuous - If true, this callback prevents the animation loop from
   *   auto-pausing due to idle timeout. Use for callbacks that need every frame (e.g.,
   *   dimension animation, recording). Default: false (on-demand callbacks that only run
   *   when animation is active but don't prevent pausing).
   *
   * @example
   * ```typescript
   * // On-demand callback: runs when animating but doesn't prevent idle pause
   * animController.addPerFrameCallback('dynamic-clipping', () => {
   *   sceneManager.updateDynamicClippingPlanes();
   * });
   *
   * // Continuous callback: keeps animation loop alive
   * animController.addPerFrameCallback('dimension-animation', () => {
   *   animationManager.onFrame();
   * }, { continuous: true });
   * ```
   */
  addPerFrameCallback(id: string, callback: () => void, options?: { continuous?: boolean }): void {
    this.perFrameCallbacks.set(id, { callback, continuous: options?.continuous ?? false });
  }

  /**
   * Remove a per-frame callback by its identifier.
   *
   * @param id - Identifier of the callback to remove
   * @returns true if callback was found and removed, false otherwise
   *
   * @example
   * ```typescript
   * // Remove dimension animation callback
   * animController.removePerFrameCallback('dimension-animation');
   * ```
   */
  removePerFrameCallback(id: string): boolean {
    return this.perFrameCallbacks.delete(id);
  }

  /**
   * Check if a per-frame callback with the given ID exists.
   *
   * @param id - Identifier to check
   * @returns true if callback exists
   */
  hasPerFrameCallback(id: string): boolean {
    return this.perFrameCallbacks.has(id);
  }

  /**
   * Set the adaptive DPR manager for dynamic resolution scaling.
   *
   * The animation loop will call recordFrame() on the manager each frame
   * to track FPS and adjust pixel ratio as needed.
   *
   * @param manager - The AdaptiveDPRManager instance, or null to disable
   */
  setAdaptiveDPRManager(manager: AdaptiveDPRManager | null): void {
    this.adaptiveDPRManager = manager;
  }

  /**
   * Inject a predicate the loop can poll to detect WebGL context
   * loss. When the predicate returns true, the animation loop skips
   * `postProcessing.render()` for that frame; controls and per-frame
   * callbacks still run so user input stays responsive. SceneManager
   * wires this to its own `isWebGLContextLost()`.
   *
   * Pass `null` to disable the guard (useful in tests / embed contexts
   * that can't lose the context).
   */
  setContextLostPredicate(predicate: (() => boolean) | null): void {
    this.isContextLost = predicate;
  }

  /**
   * Inject a predicate consulted before the idle-pause native-DPR
   * restore. When it returns false the resting frame keeps the current
   * DPR — used to protect recordings, whose resolution must stay
   * locked for the whole capture. Mirrors `setContextLostPredicate`.
   * Pass `null` to always allow the restore.
   */
  setIdleRestorePredicate(predicate: (() => boolean) | null): void {
    this.canRestoreAtIdle = predicate;
  }

  /**
   * Main animation loop function - the heart of HDR 3D rendering
   *
   * This function is called ~60 times per second (depending on display refresh rate)
   * and handles the complete HDR render pipeline:
   *
   * 1. Performance measurement begins
   * 2. Schedule next frame via requestAnimationFrame
   * 3. Update camera controls (handle user input, damping, constraints)
   * 4. Render through HDR post-processing pipeline (scene → bloom → tone mapping)
   * 5. Performance measurement ends
   *
   * Uses arrow function to maintain 'this' context when passed as callback.
   * Early return prevents unnecessary work when animation is paused.
   */
  private animate = (): void => {
    // Early exit if animation is paused - prevents unnecessary GPU work
    if (!this.isAnimating) return;

    // Begin frame timing measurement for performance analysis.
    // Emits on the event bus so subscribers (e.g., the
    // PerformanceMonitor UI panel) can record the start timestamp
    // without animation-controller importing UI code directly.
    eventBus.emit('frame-start', {});

    // Record frame for adaptive DPR - tracks FPS and adjusts pixel
    // ratio. Skipped while the rendering context is lost: those frames
    // do no GPU work, so their "speed" would drive bogus scale-ups and
    // falsely settle U-shape probes.
    if (this.adaptiveDPRManager && !this.isContextLost?.()) {
      this.adaptiveDPRManager.recordFrame(performance.now());
    }

    // Schedule the next frame - requestAnimationFrame syncs with display refresh
    // This provides smooth 60fps on most displays, or 120fps on high-refresh monitors
    this.animationId = requestAnimationFrame(this.animate);

    // Update camera controls - processes mouse/touch input and applies damping
    // This must happen before rendering to reflect user interactions
    this.controls.update();

    // Call all registered per-frame callbacks (e.g., dynamic clipping, dimension animation)
    for (const entry of this.perFrameCallbacks.values()) {
      entry.callback();
    }

    // Skip GPU rendering while the WebGL context is lost. The
    // post-processing render() would otherwise issue draw calls
    // against a dead context (noisy GL errors, driver-specific
    // exceptions on some platforms). Controls and per-frame callbacks
    // already ran above so user input stays responsive while the
    // browser drives recovery.
    if (this.isContextLost?.()) {
      eventBus.emit('frame-end', {});
      return;
    }

    // Render through HDR post-processing pipeline
    // This executes the complete chain: Scene → HDR buffer → Bloom → Tone mapping → Display
    // Includes vertex shaders, fragment shaders, HDR buffers, bloom blur, ACES tone mapping
    this.postProcessing.render();

    // End frame timing — pair with the frame-start emit above. The
    // PerformanceMonitor UI panel subscribes to both events when
    // visible and feeds them into stats.js for FPS / frame-time
    // readouts.
    eventBus.emit('frame-end', {});
  };

  /**
   * Check if any features require continuous animation
   * @returns True if animation should continue regardless of user interaction
   */
  private shouldContinueAnimating(): boolean {
    // Check if auto-rotate is enabled
    const autoRotate = this.controls.getAutoRotate();

    // Check if any post-processing effects need continuous updates
    const hasEffects = this.postProcessing.needsContinuousAnimation();

    // Check if any continuous per-frame callbacks are active (e.g., turntable recording, dimension animation)
    // On-demand callbacks (continuous: false) like dynamic-clipping and scale-bar don't prevent idle pause
    const hasContinuousCallbacks = [...this.perFrameCallbacks.values()].some(
      (entry) => entry.continuous
    );

    return autoRotate || hasEffects || hasContinuousCallbacks;
  }

  /**
   * Handle idle timeout - only stop if no continuous effects are active
   */
  private handleIdleTimeout = (): void => {
    // Check if we should continue animating due to effects or auto-rotate
    if (this.shouldContinueAnimating()) {
      // Continuous effects are active, schedule another check
      this.idleTimeout = setTimeout(this.handleIdleTimeout, config.animation.idleTimeoutMs);
    } else {
      // No continuous effects, safe to stop animation
      this.stopAnimation();

      // Idle restore: the static frame the user is about to study
      // should be at full native sharpness — reduced DPR only ever
      // traded quality for interaction smoothness. This lives ONLY in
      // the idle path (never in stopAnimation itself, which also runs
      // on tab-hide and dispose where rendering would be wrong).
      // prepareIdleFrame() returns true only when the DPR actually
      // changed; the resize clears the canvas, so exactly then we
      // render ONE frame directly — NOT via startAnimation(), which
      // would re-arm the idle timer and feed native-DPR frames back
      // into the FPS evaluator.
      if (
        this.adaptiveDPRManager?.isActive?.() &&
        this.canRestoreAtIdle?.() !== false &&
        !this.isContextLost?.() &&
        this.adaptiveDPRManager.prepareIdleFrame?.()
      ) {
        this.postProcessing.render();
      }
    }
  };

  /**
   * Start animation loop and reset idle timer for power efficiency
   *
   * This method is called whenever user interaction is detected:
   * - Mouse movement over canvas
   * - Camera control events (start, change)
   * - Keyboard input
   * - Touch events
   * - When continuous effects are enabled (noise, auto-rotate)
   *
   * The idle timer automatically pauses rendering after inactivity to:
   * - Reduce CPU/GPU usage when scene is static
   * - Improve battery life on mobile devices
   * - Lower thermal impact on laptops
   * - Maintain 0% CPU usage when user is not interacting
   *
   * Continuous effects (noise, auto-rotate) will keep animation running.
   *
   * Uses arrow function to maintain 'this' context when used as event handler.
   */
  startAnimation = (): void => {
    // Only start if not already running - prevents duplicate loops
    if (!this.isAnimating) {
      this.isAnimating = true;
      // Resuming from a rest: let the adaptive DPR manager snap back to
      // its remembered operating DPR in one step (stopped→running edge
      // only — this must not fire on every interaction poke).
      this.adaptiveDPRManager?.notifyResumed?.();
      // Kick off the first frame - subsequent frames are scheduled by animate()
      this.animate();
    }

    // Reset the idle timeout - this is called on every user interaction
    // Clear any existing timeout to prevent premature stopping
    if (this.idleTimeout !== null) {
      clearTimeout(this.idleTimeout);
    }

    // Set new timeout to check for idle - will continue if continuous effects are active
    // This is the key power-saving optimization for static scenes
    this.idleTimeout = setTimeout(this.handleIdleTimeout, config.animation.idleTimeoutMs);
  };

  /**
   * Stop animation loop and clean up timers
   *
   * This method halts all rendering activity to conserve resources:
   * - Sets flag to prevent further animate() calls
   * - Cancels pending requestAnimationFrame to stop browser scheduling
   * - Clears idle timeout to prevent memory leaks
   *
   * Called automatically after idle timeout (when no continuous effects)
   * or manually for cleanup. Scene remains visible but static until
   * next user interaction or continuous effect activation.
   */
  stopAnimation = (): void => {
    // Set flag to prevent animate() from continuing the loop
    this.isAnimating = false;

    // The FPS window, hysteresis streak, and any in-flight probe are
    // about to go stale across the pause — clear them (session state
    // only; learned floors survive). Method-level optional chaining is
    // deliberate: tests inject bare {recordFrame} manager mocks, and
    // this also runs from dispose() after the manager may be gone.
    this.adaptiveDPRManager?.notifyPaused?.();

    // Cancel any pending requestAnimationFrame call
    // This ensures no more frames are scheduled by the browser
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }

    // Clear the idle timeout to prevent memory leaks
    if (this.idleTimeout !== null) {
      clearTimeout(this.idleTimeout);
      this.idleTimeout = null;
    }
  };

  /**
   * Get current animation loop state.
   *
   * @returns true if animation loop is running, false if paused
   */
  get isActive(): boolean {
    return this.isAnimating;
  }

  /**
   * Get performance monitor for FPS and timing metrics.
   *
   * Stop animation loop and clean up resources.
   *
   * Stops rendering and cancels timers. The PerformanceMonitor UI
   * panel lives at LuxarApp; this controller emits `frame-start` /
   * `frame-end` on the event bus per frame, which is what the panel
   * listens to.
   *
   * After calling dispose(), the animation controller cannot be reused.
   */
  dispose(): void {
    this.stopAnimation();
    this.perFrameCallbacks.clear();
  }
}
