// Animation loop management for the Luxar scene player
//
// This module handles the core animation loop that drives 3D rendering:
// - RequestAnimationFrame-based rendering loop for smooth 60fps
// - Intelligent pause/resume system to conserve CPU/GPU when idle
// - Performance monitoring integration for real-time metrics
// - Proper cleanup and resource management

import { ControlsManager } from '../controls/controls-manager';
import { config } from '../config';
import { PerformanceMonitor } from '../ui/performance-monitor';
import { PostProcessingManager } from '../rendering/post-processing/post-processing-manager';
import { AdaptiveDPRManager } from '../rendering/adaptive-dpr-manager';

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

  /** Performance monitoring instance for FPS/timing metrics */
  private performanceMonitor: PerformanceMonitor;

  /** Per-frame callbacks for additional updates (keyed by ID for safe add/remove) */
  private perFrameCallbacks: Map<string, { callback: () => void; continuous: boolean }> = new Map();

  /** Adaptive DPR manager for dynamic resolution scaling */
  private adaptiveDPRManager: AdaptiveDPRManager | null = null;

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
  ) {
    // Initialize performance monitoring for frame timing analysis
    this.performanceMonitor = new PerformanceMonitor();
  }

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

    // Begin frame timing measurement for performance analysis
    // This records the start timestamp for FPS and frame time calculations
    this.performanceMonitor.begin();

    // Record frame for adaptive DPR - tracks FPS and adjusts pixel ratio
    if (this.adaptiveDPRManager) {
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

    // Render through HDR post-processing pipeline
    // This executes the complete chain: Scene → HDR buffer → Bloom → Tone mapping → Display
    // Includes vertex shaders, fragment shaders, HDR buffers, bloom blur, ACES tone mapping
    this.postProcessing.render();

    // End frame timing measurement - calculates frame duration and updates FPS
    // Now includes the cost of HDR post-processing in performance metrics
    this.performanceMonitor.end();
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
   * Provides access to stats.js panel for toggling visibility (P key)
   * and retrieving performance data.
   *
   * @returns PerformanceMonitor instance tracking FPS and frame time
   */
  get performanceStats(): PerformanceMonitor {
    return this.performanceMonitor;
  }

  /**
   * Stop animation loop and clean up resources.
   *
   * Stops rendering, cancels timers, and disposes performance monitor.
   * Should be called during application teardown.
   *
   * After calling dispose(), the animation controller cannot be reused.
   */
  dispose(): void {
    this.stopAnimation();
    this.perFrameCallbacks.clear();
    this.performanceMonitor.dispose();
  }
}
