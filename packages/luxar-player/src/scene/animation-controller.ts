// Animation loop management for the Luxar scene player
//
// This module handles the core animation loop that drives 3D rendering:
// - RequestAnimationFrame-based rendering loop for smooth 60fps
// - Intelligent pause/resume system to conserve CPU/GPU when idle
// - Performance monitoring integration for real-time metrics
// - Proper cleanup and resource management

import * as THREE from 'three';
import { ControlsManager } from '../controls/controls-manager';
import { config } from '../config';
import { PerformanceMonitor } from '../ui/performance-monitor';
import { PostProcessingManager } from '../rendering/postprocessing-manager';

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
  private idleTimeout: number = 0;

  /** Performance monitoring instance for FPS/timing metrics */
  private performanceMonitor: PerformanceMonitor;

  constructor(
    _renderer: THREE.WebGLRenderer,
    _scene: THREE.Scene,
    _camera: THREE.PerspectiveCamera,
    private controls: ControlsManager,
    private postProcessing: PostProcessingManager
  ) {
    // Initialize performance monitoring for frame timing analysis
    this.performanceMonitor = new PerformanceMonitor();
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

    // Schedule the next frame - requestAnimationFrame syncs with display refresh
    // This provides smooth 60fps on most displays, or 120fps on high-refresh monitors
    this.animationId = requestAnimationFrame(this.animate);

    // Update camera controls - processes mouse/touch input and applies damping
    // This must happen before rendering to reflect user interactions
    this.controls.update();

    // Render through HDR post-processing pipeline
    // This executes the complete chain: Scene → HDR buffer → Bloom → Tone mapping → Display
    // Includes vertex shaders, fragment shaders, HDR buffers, bloom blur, ACES tone mapping
    this.postProcessing.render();

    // End frame timing measurement - calculates frame duration and updates FPS
    // Now includes the cost of HDR post-processing in performance metrics
    this.performanceMonitor.end();
  };

  /**
   * Start animation loop and reset idle timer for power efficiency
   *
   * This method is called whenever user interaction is detected:
   * - Mouse movement over canvas
   * - Camera control events (start, change)
   * - Keyboard input
   * - Touch events
   *
   * The idle timer automatically pauses rendering after inactivity to:
   * - Reduce CPU/GPU usage when scene is static
   * - Improve battery life on mobile devices
   * - Lower thermal impact on laptops
   * - Maintain 0% CPU usage when user is not interacting
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
    clearTimeout(this.idleTimeout);

    // Set new timeout to auto-pause after configured idle period (2 seconds)
    // This is the key power-saving optimization for static scenes
    this.idleTimeout = setTimeout(this.stopAnimation, config.animation.idleTimeoutMs);
  };

  /**
   * Stop animation loop and clean up timers
   *
   * This method halts all rendering activity to conserve resources:
   * - Sets flag to prevent further animate() calls
   * - Cancels pending requestAnimationFrame to stop browser scheduling
   * - Clears idle timeout to prevent memory leaks
   *
   * Called automatically after idle timeout or manually for cleanup.
   * Scene remains visible but static until next user interaction.
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
    clearTimeout(this.idleTimeout);
  };

  /**
   * Get current animation state
   */
  get isActive(): boolean {
    return this.isAnimating;
  }

  /**
   * Get performance monitor instance
   */
  get performanceStats(): PerformanceMonitor {
    return this.performanceMonitor;
  }

  /**
   * Cleanup animation resources
   */
  dispose(): void {
    this.stopAnimation();
    this.performanceMonitor.dispose();
  }
}
