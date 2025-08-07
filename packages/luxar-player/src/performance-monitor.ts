// Performance monitoring for the Luxar scene player using Stats.js
//
// This module provides real-time performance metrics using the industry-standard
// stats.js library. It tracks FPS, frame time, and memory usage with minimal
// performance overhead.

import Stats from 'stats.js';

/**
 * PerformanceMonitor manages real-time performance statistics display
 *
 * Features:
 * - FPS monitoring (frames per second)
 * - Frame time tracking (milliseconds per frame)
 * - Memory usage monitoring (JavaScript heap size)
 * - Toggle visibility with keyboard shortcuts
 * - Accessibility support with ARIA labels
 */
export class PerformanceMonitor {
  /** The stats.js instance that handles all performance measurements */
  private stats: Stats;

  /** Current visibility state of the performance panel */
  private isVisible = false;

  constructor() {
    // Initialize stats.js - this is a lightweight library that measures
    // performance metrics with minimal impact on the application
    this.stats = new Stats();
    this.setupStats();
  }

  /**
   * Setup the stats panel with custom styling and accessibility features
   *
   * This configures the stats.js panel for optimal visibility and usability:
   * - Positions it in a non-intrusive location (bottom-left corner)
   * - Sets appropriate z-index to appear above other UI elements
   * - Adds WCAG-compliant accessibility attributes
   * - Hides by default to avoid visual clutter
   */
  private setupStats(): void {
    // Configure stats display - start with FPS panel (most commonly used)
    // Panel types: 0=FPS (green), 1=Frame Time ms (yellow), 2=Memory MB (purple)
    this.stats.showPanel(0);

    // Get the DOM element that stats.js creates internally
    const statsElement = this.stats.dom;

    // Position the panel in bottom-left corner with fixed positioning
    // This ensures it stays visible during camera movements and zoom
    statsElement.style.position = 'fixed';
    statsElement.style.bottom = '20px'; // Match dimension slider bottom margin
    statsElement.style.left = '20px';   // Standard margin from edge
    statsElement.style.top = 'auto';    // Ensure no top positioning
    statsElement.style.right = 'auto';  // Ensure no right positioning
    statsElement.style.width = 'auto';  // Use natural width
    statsElement.style.height = 'auto'; // Use natural height

    // Set z-index higher than all other UI elements to ensure visibility
    // Help overlay uses 1001, so we use 2000 for performance stats
    statsElement.style.zIndex = '2000';

    // Slightly transparent to reduce visual impact while maintaining readability
    statsElement.style.opacity = '0.9';

    // Hidden by default - only shown when user explicitly requests it
    statsElement.style.display = 'none';

    // Remove focus outlines to prevent blue selection box
    statsElement.style.outline = 'none';
    
    // Add CSS to prevent blue selection on all child elements
    const style = document.createElement('style');
    style.textContent = `
      #stats {
        outline: none !important;
      }
      #stats * {
        outline: none !important;
        user-select: none !important;
      }
      #stats canvas {
        outline: none !important;
      }
    `;
    if (!document.getElementById('stats-custom-styles')) {
      style.id = 'stats-custom-styles';
      document.head.appendChild(style);
    }

    // Add WCAG 2.1 accessibility attributes for screen readers
    // 'status' role indicates this contains status information that updates
    statsElement.setAttribute('role', 'status');
    statsElement.setAttribute(
      'aria-label',
      'Performance metrics: FPS, frame time, and memory usage'
    );

    // Remove tabindex to prevent focus and blue outline
    // statsElement.setAttribute('tabindex', '0');

    // Inject into DOM - stats.js needs this to be in the document to function
    document.body.appendChild(statsElement);
  }

  /**
   * Start performance monitoring (call at beginning of render loop)
   *
   * This should be called at the very beginning of each animation frame,
   * before any Three.js rendering operations. It starts the high-precision
   * timer that measures frame duration.
   *
   * Only measures when visible to avoid performance overhead when not needed.
   */
  begin(): void {
    if (this.isVisible) {
      // stats.begin() records the current timestamp using performance.now()
      // This provides microsecond precision timing
      this.stats.begin();
    }
  }

  /**
   * End performance monitoring (call at end of render loop)
   *
   * This should be called at the very end of each animation frame,
   * after all Three.js rendering operations are complete. It calculates
   * the frame time and updates FPS metrics.
   *
   * The stats.js library automatically:
   * - Calculates frame time (end - begin)
   * - Updates FPS counter (frames per second)
   * - Tracks memory usage (if supported by browser)
   * - Updates the visual display
   */
  end(): void {
    if (this.isVisible) {
      // stats.end() calculates frame metrics and updates the display
      this.stats.end();
    }
  }

  /**
   * Toggle stats visibility
   */
  toggle(): void {
    this.isVisible = !this.isVisible;
    this.stats.dom.style.display = this.isVisible ? 'block' : 'none';

    // Don't focus to avoid blue outline
    // if (this.isVisible) {
    //   this.stats.dom.focus();
    // }
  }

  /**
   * Show performance stats
   */
  show(): void {
    if (!this.isVisible) {
      this.toggle();
    }
  }

  /**
   * Hide performance stats
   */
  hide(): void {
    if (this.isVisible) {
      this.toggle();
    }
  }

  /**
   * Get current visibility state
   */
  get visible(): boolean {
    return this.isVisible;
  }

  /**
   * Cycle through different stats panels (FPS -> MS -> MB -> back to FPS)
   */
  cyclePanels(): void {
    if (this.isVisible) {
      const currentPanel = (this.stats.dom as any).panel || 0;
      const nextPanel = (currentPanel + 1) % 3; // 0: fps, 1: ms, 2: mb
      this.stats.showPanel(nextPanel);
    }
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    if (this.stats.dom.parentNode) {
      this.stats.dom.parentNode.removeChild(this.stats.dom);
    }
  }
}
