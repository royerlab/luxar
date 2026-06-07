// Performance monitoring for the Luxar scene player using Stats.js
//
// This module provides real-time performance metrics using the industry-standard
// stats.js library. It tracks FPS, frame time, and memory usage with minimal
// performance overhead.

import Stats from 'stats.js';
import { config } from '../config';
import { eventBus, type Unsubscribe } from '../utils/cross-layer/event-bus';
import { getViewerContainer } from '../utils/viewer-container';

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

  /**
   * Bus subscriptions for frame-start / frame-end timing. Set when
   * the panel is visible, cleared when hidden so stats.js incurs no
   * cost while the user can't see the readout.
   */
  private frameStartUnsubscribe: Unsubscribe | null = null;
  private frameEndUnsubscribe: Unsubscribe | null = null;

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

    // Get the DOM element that stats.js creates internally. The library
    // does not set an id or class on this element, so we tag it ourselves
    // — the injected <style> block below scopes its rules via this id.
    const statsElement = this.stats.dom;
    statsElement.id = 'luxar-stats';

    // Position the panel in bottom-left corner with fixed positioning
    // This ensures it stays visible during camera movements and zoom
    statsElement.style.position = 'fixed';
    statsElement.style.bottom = '20px'; // Match dimension slider bottom margin
    statsElement.style.left = '20px'; // Standard margin from edge
    statsElement.style.top = 'auto'; // Ensure no top positioning
    statsElement.style.right = 'auto'; // Ensure no right positioning
    statsElement.style.width = 'auto'; // Use natural width
    statsElement.style.height = 'auto'; // Use natural height

    // Set z-index from config for consistent layering
    statsElement.style.zIndex = String(config.ui.zIndex.statsMonitor);

    // Slightly transparent to reduce visual impact while maintaining readability
    statsElement.style.opacity = '0.9';

    // Hidden by default - only shown when user explicitly requests it
    statsElement.style.display = 'none';

    // Remove focus outlines to prevent blue selection box
    statsElement.style.outline = 'none';

    // Suppress focus outlines and text-selection on the stats panel and
    // its (canvas) children. The selectors are scoped to #luxar-stats
    // (set above) so they cannot leak into the host page.
    if (!document.getElementById('luxar-stats-custom-styles')) {
      const style = document.createElement('style');
      style.id = 'luxar-stats-custom-styles';
      style.textContent = `
        #luxar-stats { outline: none !important; }
        #luxar-stats * {
          outline: none !important;
          user-select: none !important;
        }
        #luxar-stats canvas { outline: none !important; }
      `;
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
    getViewerContainer().appendChild(statsElement);
  }

  /**
   * Subscribe to frame-start / frame-end on the event bus so stats.js
   * gets driven by the animation loop. Idempotent — subsequent calls
   * are no-ops.
   */
  private subscribeToFrameTiming(): void {
    if (this.frameStartUnsubscribe) return;
    this.frameStartUnsubscribe = eventBus.on('frame-start', () => {
      this.stats.begin();
    });
    this.frameEndUnsubscribe = eventBus.on('frame-end', () => {
      this.stats.end();
    });
  }

  /** Drop the bus subscriptions. Idempotent. */
  private unsubscribeFromFrameTiming(): void {
    this.frameStartUnsubscribe?.();
    this.frameEndUnsubscribe?.();
    this.frameStartUnsubscribe = null;
    this.frameEndUnsubscribe = null;
  }

  /**
   * Toggle stats visibility. When shown, the panel subscribes to the
   * animation loop's frame-start / frame-end events; when hidden it
   * unsubscribes so stats.js incurs no cost.
   */
  toggle(): void {
    this.isVisible = !this.isVisible;
    this.stats.dom.style.display = this.isVisible ? 'block' : 'none';

    if (this.isVisible) {
      this.subscribeToFrameTiming();
    } else {
      this.unsubscribeFromFrameTiming();
    }

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
      const currentPanel = (this.stats.dom as HTMLElement & { panel?: number }).panel ?? 0;
      const nextPanel = (currentPanel + 1) % 3; // 0: fps, 1: ms, 2: mb
      this.stats.showPanel(nextPanel);
    }
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.unsubscribeFromFrameTiming();
    if (this.stats.dom.parentNode) {
      this.stats.dom.parentNode.removeChild(this.stats.dom);
    }
    // Clean up the injected style element
    const styleEl = document.getElementById('luxar-stats-custom-styles');
    if (styleEl) {
      styleEl.remove();
    }
  }
}
