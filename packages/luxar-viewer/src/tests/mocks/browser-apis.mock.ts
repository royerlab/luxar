/**
 * Browser APIs Mock
 *
 * Mocks for browser APIs that don't exist in Node.js test environment:
 * - window.matchMedia (for media queries and HDR detection)
 * - ResizeObserver (for responsive layout)
 * - IntersectionObserver (for visibility tracking)
 * - requestAnimationFrame (for animation loops)
 * - performance.now (for timing)
 */

import { vi } from 'vitest';

/**
 * Mock window.matchMedia
 *
 * Used for HDR detection and color space queries
 */
export function installMatchMediaMock(): void {
  (globalThis as any).window.matchMedia = vi.fn().mockImplementation((query: string) => ({
    matches: false, // Default to false (no HDR/P3 support in tests)
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }));
}

/**
 * Mock ResizeObserver
 *
 * Used for responsive canvas sizing
 */
export function installResizeObserverMock(): void {
  (globalThis as any).ResizeObserver = vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  }));
}

/**
 * Mock IntersectionObserver
 *
 * Used for visibility detection
 */
export function installIntersectionObserverMock(): void {
  (globalThis as any).IntersectionObserver = vi.fn().mockImplementation(() => ({
    observe: vi.fn(),
    unobserve: vi.fn(),
    disconnect: vi.fn(),
  }));
}

/**
 * Mock requestAnimationFrame and cancelAnimationFrame
 *
 * Used for animation loops. Tracks pending frames for proper cleanup.
 */
export function installAnimationFrameMock(): void {
  const pendingFrames = new Set<number>();
  let frameIdCounter = 1;

  (globalThis as any).requestAnimationFrame = vi.fn((cb: any) => {
    const id = frameIdCounter++;
    setTimeout(() => {
      pendingFrames.delete(id);
      // Wrap callback in try-catch to prevent test environment errors
      try {
        cb(performance.now());
      } catch (error) {
        // Silently ignore errors after test teardown
        if (error && (error as any).message?.includes('test environment')) {
          return;
        }
        throw error;
      }
    }, 16);
    pendingFrames.add(id);
    return id;
  });

  (globalThis as any).cancelAnimationFrame = vi.fn((id: any) => {
    if (pendingFrames.has(id)) {
      clearTimeout(id);
      pendingFrames.delete(id);
    }
  });

  // Global cleanup helper (can be called in afterEach)
  (globalThis as any).__clearAllAnimationFrames = () => {
    pendingFrames.forEach((id) => clearTimeout(id));
    pendingFrames.clear();
  };
}

/**
 * Mock performance.now
 *
 * Used for high-precision timing
 */
export function installPerformanceMock(): void {
  (globalThis as any).performance = {
    now: vi.fn(() => Date.now()),
  };
}

/**
 * Install all browser API mocks
 *
 * Convenience function to install all browser mocks at once
 */
export function installAllBrowserMocks(): void {
  installMatchMediaMock();
  installResizeObserverMock();
  installIntersectionObserverMock();
  installAnimationFrameMock();
  installPerformanceMock();
}
