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
 * Used for animation loops
 */
export function installAnimationFrameMock(): void {
  (globalThis as any).requestAnimationFrame = vi.fn((cb: any) => setTimeout(cb, 16));
  (globalThis as any).cancelAnimationFrame = vi.fn((id: any) => clearTimeout(id));
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
