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
 *
 * No-op outside a DOM environment: this hangs the mock off `window`, which
 * does not exist under `environment: 'node'`, and the unguarded assignment
 * threw `TypeError: Cannot set properties of undefined`. Anything that reads
 * `matchMedia` needs a document anyway, so those files carry a
 * `@vitest-environment jsdom` docblock and still get the mock.
 */
export function installMatchMediaMock(): void {
  if (typeof window === 'undefined') return;
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
  // Map from our rAF ID → the real setTimeout ID (needed for proper cancellation)
  const pendingFrames = new Map<number, ReturnType<typeof setTimeout>>();
  let frameIdCounter = 1;

  (globalThis as any).requestAnimationFrame = vi.fn((cb: any) => {
    const id = frameIdCounter++;
    const timerId = setTimeout(() => {
      pendingFrames.delete(id);
      // Wrap callback in try-catch to prevent test environment errors
      try {
        if (typeof requestAnimationFrame === 'function') {
          cb(performance.now());
        }
      } catch {
        // Silently ignore errors after test teardown
      }
    }, 16);
    pendingFrames.set(id, timerId);
    return id;
  });

  (globalThis as any).cancelAnimationFrame = vi.fn((id: any) => {
    const timerId = pendingFrames.get(id);
    if (timerId !== undefined) {
      clearTimeout(timerId);
      pendingFrames.delete(id);
    }
  });

  // Global cleanup helper (can be called in afterEach)
  (globalThis as any).__clearAllAnimationFrames = () => {
    pendingFrames.forEach((timerId) => clearTimeout(timerId));
    pendingFrames.clear();
  };
}

/**
 * Mock performance.now
 *
 * Used for high-precision timing
 */
export function installPerformanceMock(): void {
  const existing = (globalThis as any).performance;
  const now = vi.fn(() => Date.now());
  if (!existing) {
    (globalThis as any).performance = { now };
    return;
  }
  // Override ONLY `now`, keeping the rest of the real Performance object.
  //
  // Replacing the whole global wholesale used to be harmless under jsdom, but
  // under `environment: 'node'` it breaks `fetch`: undici calls
  // `performance.markResourceTiming` when a response body finishes, and a
  // `{ now }` stub does not have it — which surfaced as 10 `TypeError:
  // markResourceTiming is not a function` unhandled errors from
  // config/e2e-server-identity.test.ts. `now` is a non-writable accessor on the
  // real object, so it needs defineProperty rather than assignment.
  Object.defineProperty(existing, 'now', { value: now, configurable: true, writable: true });
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
