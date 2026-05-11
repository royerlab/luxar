/**
 * Tiny platform-detection helpers, centralized so callers don't sprinkle
 * `navigator.platform` checks across the codebase and so tests can stub a
 * single export.
 *
 * @module utils/platform
 */

/**
 * Whether the current browser is running on macOS.
 *
 * Uses `navigator.platform.startsWith('Mac')`. `navigator.platform` is
 * formally deprecated in favor of `navigator.userAgentData`, but it remains
 * universally available and is honest about the OS string ("MacIntel",
 * "MacPPC", "iPhone" → starts with "Mac" only on desktop macOS). If/when we
 * adopt `navigator.userAgentData`, update this one function.
 *
 * Returns `false` in non-browser contexts (e.g., Node test runners) so
 * default-derivation code paths remain deterministic outside the browser.
 */
export function isMacPlatform(): boolean {
  return typeof navigator !== 'undefined' && navigator.platform.startsWith('Mac');
}
