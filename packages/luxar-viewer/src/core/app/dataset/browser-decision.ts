/**
 * Pure URL classification for the "should we show the dataset browser
 * panel before trying to load this URL?" decision.
 *
 * Extracted from `LuxarApp.shouldShowBrowser` so the synchronous
 * decision (empty / trailing-slash → must browse) can be tested
 * without fetch. The async zarr-metadata probe in app.ts calls this
 * first and only falls through to the network path when this returns
 * false.
 *
 * @module core/app/dataset/browser-decision
 */

/**
 * Classification of a URL with respect to the dataset browser:
 *
 * - `'must-browse'`: URL is empty or a directory (trailing slash).
 *   The browser panel must open; no point doing a zarr probe.
 * - `'maybe-zarr'`: URL has a non-empty path with no trailing slash.
 *   Caller should probe `.zgroup` / `.zattrs` / `zarr.json` to decide
 *   between "load directly" and "fall back to browser".
 */
export type BrowserDecision = 'must-browse' | 'maybe-zarr';

/**
 * Synchronous classification step for {@link shouldShowBrowser}.
 *
 * Defensive on whitespace-only input (treated as empty). Anything else
 * with a trailing slash is a directory URL the browser panel can list.
 * URLs without a trailing slash *might* be a Zarr dataset — the caller
 * runs a HEAD probe to confirm.
 */
export function classifyBrowserUrl(src: string | undefined | null): BrowserDecision {
  if (!src || src.trim() === '') return 'must-browse';
  if (src.endsWith('/')) return 'must-browse';
  return 'maybe-zarr';
}
