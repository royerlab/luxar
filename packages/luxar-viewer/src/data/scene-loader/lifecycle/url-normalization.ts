/**
 * URL normalization helpers for the scene loader.
 *
 * Pure: turns user-supplied dataset URLs into the absolute, slash-
 * terminated form the zarr loader expects. Centralized here so the
 * various edge cases (relative paths, missing protocol, missing
 * trailing slash) get a single canonical form.
 *
 * The original implementation used `window.location.origin` directly;
 * this version takes the origin as a parameter so it's testable
 * without a DOM.
 *
 * @module data/scene-loader/lifecycle/url-normalization
 */

/**
 * Normalize a dataset URL to absolute, slash-terminated form.
 *
 * - Absolute URLs (http://, https://, case-insensitive): return as-is,
 *   ensuring a trailing slash. The case-insensitive match prevents a
 *   mixed-case `HTTPS://...` (which `normalizeDataSourceUrl` already
 *   accepts via `URL.protocol`) from being treated as a relative path
 *   here.
 * - Relative paths: prepend the supplied `windowOrigin`, ensuring a leading
 *   slash on the path and a trailing slash on the result.
 *
 * Pure given an explicit origin.
 */
export function normalizeURL(url: string, windowOrigin: string): string {
  if (/^https?:\/\//i.test(url)) {
    return url.endsWith('/') ? url : url + '/';
  }
  const cleanPath = url.startsWith('/') ? url : '/' + url;
  return windowOrigin + cleanPath + (cleanPath.endsWith('/') ? '' : '/');
}
