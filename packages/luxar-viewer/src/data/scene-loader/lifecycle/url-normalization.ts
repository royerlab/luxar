/**
 * URL normalization helpers for the scene loader.
 *
 * Pure: turns user-supplied dataset URLs into the absolute, slash-
 * terminated form the scene loader's INTERNAL consumers expect.
 * Centralized here so the various edge cases (relative paths, missing
 * protocol, missing trailing slash) get a single canonical form.
 *
 * The original implementation used `window.location.origin` directly;
 * this version takes the origin as a parameter so it's testable
 * without a DOM.
 *
 * # Contract: trailing slash is INTENTIONAL (HIGH-6 audit)
 *
 * `normalizeURL` ALWAYS returns a URL ending in `/`. This is the inverse
 * of `config/url-params.ts::normalizeDataSourceUrl`, which STRIPS the
 * trailing slash from user input per the CLAUDE.md gotcha ("Data Source URLs
 * Normalize Trailing Slashes"). The two layers run in sequence:
 *
 *   user `?src=...`  →  normalizeDataSourceUrl (strip `/`)
 *                    →  scene loader stores raw
 *                    →  normalizeURL (add `/` back, absolute-ify)
 *                    →  internal consumers
 *
 * The trailing slash here is required because at least one downstream
 * consumer — `ui/overlay-manager.ts::renderImageOverlay` — builds child
 * URLs by raw string concatenation:
 *
 *     const imageUrl = `${this.baseUrl}overlays/${name}/${file}`;
 *
 * Without the trailing slash, that concatenates to `…/dataset.zarroverlays/...`
 * and 404s. Other downstream consumers tolerate the trailing slash:
 *
 *   - zarrita's `FetchStore.resolve()` explicitly appends `/` if missing
 *     (see `@zarrita/storage` `dist/src/fetch.js`).
 *   - `cache/multi-level-caching-store/fetch-retry.ts::buildUrl` strips
 *     `/+$` off the base before joining (`replace(/\/+$/, '')`).
 *
 * Result: producers that need a child-buildable base (overlay-manager)
 * get one; producers that prefer slash-stripped (the underlying network
 * layer) get one via their own normalization. Do NOT remove the trailing
 * slash without first auditing every consumer of
 * `rootGroup.userData.zarrBaseUrl` and every `ctx.normalizeURL` call site.
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
 * The trailing slash on the output is part of the function's contract —
 * see the module docstring for the rationale (downstream string-concat
 * consumers).
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
