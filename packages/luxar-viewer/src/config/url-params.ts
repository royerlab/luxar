/**
 * Centralized URL parameter parsing.
 *
 * `window.location.search` is read in exactly one place — `main.ts` — and the
 * result flows through the application as a typed object. Components that need
 * a flag declare it on their options, rather than reaching back to
 * `window.location` themselves. URL writing is centralized here for the same
 * reason.
 *
 * This makes consumers testable (no need to mock `window.location`), the URL
 * contract auditable (every recognized parameter is listed in `UrlParams`),
 * and prepares the viewer for any context where `window.location` is not the
 * right source — embedded iframes, programmatic instantiation, SSR.
 */

/**
 * All URL parameters recognized by the viewer.
 *
 * Every consumer that wants a URL-derived value should accept the relevant
 * field via constructor/init options rather than read `window.location`.
 */
const MAX_SRC_LENGTH = 4096;
const URL_SCHEME_PATTERN = /^[a-zA-Z][a-zA-Z\d+.-]*:/;
function hasUnsafeSrcCharacter(src: string): boolean {
  for (const char of src) {
    const code = char.charCodeAt(0);
    if (code <= 0x1f || code === 0x7f || char === '<' || char === '>') {
      return true;
    }
  }
  return false;
}

/**
 * Validate and normalize a data-source URL from user-controlled input.
 *
 * Accepted forms:
 * - Absolute HTTP(S) URLs: `https://example.com/data.zarr`
 * - Root-relative paths: `/datasets/data.zarr`
 * - Relative paths: `datasets/data.zarr`
 *
 * Rejected forms include unsupported schemes (`file:`, `javascript:`,
 * `data:`), protocol-relative URLs (`//host/path`), control characters,
 * obvious HTML delimiters, empty strings, and excessively long values.
 */
export function normalizeDataSourceUrl(rawSrc: string | null): string | null {
  if (rawSrc === null) return null;

  const src = rawSrc.trim();
  if (src.length === 0 || src.length > MAX_SRC_LENGTH) return null;
  if (hasUnsafeSrcCharacter(src)) return null;
  if (src.startsWith('//')) return null;

  const hasScheme = URL_SCHEME_PATTERN.test(src);
  if (!hasScheme) return src;

  try {
    const url = new URL(src);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
    return src;
  } catch {
    return null;
  }
}

export interface UrlParams {
  /** Dataset source URL (`?src=...`). Null when not provided. */
  src: string | null;
  /** Theme override (`?theme=light` etc). Null when not provided. */
  theme: string | null;
  /** Enable the `window.__luxarDebug` interface (`?debug`). */
  debug: boolean;
  /** Disable all cache layers (`?no-cache`). */
  noCache: boolean;
  /** Verbose cache logging (`?cache-debug`). */
  cacheDebug: boolean;
  /** Clear caches on init (`?clear-cache`). */
  clearCache: boolean;
  /** Disable adjacent-chunk prefetching (`?no-prefetch`). */
  noPrefetch: boolean;
  /** Verbose prefetch logging (`?prefetch-debug`). */
  prefetchDebug: boolean;
}

/**
 * Parse the supplied query string (or `window.location.search` by default)
 * into a typed `UrlParams` snapshot.
 *
 * Pass an explicit `search` string in tests; in production main.ts calls this
 * once with no argument and threads the result through the rest of the app.
 */
export function readUrlParams(search?: string): UrlParams {
  const raw = search ?? (typeof window !== 'undefined' ? window.location?.search : '') ?? '';
  const params = new URLSearchParams(raw);

  return {
    src: normalizeDataSourceUrl(params.get('src')),
    theme: params.get('theme'),
    debug: params.has('debug'),
    noCache: params.has('no-cache'),
    cacheDebug: params.has('cache-debug'),
    clearCache: params.has('clear-cache'),
    noPrefetch: params.has('no-prefetch'),
    prefetchDebug: params.has('prefetch-debug'),
  };
}

export interface BrowserUrlLocation {
  pathname: string;
  search: string;
  hash?: string;
}

export interface BrowserUrlHistory {
  replaceState(data: unknown, unused: string, url?: string | URL | null): void;
}

export interface BrowserUrlWriter {
  location: BrowserUrlLocation;
  history: BrowserUrlHistory;
}

/**
 * Strip trailing slashes from a `src` value so the viewer's downstream zarr
 * fetches don't accumulate `//` from the data root. The viewer treats trailing
 * slashes as an empty path component and they cause 404s on the loader; the
 * URL contract for `?src=` is "no trailing slash".
 */
function normalizeSrcForUrl(src: string): string {
  return src.replace(/\/+$/, '');
}

/**
 * Build a URL path for the current viewer page with `src` updated.
 *
 * This helper preserves existing query parameters and hash fragments while
 * centralizing the viewer's URL-writing contract. It returns a path-relative
 * URL suitable for `history.replaceState()`. The `src` is normalized so it
 * never carries a trailing slash.
 */
export function buildDataSourceBrowserUrl(src: string, location: BrowserUrlLocation): string {
  const params = new URLSearchParams(location.search);
  params.set('src', normalizeSrcForUrl(src));
  const query = params.toString();
  const hash = location.hash ?? '';
  return `${location.pathname}${query ? `?${query}` : ''}${hash}`;
}

/**
 * Replace the current browser URL with the selected dataset source.
 *
 * Returns `false` if the environment has no browser history/location or if
 * `history.replaceState()` is blocked, e.g. by a sandboxed iframe. Callers
 * should treat failure as non-fatal and continue loading the selected dataset.
 * The `src` is always normalized to drop trailing slashes — callers do not
 * need to pre-normalize.
 */
export function replaceBrowserDataSourceUrl(src: string, target?: BrowserUrlWriter): boolean {
  const writer = target ?? (typeof window !== 'undefined' ? window : undefined);
  if (!writer?.location || !writer.history) return false;

  try {
    writer.history.replaceState({}, '', buildDataSourceBrowserUrl(src, writer.location));
    return true;
  } catch {
    return false;
  }
}
