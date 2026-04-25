/**
 * Centralized URL parameter parsing.
 *
 * `window.location.search` is read in exactly one place — `main.ts` — and the
 * result flows through the application as a typed object. Components that need
 * a flag declare it on their options, rather than reaching back to
 * `window.location` themselves.
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
  const raw =
    search ?? (typeof window !== 'undefined' ? window.location?.search : '') ?? '';
  const params = new URLSearchParams(raw);

  return {
    src: params.get('src'),
    theme: params.get('theme'),
    debug: params.has('debug'),
    noCache: params.has('no-cache'),
    cacheDebug: params.has('cache-debug'),
    clearCache: params.has('clear-cache'),
    noPrefetch: params.has('no-prefetch'),
    prefetchDebug: params.has('prefetch-debug'),
  };
}
