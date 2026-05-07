/**
 * Pure URL helpers for the dataset browser. Extracted from
 * dataset-browser.ts so the parsing logic is unit-testable without
 * instantiating a DatasetBrowser (which needs a DOM container plus a
 * DirectoryNavigator).
 *
 * @module ui/panels/dataset-url-utils
 */

/**
 * Compute the "base URL" the browser should start at, given a current
 * dataset URL (which may itself end in `.zarr` and therefore live one
 * directory below the listing target).
 *
 * Logic:
 *   - empty / falsy URL: return `${origin}/` (top of the configured
 *     server).
 *   - URL ending in `.zarr` or `.zarr/`: strip the trailing dataset
 *     segment so we list the directory containing the dataset.
 *   - otherwise: return `origin + pathname` unchanged.
 *   - on parse failure: return the input verbatim (defensive — the
 *     caller does not need to handle the throw).
 *
 * @param url - The current `?src=...` URL, possibly empty.
 * @param origin - Fallback origin when `url` is empty.
 */
export function extractBaseUrl(url: string, origin: string): string {
  if (!url) return origin + '/';

  try {
    const parsed = new URL(url);
    let pathname = parsed.pathname;
    if (pathname.endsWith('.zarr') || pathname.endsWith('.zarr/')) {
      const parts = pathname.split('/').filter(Boolean);
      parts.pop();
      pathname = '/' + parts.join('/') + '/';
    }
    return parsed.origin + pathname;
  } catch {
    return url;
  }
}

/**
 * Extract the relative dataset path from a full URL — the path-portion
 * up to and including the first `.zarr` segment. Used to pre-populate
 * the breadcrumb when re-opening the browser on a dataset that's
 * already loaded.
 *
 * Returns an empty string when:
 *   - the URL is empty / falsy;
 *   - parsing fails (defensive);
 *   - no `.zarr` segment is found in the URL path.
 */
export function extractPath(url: string): string {
  if (!url) return '';

  try {
    const parsed = new URL(url);
    const pathname = parsed.pathname;

    if (pathname.includes('.zarr')) {
      const parts = pathname.split('/').filter(Boolean);
      const zarrIndex = parts.findIndex((p) => p.endsWith('.zarr'));
      if (zarrIndex >= 0) {
        return parts.slice(0, zarrIndex + 1).join('/');
      }
    }

    return '';
  } catch {
    return '';
  }
}
