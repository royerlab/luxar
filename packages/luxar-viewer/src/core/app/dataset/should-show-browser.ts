import { isZippedStoreUrl } from '../../../data/zip/entries';
import { classifyBrowserUrl } from './browser-decision';

/**
 * Decide whether to open the dataset browser or load `src` directly.
 *
 * Order of checks:
 *   1. Synchronous URL classification — empty / trailing-slash URLs are
 *      always must-browse, no need to probe.
 *   2. Zipped stores load directly — see below.
 *   3. A `.zarr`-suffixed URL loads directly — see below.
 *   4. HEAD-probe zarr v2 (`.zgroup`, `.zattrs`) and v3 (`zarr.json`)
 *      markers in parallel; short-circuit on the first 2xx response.
 *      Returns `false` (load directly) when any probe hits.
 *   5. If all probes fail or time out after 5s, fall through to `true`
 *      (show the browser — likely a directory listing or non-zarr URL).
 */
/**
 * Whether the URL's PATH ends in `.zarr` — the convention every Luxar store
 * follows (`scene.luxar.zarr`, `fit.gsplats.zarr`).
 *
 * Tested on the pathname, not the raw string, so a presigned/tokenized source
 * (`…/scene.luxar.zarr?token=…`) is still recognized. Falls back to the raw
 * string for anything `URL` cannot parse (relative `?src=` values).
 */
function isZarrStoreSuffix(src: string): boolean {
  try {
    return new URL(src).pathname.endsWith('.zarr');
  } catch {
    return src.endsWith('.zarr');
  }
}

export async function shouldShowBrowser(src: string): Promise<boolean> {
  // Synchronous classification: empty / trailing-slash URLs always
  // need the browser, no point firing a zarr-metadata probe.
  if (classifyBrowserUrl(src) === 'must-browse') return true;

  // A `.zarr.zip` is a FILE whose store documents live INSIDE it, so the
  // child probes below are meaningless: `archive.zip/zarr.json` 404s for
  // every archive. Decide from the suffix and let the loader's zip store do
  // the real work.
  if (isZippedStoreUrl(src)) return false;

  // Same reasoning one step further: a `.zarr` suffix (so `.luxar.zarr`,
  // `.gsplats.zarr`) with no trailing slash NAMES a store, and the three HEAD
  // probes below only re-confirm what the suffix already says — at the cost of
  // a full round trip on the critical path, before scene loading may even
  // begin. `classifyBrowserUrl` has already sent every trailing-slash URL to
  // the browser above, so a *directory* called `foo.zarr/` is unaffected; this
  // only ever sees non-directory URLs.
  //
  // The trade: a dead or mistyped `…/typo.zarr` now surfaces a load error
  // instead of silently opening the dataset browser. That is the more honest
  // failure for a URL that explicitly names a store.
  if (isZarrStoreSuffix(src)) return false;

  // Check if it's a Zarr dataset by looking for zarr metadata files
  // Try both v2 (.zgroup) and v3 (zarr.json) formats
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);

  const zarrChecks = [
    fetch(src + '/.zgroup', { method: 'HEAD', signal: controller.signal }),
    fetch(src + '/.zattrs', { method: 'HEAD', signal: controller.signal }),
    fetch(src + '/zarr.json', { method: 'HEAD', signal: controller.signal }),
  ];

  try {
    // Short-circuit: return as soon as any probe confirms zarr metadata exists.
    // This avoids waiting for the zarr.json 404 on v2 stores (~200ms on slow networks).
    await Promise.any(
      zarrChecks.map((p) =>
        p.then((r) => {
          if (!r.ok) throw new Error('not ok');
          return r;
        })
      )
    );
    return false; // At least one zarr metadata file exists — load directly
  } catch {
    // All probes failed or errored — likely a directory, show browser
  } finally {
    clearTimeout(timeoutId);
  }

  return true;
}
