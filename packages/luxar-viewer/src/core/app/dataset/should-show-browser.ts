import { isZippedStoreUrl } from '../../../data/zip/entries';
import { classifyBrowserUrl } from './browser-decision';

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

/**
 * Whether {@link shouldShowBrowser} will send `src` straight to the loader
 * without a network probe — a non-directory URL whose path names a store
 * (`.zarr`) or an archive (`.zip`). Synchronous, so bootstrap can start the
 * root-document fetch before `app.init` for exactly the URLs the loader is
 * certain to open, and never for one that may end in the dataset browser.
 */
export function loadsDirectlyWithoutProbe(src: string | undefined | null): boolean {
  if (classifyBrowserUrl(src) === 'must-browse' || !src) return false;
  return isZippedStoreUrl(src) || isZarrStoreSuffix(src);
}

/**
 * Decide whether to open the dataset browser or load `src` directly.
 *
 * Order of checks:
 *   1. Synchronous URL classification — empty / trailing-slash URLs are
 *      always must-browse, no need to probe.
 *   2. Zipped stores load directly — see below.
 *   3. A `.zarr`-suffixed URL loads directly — see below.
 *   4. HEAD-probe the store markers ONE AT A TIME, format 3 first
 *      ({@link STORE_MARKERS}); the first 2xx returns `false` (load directly).
 *   5. If every probe fails, or one hangs past its own 5 s budget (the server
 *      is not answering, so the next marker would hang too), fall through to
 *      `true` (show the browser — likely a directory listing or non-zarr URL).
 *
 * Sequential, not parallel: each miss is a console `HEAD … 404` the user sees
 * (and reports), and the URLs that reach this probe are mostly suffix-less
 * store roots — `luxar serve` mounts the store at `/`, an exported folder at
 * `/data`. A parallel probe of all three markers logged two 404s on EVERY such
 * load of a format-3 store; v3-first logs none. A format-2 store pays one miss
 * (`zarr.json`) and one extra round trip, once, before its first byte; a
 * format-2 store with root attributes but no `.zgroup` pays two.
 */
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

  for (const marker of STORE_MARKERS) {
    const verdict = await probeMarker(`${src}/${marker}`);
    if (verdict === 'found') return false;
    if (verdict === 'timeout') break;
  }
  // No marker answered — likely a directory listing: show the browser.
  return true;
}

/**
 * The store-root documents that prove `src` is a zarr store, most likely
 * first: a format-3 root (`zarr.json`, what Luxar writes today), then a
 * format-2 group (`.zgroup`), then format-2 attributes alone (`.zattrs`).
 */
export const STORE_MARKERS = ['zarr.json', '.zgroup', '.zattrs'] as const;

/** How long ONE marker probe may take before it is abandoned. */
const PROBE_TIMEOUT_MS = 5000;

/**
 * HEAD `url` under its own {@link PROBE_TIMEOUT_MS} budget: `found` on 2xx,
 * `timeout` when it was abandoned, `missing` for any other answer or a
 * network error.
 */
async function probeMarker(url: string): Promise<'found' | 'missing' | 'timeout'> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
  try {
    const response = await fetch(url, { method: 'HEAD', signal: controller.signal });
    return response.ok ? 'found' : 'missing';
  } catch {
    return controller.signal.aborted ? 'timeout' : 'missing';
  } finally {
    clearTimeout(timeoutId);
  }
}
