import { classifyBrowserUrl } from './browser-decision';

/**
 * Decide whether to open the dataset browser or load `src` directly.
 *
 * Order of checks:
 *   1. Synchronous URL classification — empty / trailing-slash URLs are
 *      always must-browse, no need to probe.
 *   2. HEAD-probe zarr v2 (`.zgroup`, `.zattrs`) and v3 (`zarr.json`)
 *      markers in parallel; short-circuit on the first 2xx response.
 *      Returns `false` (load directly) when any probe hits.
 *   3. If all probes fail or time out after 5s, fall through to `true`
 *      (show the browser — likely a directory listing or non-zarr URL).
 */
export async function shouldShowBrowser(src: string): Promise<boolean> {
  // Synchronous classification: empty / trailing-slash URLs always
  // need the browser, no point firing a zarr-metadata probe.
  if (classifyBrowserUrl(src) === 'must-browse') return true;

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
