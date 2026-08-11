/**
 * Browser tab title (`document.title`) for the viewer.
 *
 * Three sources, highest first: a scene's authored `viewer_config.title`, the
 * `?title=` URL parameter serve-family commands derive from the dataset file
 * name, and the page's own `<title>`. The last one is why this module exists —
 * switching datasets invalidates both of the others (`?title=` names the
 * dataset the server started with and is dropped from the address bar by
 * `buildDataSourceBrowserUrl`; an authored title names the scene being
 * replaced), so something has to remember what the tab was called before the
 * first override.
 */

/**
 * The page's own `<title>`, captured the first time we overwrite it. Left
 * `undefined` while nothing has overwritten the title — there is then nothing
 * to restore, because the page title is still in place.
 */
let pageTitle: string | undefined;

/** Archive wrappers a served store may arrive in, longest spelling first. */
const ARCHIVE_SUFFIXES = ['.tar.gz', '.tgz', '.zip'];

/** Dataset-store suffixes, longest spelling first. */
const STORE_SUFFIXES = ['.luxar.zarr', '.gsplats.zarr', '.zarr'];

/**
 * Set the browser tab title. A `null`/blank title restores the page's own
 * `<title>` rather than leaving the previous scene's name in place.
 */
export function setDocumentTitle(title: string | null | undefined): void {
  if (typeof document === 'undefined') return;
  const next = title?.trim();
  if (pageTitle === undefined) {
    if (!next) return; // nothing overwritten yet, so nothing to restore
    pageTitle = document.title ?? '';
  }
  document.title = next || pageTitle;
}

/**
 * A tab title derived from a dataset URL's file name, or `null` when the URL
 * does not name a store (a bare server root like `http://127.0.0.1:8000`, say,
 * whose last path segment is a host:port).
 *
 * The browser-side twin of `luxar.cli.utils.dataset_title`, which is what
 * spells `?title=` for the initially served dataset. This one covers what the
 * server cannot: the dataset the user picks in the viewer's own browser modal.
 * It is deliberately stricter — the server knows it was handed a dataset path,
 * whereas here an unrecognized name is better left to the page title.
 */
export function dataSourceDocumentTitle(src: string): string | null {
  const path = src.split(/[?#]/)[0].replace(/\/+$/, '');
  let name = path.slice(path.lastIndexOf('/') + 1);
  try {
    name = decodeURIComponent(name);
  } catch {
    // Malformed percent-escape — the raw segment is still a usable name.
  }
  for (const archive of ARCHIVE_SUFFIXES) {
    if (name.toLowerCase().endsWith(archive)) {
      name = name.slice(0, -archive.length);
      break;
    }
  }
  for (const store of STORE_SUFFIXES) {
    if (name.toLowerCase().endsWith(store)) {
      return name.slice(0, -store.length).trim() || null;
    }
  }
  return null;
}
