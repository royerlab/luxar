/**
 * Zip entry-name normalization for zipped Zarr stores.
 *
 * `ZipFileStore` looks a key up verbatim against the archive's entry names
 * (its `stripPrefix` only drops the leading `/` of the zarr key — it does NOT
 * strip a directory prefix). So the store's documents must sit at the archive
 * ROOT: `zarr.json`, `points/c/0`, …
 *
 * Two shapes are produced in the wild and only the first works unaided:
 *
 * - **flat** — what `zarr.storage.ZipStore` and `luxar optimise`'s packaging
 *   step write: members keyed store-relative.
 * - **nested** — what `zip -r scene.zip scene.luxar.zarr` or
 *   `shutil.make_archive` produce: every member prefixed `scene.luxar.zarr/`.
 *
 * A nested archive would otherwise miss on every single key and surface as an
 * empty scene with no error at all, which is the worst possible failure for
 * something a user just hand-rolled. This module detects the single-root case
 * and re-keys it, and refuses anything ambiguous with a message that says what
 * the archive actually contains.
 *
 * @module data/zarr/zip-entries
 */

/** Root documents that identify a zarr store: v3, then v2. */
const ROOT_DOCUMENTS = ['zarr.json', '.zgroup', '.zarray'] as const;

function hasRootDocument(names: Iterable<string>, prefix = ''): boolean {
  const wanted = new Set(ROOT_DOCUMENTS.map((doc) => `${prefix}${doc}`));
  for (const name of names) {
    if (wanted.has(name)) return true;
  }
  return false;
}

/**
 * Return `entries` keyed store-relative, stripping a single wrapping directory
 * when the archive has one.
 *
 * Entry VALUES are returned untouched — only the record's keys change, which is
 * all `ZipFileStore` looks at. The entry's own `name` still points at the real
 * member, so reads keep working.
 *
 * @throws If the archive holds no recognizable store, or more than one, since
 *   guessing between them would silently open the wrong data.
 */
export function normalizeZipEntries<T>(
  entries: Record<string, T>,
  archiveUrl: string
): Record<string, T> {
  const names = Object.keys(entries);

  // Already store-relative — the common case, and cheapest to check first.
  if (hasRootDocument(names)) return entries;

  const prefixes = new Set<string>();
  for (const name of names) {
    const slash = name.indexOf('/');
    if (slash > 0) prefixes.add(name.slice(0, slash + 1));
  }

  const withStore = [...prefixes].filter((prefix) => hasRootDocument(names, prefix));

  if (withStore.length === 1) {
    const prefix = withStore[0];
    const rekeyed: Record<string, T> = {};
    for (const [name, entry] of Object.entries(entries)) {
      if (name.startsWith(prefix)) rekeyed[name.slice(prefix.length)] = entry;
    }
    return rekeyed;
  }

  if (withStore.length > 1) {
    throw new Error(
      `The archive at ${archiveUrl} contains ${withStore.length} zarr stores ` +
        `(${withStore.map((p) => p.slice(0, -1)).join(', ')}). ` +
        'Point `?src=` at an archive holding exactly one store.'
    );
  }

  throw new Error(
    `The archive at ${archiveUrl} does not contain a zarr store: no ` +
      `${ROOT_DOCUMENTS.join(' / ')} at its root or one directory deep. ` +
      'A zipped scene must be created so that its members are keyed relative to ' +
      'the store root (what `zarr.storage.ZipStore` writes), not relative to the ' +
      'directory above it.'
  );
}

/**
 * Does this dataset URL name a zipped store?
 *
 * Query and fragment are ignored so a signed or parameterized URL
 * (`…/scene.luxar.zarr.zip?token=…`) still resolves correctly.
 */
export function isZippedStoreUrl(url: string): boolean {
  const withoutQuery = url.split(/[?#]/, 1)[0];
  return withoutQuery.toLowerCase().endsWith('.zip');
}
