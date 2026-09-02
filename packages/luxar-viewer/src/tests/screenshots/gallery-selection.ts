/**
 * Resolve the gallery harness's `GALLERY_ONLY` selector without a browser.
 *
 * Most tokens are manifest ids. The reserved `readme` token derives the
 * front-page set from the media embedded in the repository README, so the
 * README remains the only source of truth for that curation.
 *
 * The README now points at content-addressed hosted media
 * (`https://data.luxarviewer.dev/media/<16hex>.<ext>`), whose keys carry no demo
 * id — the id lives in `scripts/gallery/media-manifest.json`. So the reserved
 * token resolves in two steps: scrape the keys, then map key -> id through that
 * manifest. The legacy in-repo path form is still recognised, so a README that
 * is only half migrated resolves what it can rather than silently shrinking the
 * set to nothing.
 */

/** Legacy in-repo tiles, whose filename stem WAS the demo id. */
const README_GALLERY_PATH = /docs\/images\/readme\/gallery\/([A-Za-z0-9_-]+)\.(?:webp|webm)\b/g;

function hostedGalleryPattern(baseUrl: string): RegExp {
  const escapedBaseUrl = baseUrl.replace(/\/+$/, '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`${escapedBaseUrl}/([A-Za-z0-9_-]+\\.(?:webp|webm))\\b`, 'g');
}

export interface GallerySelection {
  wantedIds: ReadonlySet<string>;
  unknownTokens: string[];
}

export function resolveGalleryOnly(
  only: string,
  readmeSource: string,
  manifestIds: readonly string[],
  /** Media key (e.g. `d9d1994630f8b126.webp`) -> demo id, from media-manifest.json. */
  mediaKeyToId: ReadonlyMap<string, string> = new Map(),
  mediaBaseUrl?: string
): GallerySelection {
  const manifestIdSet = new Set(manifestIds);
  const wantedIds = new Set<string>();
  const unknownTokens = new Set<string>();

  for (const token of only
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean)) {
    if (token !== 'readme') {
      if (manifestIdSet.has(token)) wantedIds.add(token);
      else unknownTokens.add(token);
      continue;
    }

    const readmeIds = new Set(
      Array.from(readmeSource.matchAll(README_GALLERY_PATH), (match) => match[1])
    );

    // Hosted media: the key is opaque, so an unmapped one means the README and
    // the media manifest have drifted. Report it rather than resolving the rest
    // — quietly dropping it would capture a smaller set than the README shows,
    // which is the failure this whole reserved token exists to prevent.
    const unmappedKeys: string[] = [];
    if (mediaBaseUrl !== undefined) {
      for (const match of readmeSource.matchAll(hostedGalleryPattern(mediaBaseUrl))) {
        const id = mediaKeyToId.get(match[1]);
        if (id === undefined) unmappedKeys.push(match[1]);
        else readmeIds.add(id);
      }
    }
    if (unmappedKeys.length > 0) {
      throw new Error(
        'README references hosted gallery media absent from ' +
          `scripts/gallery/media-manifest.json: ${[...new Set(unmappedKeys)].join(', ')}`
      );
    }

    if (readmeIds.size === 0) {
      throw new Error('GALLERY_ONLY=readme resolved to no gallery media in README.md');
    }

    const missingIds = [...readmeIds].filter((id) => !manifestIdSet.has(id));
    if (missingIds.length > 0) {
      throw new Error(
        `README gallery media ids are absent from the gallery manifest: ${missingIds.join(', ')}`
      );
    }
    for (const id of readmeIds) wantedIds.add(id);
  }

  return { wantedIds, unknownTokens: [...unknownTokens] };
}

/** Build the key -> id index from a parsed `media-manifest.json`. */
export function mediaKeyIndex(manifest: {
  base_url?: string;
  tiles?: Record<string, Record<string, { key: string }>>;
}): Map<string, string> {
  const index = new Map<string, string>();
  for (const [id, variants] of Object.entries(manifest.tiles ?? {})) {
    for (const entry of Object.values(variants)) index.set(entry.key, id);
  }
  return index;
}
