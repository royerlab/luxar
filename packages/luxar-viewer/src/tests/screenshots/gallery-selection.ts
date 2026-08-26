/**
 * Resolve the gallery harness's `GALLERY_ONLY` selector without a browser.
 *
 * Most tokens are manifest ids. The reserved `readme` token derives the
 * front-page set from the media paths embedded in the repository README, so
 * the README remains the only source of truth for that curation.
 */

const README_GALLERY_MEDIA = /docs\/images\/readme\/gallery\/([A-Za-z0-9_-]+)\.(?:webp|webm)\b/g;

export interface GallerySelection {
  wantedIds: ReadonlySet<string>;
  unknownTokens: string[];
}

export function resolveGalleryOnly(
  only: string,
  readmeSource: string,
  manifestIds: readonly string[]
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
      Array.from(readmeSource.matchAll(README_GALLERY_MEDIA), (match) => match[1])
    );
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
