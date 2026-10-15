/**
 * Hand a packed store's `chunk_packs` index (`luxar optimize --pack`) to the
 * {@link PackedChunkSource} under the caching store.
 *
 * Trust a consolidated index only when it came from the network this load —
 * the opened store's own, or the load-time root-document fetch's (see
 * `root-listing.ts`). A warm root index can predate a pack rebuild that kept
 * `content_hash`, so then read the sidecar directly from the source store.
 * Packs are optional; any doubt leaves every read on the plain path.
 *
 * @module data/scene-loader/cache/chunk-packs
 */

import * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';
import type { PackedChunkSource } from '../../../cache/chunk-source/packed-chunk-source';
import type { RootListing } from './root-listing';

/** The root-level sidecar group (Python: `CHUNK_PACKS_GROUP`). */
export const CHUNK_PACKS_GROUP = 'chunk_packs';

/**
 * Adopt the store's packs for `contentHash`; returns how many are in use.
 *
 * `listing` can answer from the load-time root-document fetch on a warm load,
 * and carries the store's format so a probe asks for one format's document.
 */
export async function adoptChunkPacks(
  packs: PackedChunkSource | null,
  rootLoc: zarr.Location<zarr.Readable>,
  contentHash: unknown,
  listing: RootListing,
  sourceRootLoc: () => zarr.Location<zarr.Readable>
): Promise<number> {
  if (!packs) return 0;
  try {
    const listed = listing.lists(CHUNK_PACKS_GROUP);
    // A current index without the group: there are no packs. No request.
    if (listed === false) return 0;
    // Read the sidecar through the opened store only when its OWN index is
    // current and lists it; otherwise from the source, past any warm tier
    // whose copy may predate a pack rebuild that kept `content_hash`.
    const location = listing.fromOpenedIndex && listed === true ? rootLoc : sourceRootLoc();
    const group = await zarr.openGroupOfFormat(location.resolve(CHUNK_PACKS_GROUP), listing.format);
    const used = packs.usePacks(group.attrs, contentHash);
    log.info(Modules.SCENE_LOADER, `chunk packs: ${used} in use`);
    return used;
  } catch (error) {
    if (zarr.isNotFoundError(error)) return 0;
    log.warning(Modules.SCENE_LOADER, 'chunk packs unreadable; reading chunks plainly', error);
    return 0;
  }
}
