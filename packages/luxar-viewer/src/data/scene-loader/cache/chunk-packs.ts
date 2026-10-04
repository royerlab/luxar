/**
 * Hand a packed store's `chunk_packs` index (`luxar optimize --pack`) to the
 * {@link PackedChunkSource} under the caching store.
 *
 * Trust the consolidated index only when it came from the network. A warm
 * root index can predate a pack rebuild that kept `content_hash`, so in that
 * case read the sidecar directly from the source store. Packs are optional;
 * any doubt leaves every read on the plain path.
 *
 * @module data/scene-loader/cache/chunk-packs
 */

import * as zarr from '../../zarr';
import { hasContentsMethod } from '../../../types/zarr';
import { log, Modules } from '../../../utils/log';
import type { PackedChunkSource } from '../../../cache/chunk-source/packed-chunk-source';

/** The root-level sidecar group (Python: `CHUNK_PACKS_GROUP`). */
export const CHUNK_PACKS_GROUP = 'chunk_packs';

/** Adopt the store's packs for `contentHash`; returns how many are in use. */
export async function adoptChunkPacks(
  packs: PackedChunkSource | null,
  rootLoc: zarr.Location<zarr.Readable>,
  contentHash: unknown,
  indexFresh: boolean,
  sourceRootLoc: zarr.Location<zarr.Readable>
): Promise<number> {
  if (!packs) return 0;
  try {
    const listed =
      indexFresh && hasContentsMethod(rootLoc.store)
        ? (await rootLoc.store.contents()).some(
            (entry) =>
              entry.path.replace(/^\/+/, '') === CHUNK_PACKS_GROUP && entry.kind === 'group'
          )
        : undefined;
    if (indexFresh && listed === false) return 0;
    const location = indexFresh && listed === true ? rootLoc : sourceRootLoc;
    const group = await zarr.openGroupPreferV3(location.resolve(CHUNK_PACKS_GROUP));
    const used = packs.usePacks(group.attrs, contentHash);
    log.info(Modules.SCENE_LOADER, `chunk packs: ${used} in use`);
    return used;
  } catch (error) {
    if (zarr.isNotFoundError(error)) return 0;
    log.warning(Modules.SCENE_LOADER, 'chunk packs unreadable; reading chunks plainly', error);
    return 0;
  }
}
