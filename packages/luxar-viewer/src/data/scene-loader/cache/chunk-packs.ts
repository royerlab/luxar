/**
 * Hand a packed store's `chunk_packs` index (`luxar optimize --pack`) to the
 * {@link PackedChunkSource} under the caching store.
 *
 * Read from the consolidated index the store was opened with, so an unpacked
 * store costs no request: the sidecar group is opened only when the index
 * lists it. Packs are an optional transport optimisation, so nothing here can
 * fail a load — any doubt leaves every read on the plain path.
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
  contentHash: unknown
): Promise<number> {
  if (!packs || !hasContentsMethod(rootLoc.store)) return 0;
  try {
    const contents = await rootLoc.store.contents();
    const listed = contents.some(
      (entry) => entry.path.replace(/^\/+/, '') === CHUNK_PACKS_GROUP && entry.kind === 'group'
    );
    if (!listed) return 0;
    const group = await zarr.open(rootLoc.resolve(CHUNK_PACKS_GROUP), { kind: 'group' });
    const used = packs.usePacks(group.attrs, contentHash);
    log.info(Modules.SCENE_LOADER, `chunk packs: ${used} in use`);
    return used;
  } catch (error) {
    log.warning(Modules.SCENE_LOADER, 'chunk packs unreadable; reading chunks plainly', error);
    return 0;
  }
}
