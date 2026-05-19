/**
 * Enumerate the entries of a zarr store — pure data-fetch primitive
 * used by `buildSceneGraph`. Pulls from the consolidated-metadata
 * `contents()` method when the store supports it; otherwise returns a
 * single-root fallback entry that lets the loader still construct a
 * root scene node (subsequent loaders may then fan out by path).
 */

import * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';
import { hasContentsMethod } from '../../../types/zarr';

/**
 * Enumerate all groups and arrays in the store. Returns an empty array
 * when the store is null (e.g. `loadScene` called before init).
 */
export async function enumerateStore(
  store: zarr.Readable | null
): Promise<Array<{ path: string; kind: string }>> {
  if (!store) return [];

  // Try to use consolidated metadata
  if (hasContentsMethod(store)) {
    const contents = await store.contents();
    log.custom('📋', Modules.SCENE_LOADER, `Found ${contents.length} items in store`);
    return contents;
  }

  // Fallback enumeration
  log.warning(Modules.SCENE_LOADER, 'Store does not support contents(), using fallback');
  return [{ path: '/', kind: 'group' }];
}
