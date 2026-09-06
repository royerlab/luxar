/**
 * Initial-load path for a single `sound` leaf node.
 *
 * A sound node is heard rather than drawn, so this is the smallest leaf loader:
 * it attaches a placeholder `THREE.Group` (with the node transform, so a spatial
 * voice parented under it inherits the scene-graph pose), reads the optional
 * `(K, ndim)` positions array, and binds a lazy clip reader to the scene's
 * store — the same raw `store.get()` the overlay manager's opaque-file reader
 * wraps. The clip bytes are NOT fetched here; the audio engine decodes them
 * after the scene attaches (`audio/audio-engine.ts`), so scene load stays fast
 * and a viewer that never creates an `AudioContext` never pays for a decode.
 *
 * No loader-registry registration: the node has no per-slice fetch, so it
 * takes no part in the update sweep and has no retry path. Errors surface
 * through `loadLeafNode` like every other leaf, so siblings still render.
 *
 * @module data/scene-loader/nodes/load-sound-node
 */

import * as THREE from 'three';
import * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';
import type { SceneNode } from '../../data-loader-types';
import type { SoundSourceDescriptor } from '../../../types/audio';
import { LoaderError, classifyLoaderError } from './load-leaf-error-dispatch';
import type { NodeBuildCtx } from './build-ctx';

/** Build the placeholder, read the positions, bind the clip reader. */
export async function loadSoundNode(
  node: SceneNode,
  parentThree: THREE.Object3D,
  loc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): Promise<THREE.Group> {
  log.custom('🔈', Modules.SCENE_LOADER, `Loading sound: ${node.path}`);
  const attrs = node.attrs as Record<string, unknown>;

  const placeholder = new THREE.Group();
  placeholder.name = node.path;
  placeholder.userData.nodeType = 'sound';
  placeholder.userData.attrs = node.attrs;
  if (node.attrs.transform) {
    ctx.nodeFactory.applyTransform(placeholder, node.attrs.transform);
  }
  // Attached BEFORE any await, like every other leaf's placeholder.
  parentThree.add(placeholder);

  const nodeLoc =
    node.path === '/' ? loc : zarr.root(ctx.factoryDeps.zarrStore).resolve(node.path.slice(1));
  const audioFile = typeof attrs.audio_file === 'string' ? attrs.audio_file : undefined;
  if (!audioFile) {
    throw new LoaderError(
      'Validation',
      node.path,
      new Error('sound node has no audio_file attr — was it written by luxar.add_sound?')
    );
  }

  let positions: Float32Array | null = null;
  let nPositions = 0;
  let ndim = typeof attrs.ndim === 'number' ? attrs.ndim : 0;
  if (attrs.has_positions === true) {
    try {
      const array = await zarr.open(nodeLoc.resolve('positions'), { kind: 'array' });
      const result = await zarr.readArray(array);
      if (result.shape.length !== 2 || result.shape[0] < 1) {
        throw new Error(`positions must be (K, ndim), got shape [${result.shape.join(', ')}]`);
      }
      [nPositions, ndim] = result.shape;
      positions = Float32Array.from(result.data as ArrayLike<number>);
    } catch (error) {
      throw new LoaderError(classifyLoaderError(error), node.path, error);
    }
  }

  const clipKey = nodeLoc.resolve(audioFile).path as zarr.AbsolutePath;
  const store = nodeLoc.store;
  const descriptor: SoundSourceDescriptor = {
    path: node.path,
    name: node.path.split('/').filter(Boolean).pop() ?? node.path,
    rawAttrs: attrs,
    positions,
    nPositions,
    ndim,
    readClip: () => Promise.resolve(store.get(clipKey)),
  };
  placeholder.userData.sound = descriptor;
  log.info(
    Modules.SCENE_LOADER,
    `  ${audioFile}, ${nPositions > 0 ? `${nPositions} position(s)` : 'non-spatial'}, ` +
      `trigger ${String(attrs.trigger ?? 'continuous')}, bus ${String(attrs.bus ?? 'ambient')}`
  );
  return placeholder;
}
