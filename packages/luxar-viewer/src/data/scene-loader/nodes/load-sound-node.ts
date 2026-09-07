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

/** The placeholder group every sound node hangs its voices from. */
function buildPlaceholder(
  node: SceneNode,
  parentThree: THREE.Object3D,
  ctx: NodeBuildCtx
): THREE.Group {
  const placeholder = new THREE.Group();
  placeholder.name = node.path;
  placeholder.userData.nodeType = 'sound';
  placeholder.userData.attrs = node.attrs;
  if (node.attrs.transform) {
    ctx.nodeFactory.applyTransform(placeholder, node.attrs.transform);
  }
  // Attached BEFORE any await, like every other leaf's placeholder.
  parentThree.add(placeholder);
  return placeholder;
}

interface SoundPositions {
  positions: Float32Array | null;
  nPositions: number;
  ndim: number;
}

/** The optional `(K, ndim)` positions rows; absent for a clip live everywhere. */
async function readPositions(
  nodeLoc: zarr.Location<zarr.Readable>,
  attrs: Record<string, unknown>,
  path: string
): Promise<SoundPositions> {
  const ndim = typeof attrs.ndim === 'number' ? attrs.ndim : 0;
  if (attrs.has_positions !== true) return { positions: null, nPositions: 0, ndim };
  try {
    const array = await zarr.open(nodeLoc.resolve('positions'), { kind: 'array' });
    const result = await zarr.readArray(array);
    if (result.shape.length !== 2 || result.shape[0] < 1) {
      throw new Error(`positions must be (K, ndim), got shape [${result.shape.join(', ')}]`);
    }
    return {
      positions: Float32Array.from(result.data as ArrayLike<number>),
      nPositions: result.shape[0],
      ndim: result.shape[1],
    };
  } catch (error) {
    throw new LoaderError(classifyLoaderError(error), path, error);
  }
}

/** A string attr for the log line, or its default. */
function attrLabel(value: unknown, fallback: string): string {
  return typeof value === 'string' ? value : fallback;
}

/** Build the placeholder, read the positions, bind the clip reader. */
export async function loadSoundNode(
  node: SceneNode,
  parentThree: THREE.Object3D,
  loc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): Promise<THREE.Group> {
  log.custom('🔈', Modules.SCENE_LOADER, `Loading sound: ${node.path}`);
  const attrs = node.attrs as Record<string, unknown>;
  const placeholder = buildPlaceholder(node, parentThree, ctx);

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
  const { positions, nPositions, ndim } = await readPositions(nodeLoc, attrs, node.path);

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
  const where = nPositions > 0 ? `${nPositions} position(s)` : 'non-spatial';
  log.info(
    Modules.SCENE_LOADER,
    `  ${audioFile}, ${where}, trigger ${attrLabel(attrs.trigger, 'continuous')}, ` +
      `bus ${attrLabel(attrs.bus, 'ambient')}`
  );
  return placeholder;
}
