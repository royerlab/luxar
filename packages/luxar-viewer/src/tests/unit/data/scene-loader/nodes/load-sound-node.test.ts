/**
 * `load-sound-node.ts` — the placeholder is attached before any await, the
 * descriptor carries positions + a store-bound clip reader, the clip is NOT
 * fetched at load, and a bad positions array surfaces as a `LoaderError`.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';

const openMock = vi.fn();
const readArrayMock = vi.fn();
vi.mock('../../../../../data/zarr', async () => {
  const actual = await vi.importActual<typeof import('../../../../../data/zarr')>(
    '../../../../../data/zarr'
  );
  return {
    ...actual,
    open: (...args: unknown[]) => openMock(...args),
    readArray: (...args: unknown[]) => readArrayMock(...args),
  };
});

import { loadSoundNode } from '../../../../../data/scene-loader/nodes/load-sound-node';
import { LoaderError } from '../../../../../data/scene-loader/nodes/load-leaf-error-dispatch';
import { makeTestNodeBuildCtx } from '../../../../helpers/make-test-node-build-ctx';
import type { NodeBuildCtx } from '../../../../../data/scene-loader/nodes/build-ctx';
import type { SceneNode } from '../../../../../data/data-loader-types';
import type { SoundSourceDescriptor } from '../../../../../types/audio';
import * as zarr from '../../../../../data/zarr';

function makeStore(bytes: Record<string, Uint8Array>): zarr.Readable & { gets: string[] } {
  const gets: string[] = [];
  return {
    gets,
    async get(key: string) {
      gets.push(key);
      return bytes[key];
    },
  } as unknown as zarr.Readable & { gets: string[] };
}

function makeNode(attrs: Record<string, unknown>): SceneNode {
  return {
    path: '/sounds/hum',
    type: 'sound',
    attrs: { type: 'sound', audio_file: 'audio.mp3', ...attrs } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [],
  };
}

function makeCtx(store: zarr.Readable): NodeBuildCtx {
  const applyTransform = vi.fn();
  return makeTestNodeBuildCtx({
    nodeFactory: { applyTransform } as unknown as NodeBuildCtx['nodeFactory'],
    factoryDeps: { zarrStore: store } as unknown as NodeBuildCtx['factoryDeps'],
  });
}

describe('loadSoundNode', () => {
  it('attaches a transformed placeholder and binds a lazy clip reader without fetching', async () => {
    const clip = new Uint8Array([0xff, 0xfb, 0x90, 0x00]);
    const store = makeStore({ '/sounds/hum/audio.mp3': clip });
    const ctx = makeCtx(store);
    const parent = new THREE.Group();
    const transform = new Array(16).fill(0);
    const node = makeNode({ has_positions: false, transform, trigger: 'continuous' });

    const placeholder = await loadSoundNode(node, parent, zarr.root(store), ctx);
    expect(parent.children[0]).toBe(placeholder);
    expect(placeholder.name).toBe('/sounds/hum');
    expect(placeholder.userData.nodeType).toBe('sound');
    expect(ctx.nodeFactory.applyTransform).toHaveBeenCalledWith(placeholder, transform);

    const desc = placeholder.userData.sound as SoundSourceDescriptor;
    expect(desc.name).toBe('hum');
    expect(desc.positions).toBeNull();
    expect(desc.nPositions).toBe(0);
    expect(store.gets).toEqual([]); // nothing fetched at load
    expect(await desc.readClip()).toBe(clip);
    expect(store.gets).toEqual(['/sounds/hum/audio.mp3']);
    expect(openMock).not.toHaveBeenCalled();
  });

  it('reads the (K, ndim) positions array into the descriptor', async () => {
    const store = makeStore({});
    openMock.mockResolvedValue({ shape: [2, 4] });
    readArrayMock.mockResolvedValue({
      data: Float32Array.from([1, 10, 20, 30, 2, 40, 50, 60]),
      shape: [2, 4],
      stride: [4, 1],
    });
    const node = makeNode({ has_positions: true, n_positions: 2, ndim: 4 });
    const placeholder = await loadSoundNode(
      node,
      new THREE.Group(),
      zarr.root(store),
      makeCtx(store)
    );
    const desc = placeholder.userData.sound as SoundSourceDescriptor;
    expect(desc.nPositions).toBe(2);
    expect(desc.ndim).toBe(4);
    expect(Array.from(desc.positions!)).toEqual([1, 10, 20, 30, 2, 40, 50, 60]);
  });

  it('a malformed positions array is a LoaderError, with the placeholder left in place', async () => {
    const store = makeStore({});
    openMock.mockResolvedValue({});
    readArrayMock.mockResolvedValue({ data: new Float32Array(3), shape: [3], stride: [1] });
    const parent = new THREE.Group();
    await expect(
      loadSoundNode(makeNode({ has_positions: true }), parent, zarr.root(store), makeCtx(store))
    ).rejects.toBeInstanceOf(LoaderError);
    expect(parent.children).toHaveLength(1);
  });

  it('a node without audio_file is refused as a validation error', async () => {
    const store = makeStore({});
    const node = makeNode({});
    delete (node.attrs as Record<string, unknown>).audio_file;
    await expect(
      loadSoundNode(node, new THREE.Group(), zarr.root(store), makeCtx(store))
    ).rejects.toMatchObject({ kind: 'Validation' });
  });
});
