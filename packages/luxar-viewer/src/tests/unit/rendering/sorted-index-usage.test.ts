/**
 * The ordering pair's buffer usage decides how often three re-uploads it.
 *
 * three's shared attribute cache (`renderers/common/Attributes.js`, used by
 * BOTH WebGPURenderer backends) re-uploads a `DynamicDrawUsage` attribute on
 * EVERY render, whether or not its version changed, and the backends clear
 * `updateRanges` after one upload — so after the first ranged write every
 * later frame paid a full `writeBuffer` of the ordering pair: 8 B x capacity
 * per node per frame (80 MB/frame at 10M, #2944 A1). On WebGPU the pair must
 * therefore be `StaticDrawUsage` (uploads on version bumps only); on the
 * classic WebGLRenderer usage is only a `bufferData` hint and stays as it was.
 */
import * as THREE from 'three';
import { afterEach, describe, expect, it } from 'vitest';

import {
  attachElementStorage,
  configureSortedIndexChunkedApply,
  writeSortedIndexIdentity,
} from '../../../rendering/element-storage';
import { POINT_TEXTURE_LAYOUT } from '../../../rendering/element-texture-layout';

interface AttributesLike {
  update(attribute: THREE.BufferAttribute, type: number): void;
}
type AttributesCtor = new (backend: unknown, info: unknown) => AttributesLike;

// three ships this module without a type declaration; import it by a
// non-literal specifier and type it locally.
const ATTRIBUTES_MODULE = 'three/src/renderers/common/Attributes.js';
const VERTEX = 1; // AttributeType.VERTEX in three/src/renderers/common/Constants.js

async function countUploads(attribute: THREE.BufferAttribute, frames: number): Promise<number> {
  const { default: Attributes } = (await import(/* @vite-ignore */ ATTRIBUTES_MODULE)) as {
    default: AttributesCtor;
  };
  let uploads = 0;
  const backend = {
    createAttribute: () => {},
    updateAttribute: () => {
      uploads++;
    },
  };
  const info = { createAttribute: () => {} };
  const attributes = new Attributes(backend, info);
  for (let f = 0; f < frames; f++) attributes.update(attribute, VERTEX);
  return uploads;
}

function usageOf(geometry: THREE.InstancedBufferGeometry, name: string): THREE.Usage {
  return (geometry.getAttribute(name) as THREE.BufferAttribute).usage;
}

function makeGeometry(): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();
  attachElementStorage(geometry, 1024, POINT_TEXTURE_LAYOUT);
  return geometry;
}

describe('ordering pair buffer usage', () => {
  afterEach(() => configureSortedIndexChunkedApply(true));

  it('keeps DynamicDraw on the classic WebGL backend', () => {
    configureSortedIndexChunkedApply(true);
    const geometry = makeGeometry();
    expect(usageOf(geometry, 'aSortedIndex')).toBe(THREE.DynamicDrawUsage);
    expect(usageOf(geometry, 'aSortedIndexB')).toBe(THREE.DynamicDrawUsage);
  });

  // Fails until the ordering pair uses StaticDraw on WebGPU (next commit).
  it.fails('uses StaticDraw on the WebGPU renderer backends', () => {
    configureSortedIndexChunkedApply(false);
    const geometry = makeGeometry();
    expect(usageOf(geometry, 'aSortedIndex')).toBe(THREE.StaticDrawUsage);
    expect(usageOf(geometry, 'aSortedIndexB')).toBe(THREE.StaticDrawUsage);
  });

  // Fails until the ordering pair uses StaticDraw on WebGPU (next commit).
  it.fails('on WebGPU, an unchanged ordering is not re-uploaded on later frames', async () => {
    configureSortedIndexChunkedApply(false);
    const geometry = makeGeometry();
    writeSortedIndexIdentity(geometry, 1024);
    const attr = geometry.getAttribute('aSortedIndex') as THREE.BufferAttribute;
    // First frame creates the buffer; nine more idle frames must upload nothing.
    expect(await countUploads(attr, 10)).toBe(0);
  });

  // Fails until the ordering pair uses StaticDraw on WebGPU (next commit).
  it.fails('on WebGPU, a new ordering write still reaches the GPU', async () => {
    configureSortedIndexChunkedApply(false);
    const geometry = makeGeometry();
    const attr = geometry.getAttribute('aSortedIndex') as THREE.BufferAttribute;
    const { default: Attributes } = (await import(/* @vite-ignore */ ATTRIBUTES_MODULE)) as {
      default: AttributesCtor;
    };
    let uploads = 0;
    const attributes = new Attributes(
      {
        createAttribute: () => {},
        updateAttribute: () => {
          uploads++;
        },
      },
      { createAttribute: () => {} }
    );
    attributes.update(attr, VERTEX); // create
    writeSortedIndexIdentity(geometry, 1024); // bumps the version
    attributes.update(attr, VERTEX);
    attributes.update(attr, VERTEX);
    expect(uploads).toBe(1);
  });
});
