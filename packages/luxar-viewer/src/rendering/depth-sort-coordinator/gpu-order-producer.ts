/** Three.js resource handoff for the texture-based WebGPU depth sorter. */
import type * as THREE from 'three';
import {
  activeSortedIndexSlot,
  canComputeSortedIndex,
  getElementTexture,
  getElementTextureLayout,
  isElementTextureCurrent,
} from '../element-storage';
import { rendererBackendPolicy } from '../renderer-backend-policy';
import { setGpuOrderScratchBytes } from '../gpu-buffer-pool/geometry-bytes';
import { GpuDepthSorter } from '../gpu-depth-sort/gpu-depth-sorter';
import type { GpuBuffer, GpuDevice, GpuTexture } from '../gpu-depth-sort/gpu-types';
import type { Renderer } from '../renderer-capabilities';
import { log, Modules } from '../../utils/log';

interface NativeBackend {
  device: GpuDevice;
  get(resource: THREE.Texture | THREE.BufferAttribute): {
    texture?: GpuTexture;
    buffer?: GpuBuffer;
  };
  createStorageAttribute(attribute: THREE.BufferAttribute): void;
}

let backend: NativeBackend | null = null;
let sorter: GpuDepthSorter | null = null;
let epoch = 0;
let scratchOwner: THREE.InstancedBufferGeometry | null = null;

function releaseOwnedScratch(): void {
  if (!scratchOwner) return;
  const owner = scratchOwner;
  scratchOwner = null;
  owner.removeEventListener('dispose', releaseOwnedScratch);
  sorter?.releaseScratch();
  setGpuOrderScratchBytes(owner, 0);
}

/** Compile once for the current native device; a backend switch retires it. */
export function configureGpuOrderProducer(renderer: Renderer): void {
  const candidate = rendererBackendPolicy().gpuCompute
    ? ((renderer as unknown as { backend?: NativeBackend }).backend ?? null)
    : null;
  if (candidate === backend) return;
  releaseOwnedScratch();
  backend = candidate;
  sorter?.dispose();
  sorter = null;
  const current = ++epoch;
  if (!candidate?.device?.createComputePipelineAsync) return;
  void GpuDepthSorter.create(candidate.device).then(
    (compiled) => {
      if (current === epoch) sorter = compiled;
      else compiled.dispose();
    },
    (error: unknown) => {
      if (current === epoch) {
        log.warning(
          Modules.WORKER_POOL,
          'WebGPU depth sort unavailable; using worker order',
          error
        );
      }
    }
  );
}

/** Whether the texture sort can produce this node's next ordering. */
export function canGpuSort(geometry: THREE.InstancedBufferGeometry, count: number): boolean {
  if (!rendererBackendPolicy().gpuCompute || !backend || !sorter?.accepts(count)) return false;
  if (!canComputeSortedIndex(geometry)) return false;
  const texture = getElementTexture(geometry);
  const layout = getElementTextureLayout(geometry);
  const target = geometry.getAttribute(
    activeSortedIndexSlot(geometry) === 0 ? 'aSortedIndexB' : 'aSortedIndex'
  );
  return !!(
    texture &&
    layout &&
    isElementTextureCurrent(texture) &&
    backend.get(texture).texture &&
    target &&
    count <= target.array.length
  );
}

/** Submit into the inactive ordering buffer. The caller selects it only on success. */
export function submitGpuSort(
  geometry: THREE.InstancedBufferGeometry,
  count: number,
  modelView: THREE.Matrix4
): boolean {
  if (!canGpuSort(geometry, count) || !backend || !sorter) return false;
  const texture = getElementTexture(geometry)!;
  const layout = getElementTextureLayout(geometry)!;
  const inactive = geometry.getAttribute(
    activeSortedIndexSlot(geometry) === 0 ? 'aSortedIndexB' : 'aSortedIndex'
  ) as THREE.BufferAttribute;
  let target = backend.get(inactive).buffer;
  if (!target) {
    backend.createStorageAttribute(inactive);
    target = backend.get(inactive).buffer;
  }
  if (!target) return false;
  const sorted = sorter.sort({
    centers: backend.get(texture).texture!,
    textureWidth: texture.image.width,
    texelsPerElement: layout.texelsPerElement,
    centerMode: layout.label === 'segment' ? 'midpoint01' : 'texel0',
    count,
    modelView: modelView.elements,
    target,
  });
  if (!sorted) return false;
  if (!scratchOwner) {
    scratchOwner = geometry;
    geometry.addEventListener('dispose', releaseOwnedScratch);
  }
  setGpuOrderScratchBytes(scratchOwner, sorter.scratchBytes);
  return true;
}

/** Stop using this device after a failed sort; worker ordering remains available. */
export function disableGpuOrderProducer(): void {
  ++epoch;
  releaseOwnedScratch();
  sorter?.dispose();
  sorter = null;
}
