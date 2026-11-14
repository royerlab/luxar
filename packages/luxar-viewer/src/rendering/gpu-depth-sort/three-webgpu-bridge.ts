/**
 * The GPU depth sort's view of a three.js `WebGPURenderer`: its device, and the
 * GPU resources behind an element texture and an ordering attribute.
 *
 * This is the one file that reaches into three's internals for the sort
 * (`renderer.backend`), the same containment as
 * `rendering/element-texture-row-upload.ts`, so a three upgrade that moves them
 * breaks here and nowhere else. The bridge never uploads anything itself:
 *
 * - a texture is handed over only once three has uploaded its CURRENT version
 *   (`isElementTextureCurrent`); before that the node keeps the SortWorker, so
 *   the sort can never read texels older than the commit it orders;
 * - an ordering attribute's buffer is the one three created, or created here
 *   through the backend's own `createStorageAttribute` when three has not drawn
 *   it yet (ordering pairs on native WebGPU are storage attributes,
 *   `element-storage.ts`). Three later finds it and records the version, so it
 *   does not re-upload the host array over the sort. A host write still
 *   pending on the attribute makes the element-storage pump wait before the
 *   device write, for the same reason.
 *
 * @module rendering/gpu-depth-sort/three-webgpu-bridge
 */

import type * as THREE from 'three';

import { isElementTextureCurrent } from '../element-storage';
import { rendererBackendPolicy } from '../renderer-backend-policy';
import { BufferUsage, type GpuBuffer, type GpuDevice, type GpuTexture } from './gpu-types';

/** What the sort needs from a renderer. */
export interface GpuSortContext {
  readonly device: GpuDevice;
  /** The GPU texture behind `texture` once its current version is uploaded, else null. */
  texture(texture: THREE.DataTexture): GpuTexture | null;
  /** The GPU buffer behind an ordering `attribute`, created if three has none yet. */
  buffer(attribute: THREE.BufferAttribute): GpuBuffer | null;
  /** Read `count` Uint32s back from `buffer` (diagnostics and tests only: it stalls). */
  readUint32(buffer: GpuBuffer, count: number): Promise<Uint32Array>;
}

/** The internals of `WebGPURenderer` the bridge uses. */
interface WebGPUBackendInternals {
  isWebGPUBackend?: boolean;
  device?: GpuDevice | null;
  get(object: object): { texture?: GpuTexture; buffer?: GpuBuffer };
  createStorageAttribute(attribute: THREE.BufferAttribute): void;
}

/** The readback-buffer surface `readUint32` needs beyond {@link GpuBuffer}. */
interface MappableBuffer extends GpuBuffer {
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
}

/** `GPUMapMode.READ`, fixed by the WebGPU specification. */
const MAP_READ = 0x0001;

/**
 * A host's depth-sort context getter: undefined when the GPU sort is disabled,
 * otherwise a getter that reads `getRenderer()` live (so a renderer recreated
 * after a device loss is picked up) and answers null off native WebGPU.
 */
export function gpuSortContextGetter(
  enabled: boolean,
  getRenderer: () => unknown
): (() => GpuSortContext | null) | undefined {
  if (!enabled) return undefined;
  return () => (rendererBackendPolicy().gpuCompute ? gpuSortContextFor(getRenderer()) : null);
}

/** Contexts by renderer, so a getter called every sort allocates nothing. */
const contexts = new WeakMap<object, GpuSortContext>();

/**
 * The context for `renderer` (see {@link createGpuSortContext}), made once per
 * renderer. A renderer recreated after a device loss is a new object, so it
 * gets a new context, on the new device.
 */
export function gpuSortContextFor(renderer: unknown): GpuSortContext | null {
  if (typeof renderer !== 'object' || renderer === null) return null;
  let context = contexts.get(renderer) ?? null;
  // Not cached while null: a renderer whose backend is not initialized yet
  // gets its context once it is.
  if (!context) {
    context = createGpuSortContext(renderer);
    if (context) contexts.set(renderer, context);
  }
  return context;
}

/**
 * The context for `renderer`, or null unless it is a `WebGPURenderer` on a
 * native WebGPU device. The context holds no state of its own: it reads the
 * backend on every call, so it stays valid exactly as long as the renderer.
 */
export function createGpuSortContext(renderer: unknown): GpuSortContext | null {
  const backend = (renderer as { backend?: WebGPUBackendInternals } | null)?.backend;
  const device = backend?.device;
  if (!backend?.isWebGPUBackend || !device || typeof backend.get !== 'function') return null;
  return {
    device,
    texture(texture) {
      return isElementTextureCurrent(texture) ? (backend.get(texture).texture ?? null) : null;
    },
    buffer(attribute) {
      const existing = backend.get(attribute).buffer;
      if (existing) return existing;
      backend.createStorageAttribute(attribute);
      return backend.get(attribute).buffer ?? null;
    },
    readUint32: (buffer, count) => readUint32(device, buffer, count),
  };
}

async function readUint32(
  device: GpuDevice,
  buffer: GpuBuffer,
  count: number
): Promise<Uint32Array> {
  const bytes = count * 4;
  const staging = device.createBuffer({
    label: 'luxar depth sort: readback',
    size: bytes,
    usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST,
  }) as MappableBuffer;
  const encoder = device.createCommandEncoder({ label: 'luxar depth sort: readback' });
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(MAP_READ);
  const out = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}
