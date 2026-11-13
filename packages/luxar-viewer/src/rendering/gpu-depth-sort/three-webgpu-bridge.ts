/**
 * The GPU depth sort's view of a three.js `WebGPURenderer`: its device, and the
 * GPU resources behind a texture and an attribute.
 *
 * This is the one file that reaches into three's internals for the sort
 * (`renderer.backend`, `renderer._attributes`), the same containment as
 * `rendering/element-texture-row-upload.ts`, so a three upgrade that moves them
 * breaks here and nowhere else. Every lookup brings the resource up to date
 * first, through three's own update paths:
 *
 * - a texture through `renderer.initTexture`, which uploads it when its version
 *   moved (through the element-texture row upload, when installed). The upload
 *   is a queue write, so it lands before the sort's later submit, and the sort
 *   reads the texels the commit wrote;
 * - an attribute through three's attribute cache (`_attributes.update`), which
 *   creates its buffer or uploads a pending host write and records the version.
 *   The sort's result is then the newest content three knows of, and three does
 *   not later overwrite it with the host array.
 *
 * @module rendering/gpu-depth-sort/three-webgpu-bridge
 */

import type * as THREE from 'three';

import { rendererBackendPolicy } from '../renderer-backend-policy';
import { BufferUsage, type GpuBuffer, type GpuDevice, type GpuTexture } from './gpu-types';

/** What the sort needs from a renderer. */
export interface GpuSortContext {
  readonly device: GpuDevice;
  /** The current GPU texture behind `texture`, or null when it has none. */
  texture(texture: THREE.Texture): GpuTexture | null;
  /** The current GPU buffer behind a vertex `attribute`, or null when it has none. */
  buffer(attribute: THREE.BufferAttribute): GpuBuffer | null;
  /** Read `count` Uint32s back from `buffer` (diagnostics and tests only: it stalls). */
  readUint32(buffer: GpuBuffer, count: number): Promise<Uint32Array>;
}

/** The readback-buffer surface `readUint32` needs beyond {@link GpuBuffer}. */
interface MappableBuffer extends GpuBuffer {
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
}

/** `GPUMapMode.READ`, fixed by the WebGPU specification. */
const MAP_READ = 0x0001;

/** The internals of `WebGPURenderer` the bridge uses. */
interface WebGPURendererInternals {
  initTexture?(texture: THREE.Texture): void;
  _attributes?: { update(attribute: THREE.BufferAttribute, type: number): void } | null;
  backend?: {
    isWebGPUBackend?: boolean;
    device?: GpuDevice | null;
    get(object: object): { texture?: GpuTexture; buffer?: GpuBuffer };
  };
}

/** three's `AttributeType.VERTEX` (`renderers/common/Constants.js`). */
const VERTEX_ATTRIBUTE = 1;

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
 * renderer on every call, so it stays valid exactly as long as the renderer.
 */
export function createGpuSortContext(renderer: unknown): GpuSortContext | null {
  const r = renderer as WebGPURendererInternals | null;
  const backend = r?.backend;
  const device = backend?.device;
  if (!r || !backend?.isWebGPUBackend || !device || typeof r.initTexture !== 'function') {
    return null;
  }
  return {
    device,
    texture(texture) {
      r.initTexture!(texture);
      return backend.get(texture).texture ?? null;
    },
    buffer(attribute) {
      if (!r._attributes) return null;
      r._attributes.update(attribute, VERTEX_ATTRIBUTE);
      return backend.get(attribute).buffer ?? null;
    },
    async readUint32(buffer, count) {
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
    },
  };
}
