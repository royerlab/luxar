/**
 * The GPU depth sort's bridge to three's WebGPURenderer: which renderers get a
 * context, and that each lookup brings the resource up to date through three's
 * own update paths before handing over the GPU object.
 */
import * as THREE from 'three';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { configureRendererBackendPolicy } from '../../../../rendering/renderer-backend-policy';
import {
  createGpuSortContext,
  gpuSortContextFor,
  gpuSortContextGetter,
} from '../../../../rendering/gpu-depth-sort/three-webgpu-bridge';
import { fakeDevice } from './fake-device';
import { attachElementStorage } from '../../../../rendering/element-storage';
import { POINT_TEXTURE_LAYOUT } from '../../../../rendering/element-texture-layout';

function fakeRenderer(options: { native?: boolean } = {}) {
  const resources = new Map<object, { texture?: object; buffer?: object }>();
  const backend = {
    isWebGPUBackend: options.native ?? true,
    device: fakeDevice().device,
    get: (o: object) => resources.get(o) ?? {},
    createStorageAttribute: vi.fn((a: THREE.BufferAttribute) =>
      resources.set(a, { buffer: { size: 4, usage: 0, destroy: vi.fn() } })
    ),
  };
  return { renderer: { backend }, resources, backend };
}

function elementTexture(): THREE.DataTexture {
  const geometry = new THREE.InstancedBufferGeometry();
  return attachElementStorage(geometry, 4, POINT_TEXTURE_LAYOUT);
}

afterEach(() => configureRendererBackendPolicy('webgl'));

describe('GPU depth sort bridge', () => {
  it('makes a context only for a WebGPURenderer on a native device', () => {
    expect(createGpuSortContext(fakeRenderer({ native: false }).renderer)).toBeNull();
    expect(createGpuSortContext({})).toBeNull();
    expect(createGpuSortContext(null)).toBeNull();
    expect(createGpuSortContext(fakeRenderer().renderer)).not.toBeNull();
  });

  it('hands over an element texture only once three has uploaded its current version', () => {
    const { renderer, resources } = fakeRenderer();
    const ctx = createGpuSortContext(renderer)!;
    const texture = elementTexture();
    resources.set(texture, { texture: { createView: vi.fn() } });
    expect(ctx.texture(texture)).toBeNull(); // never uploaded
    texture.onUpdate?.(texture);
    expect(ctx.texture(texture)).not.toBeNull();
    texture.needsUpdate = true; // a newer version is pending
    expect(ctx.texture(texture)).toBeNull();
  });

  it('returns three’s buffer, or creates the storage buffer three has not yet', () => {
    const { renderer, resources, backend } = fakeRenderer();
    const ctx = createGpuSortContext(renderer)!;
    const created = new THREE.InstancedBufferAttribute(new Uint32Array(1), 1);
    expect(ctx.buffer(created)).not.toBeNull();
    expect(backend.createStorageAttribute).toHaveBeenCalledWith(created);
    const existing = new THREE.InstancedBufferAttribute(new Uint32Array(1), 1);
    const buffer = { size: 4, usage: 0, destroy: vi.fn() };
    resources.set(existing, { buffer });
    expect(ctx.buffer(existing)).toBe(buffer);
    expect(backend.createStorageAttribute).toHaveBeenCalledTimes(1);
  });

  it('keeps one context per renderer and none until the renderer has a device', () => {
    const { renderer } = fakeRenderer();
    expect(gpuSortContextFor(renderer)).toBe(gpuSortContextFor(renderer));
    const uninitialized: { backend?: unknown } = {};
    expect(gpuSortContextFor(uninitialized)).toBeNull();
    uninitialized.backend = fakeRenderer().backend;
    expect(gpuSortContextFor(uninitialized)).not.toBeNull();
  });

  it('offers a getter only when enabled, answering only on native WebGPU', () => {
    const { renderer } = fakeRenderer();
    expect(gpuSortContextGetter(false, () => renderer)).toBeUndefined();
    const getter = gpuSortContextGetter(true, () => renderer)!;
    configureRendererBackendPolicy('webgpu-gl');
    expect(getter()).toBeNull();
    configureRendererBackendPolicy('webgpu-native');
    expect(getter()).not.toBeNull();
  });
});
