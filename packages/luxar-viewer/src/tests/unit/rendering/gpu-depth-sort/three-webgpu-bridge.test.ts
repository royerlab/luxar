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

function fakeRenderer(options: { native?: boolean } = {}) {
  const resources = new Map<object, { texture?: object; buffer?: object }>();
  const renderer = {
    initTexture: vi.fn((t: THREE.Texture) =>
      resources.set(t, { texture: { createView: vi.fn() } })
    ),
    _attributes: {
      update: vi.fn((a: THREE.BufferAttribute) =>
        resources.set(a, { buffer: { size: 4, destroy: vi.fn() } })
      ),
    },
    backend: {
      isWebGPUBackend: options.native ?? true,
      device: fakeDevice().device,
      get: (o: object) => resources.get(o) ?? {},
    },
  };
  return renderer;
}

afterEach(() => configureRendererBackendPolicy('webgl'));

describe('GPU depth sort bridge', () => {
  it('makes a context only for a WebGPURenderer on a native device', () => {
    expect(createGpuSortContext(fakeRenderer({ native: false }))).toBeNull();
    expect(createGpuSortContext({})).toBeNull();
    expect(createGpuSortContext(null)).toBeNull();
    expect(createGpuSortContext(fakeRenderer())).not.toBeNull();
  });

  it('uploads a texture through initTexture before returning its GPU texture', () => {
    const renderer = fakeRenderer();
    const ctx = createGpuSortContext(renderer)!;
    const texture = new THREE.DataTexture();
    expect(ctx.texture(texture)).not.toBeNull();
    expect(renderer.initTexture).toHaveBeenCalledWith(texture);
  });

  it('syncs an attribute through three’s vertex-attribute cache before returning its buffer', () => {
    const renderer = fakeRenderer();
    const ctx = createGpuSortContext(renderer)!;
    const attribute = new THREE.InstancedBufferAttribute(new Uint32Array(1), 1);
    expect(ctx.buffer(attribute)).not.toBeNull();
    expect(renderer._attributes.update).toHaveBeenCalledWith(attribute, 1);
  });

  it('keeps one context per renderer and none until the renderer has a device', () => {
    const renderer = fakeRenderer();
    expect(gpuSortContextFor(renderer)).toBe(gpuSortContextFor(renderer));
    const uninitialized = { ...fakeRenderer(), backend: undefined };
    expect(gpuSortContextFor(uninitialized)).toBeNull();
    (uninitialized as { backend?: unknown }).backend = fakeRenderer().backend;
    expect(gpuSortContextFor(uninitialized)).not.toBeNull();
  });

  it('offers a getter only when enabled, answering only on native WebGPU', () => {
    const renderer = fakeRenderer();
    expect(gpuSortContextGetter(false, () => renderer)).toBeUndefined();
    const getter = gpuSortContextGetter(true, () => renderer)!;
    configureRendererBackendPolicy('webgpu-gl');
    expect(getter()).toBeNull();
    configureRendererBackendPolicy('webgpu-native');
    expect(getter()).not.toBeNull();
  });
});
