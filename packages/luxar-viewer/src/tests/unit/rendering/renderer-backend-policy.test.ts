import * as THREE from 'three';
import { afterEach, describe, expect, it } from 'vitest';

import {
  configureRendererBackendPolicy,
  policyForBackend,
  rendererBackendPolicy,
} from '../../../rendering/renderer-backend-policy';

describe('renderer backend policy', () => {
  afterEach(() => configureRendererBackendPolicy('webgl'));

  it('streams orderings with back-pressure and skips eviction on classic WebGL', () => {
    expect(policyForBackend('webgl')).toEqual({
      kind: 'webgl',
      ordering: { chunked: true, adoptHostBuffer: false, usage: THREE.DynamicDrawUsage },
      evictRenderObjects: false,
      gpuCompute: false,
    });
  });

  it('writes orderings whole, evicts render objects and offers compute on native WebGPU', () => {
    expect(policyForBackend('webgpu-native')).toEqual({
      kind: 'webgpu-native',
      ordering: { chunked: false, adoptHostBuffer: true, usage: THREE.StaticDrawUsage },
      evictRenderObjects: true,
      gpuCompute: true,
    });
  });

  it('treats the WebGL2 bridge like WebGPU for uploads but offers no compute', () => {
    const bridge = policyForBackend('webgpu-gl');
    expect(bridge.ordering).toEqual(policyForBackend('webgpu-native').ordering);
    expect(bridge.evictRenderObjects).toBe(true);
    expect(bridge.gpuCompute).toBe(false);
  });

  it('defaults to classic WebGL and follows the last configuration', () => {
    expect(rendererBackendPolicy().kind).toBe('webgl');
    configureRendererBackendPolicy('webgpu-native');
    expect(rendererBackendPolicy()).toEqual(policyForBackend('webgpu-native'));
  });
});
