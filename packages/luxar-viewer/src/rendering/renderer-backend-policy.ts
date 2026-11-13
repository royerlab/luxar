/**
 * The page-wide rendering behaviour that depends on which GPU backend runs.
 *
 * Several rendering modules need one answer per page about the backend: how
 * the depth-sort ordering pair is uploaded, whether a pool rebuild must evict
 * three's cached `RenderObject`, and whether WebGPU compute is available. They
 * used to each keep their own boolean, set from a different capability field
 * (`apiSurface === 'webgl2'`, `apiSurface === 'webgpu'`) or inferred from
 * another module's switch. This module derives all of them from ONE input,
 * {@link RendererCapabilities.backendKind}, in one pure function
 * ({@link policyForBackend}), so the derivations sit side by side and a reader
 * can see what differs between backends in one place.
 *
 * Renderer setup (`configureRendererBackend`) is the only writer. The policy is
 * page-wide and the last call wins: every Luxar host on a page renders through
 * one backend (renderer-setup warns on a mix).
 *
 * @module rendering/renderer-backend-policy
 */

import * as THREE from 'three';
import type { RendererCapabilities } from './renderer-capabilities';

/** Which GPU API actually runs (see {@link RendererCapabilities.backendKind}). */
export type RendererBackendKind = RendererCapabilities['backendKind'];

/** How a depth-sort ordering reaches the `aSortedIndex` / `aSortedIndexB` pair. */
export interface OrderingUploadPolicy {
  /**
   * Stream large orderings in slices with upload back-pressure
   * (`element-storage.ts`, issue #715). Classic WebGL only: the back-pressure
   * waits on the attribute's `onUploadCallback`, which neither WebGPU backend
   * fires, so the WebGPU backends write an ordering whole.
   */
  readonly chunked: boolean;
  /**
   * Adopt the worker's capacity-sized ordering buffer as the attribute's array
   * instead of copying it. Only for a whole-ordering write: a sliced stream
   * copies slice by slice.
   */
  readonly adoptHostBuffer: boolean;
  /**
   * Buffer usage for the ordering pair. `StaticDrawUsage` on both WebGPU
   * backends: three's shared attribute cache (`renderers/common/Attributes.js`)
   * re-uploads a `DynamicDrawUsage` attribute on every render even when its
   * version is unchanged, which cost a full `writeBuffer` of both buffers per
   * node per frame (80 MB/frame at 10M elements, #2944). Every buffer three
   * creates there carries COPY_DST, so a static attribute still uploads on each
   * version bump. Classic WebGL keeps `DynamicDrawUsage`, where usage is only
   * the `bufferData` hint.
   */
  readonly usage: THREE.Usage;
}

/** Everything that depends on the backend, derived from {@link RendererBackendKind}. */
export interface RendererBackendPolicy {
  readonly kind: RendererBackendKind;
  readonly ordering: OrderingUploadPolicy;
  /**
   * Flush three's cached `RenderObject` after a pool rebuild
   * (`invalidate-render-object.ts`). Both WebGPU backends cache the vertex
   * buffers there; on classic WebGL the same `dispose` dispatch would instead
   * destroy the compiled program and force a recompile on every commit.
   */
  readonly evictRenderObjects: boolean;
  /**
   * A real WebGPU device runs, so compute pipelines are available. False on
   * three's internal WebGL2 backend (`webgpu-gl`), which has no compute.
   */
  readonly gpuCompute: boolean;
}

/** The policy for a backend. Pure: the single place the per-backend answers are decided. */
export function policyForBackend(kind: RendererBackendKind): RendererBackendPolicy {
  const classicWebGL = kind === 'webgl';
  return {
    kind,
    ordering: {
      chunked: classicWebGL,
      adoptHostBuffer: !classicWebGL,
      usage: classicWebGL ? THREE.DynamicDrawUsage : THREE.StaticDrawUsage,
    },
    evictRenderObjects: !classicWebGL,
    gpuCompute: kind === 'webgpu-native',
  };
}

/**
 * Classic WebGL until renderer setup says otherwise: it is the production
 * default, and headless or unit contexts have no renderer whose behaviour the
 * WebGPU answers would describe.
 */
let current: RendererBackendPolicy = policyForBackend('webgl');

/** Set the page-wide policy. Renderer setup calls this once per renderer. */
export function configureRendererBackendPolicy(kind: RendererBackendKind): void {
  current = policyForBackend(kind);
}

/** The page-wide policy for the backend renderer setup last configured. */
export function rendererBackendPolicy(): RendererBackendPolicy {
  return current;
}
