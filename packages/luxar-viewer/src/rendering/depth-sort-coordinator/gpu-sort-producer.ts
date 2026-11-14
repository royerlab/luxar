/**
 * The GPU depth sort as a {@link SortProducer} (#3162).
 *
 * On a native WebGPU renderer it re-sorts instanced nodes (gsplats, points,
 * lines) with `rendering/gpu-depth-sort/`: the centers are read from the node's
 * element texture and the order lands in the inactive ordering buffer, so a
 * re-sort moves nothing across the CPU. Its outcome is a DEVICE ordering
 * ({@link DeviceOrderingWrite}): the element-storage pump runs the sort against
 * the inactive buffer (after any host upload pending on it) and then flips,
 * under the same rules as a host ordering.
 *
 * What it takes:
 *
 * - camera-motion and capture re-sorts only. A commit's first sort stays on the
 *   SortWorker, whose host array is what the grow seed and the repairs read
 *   (a GPU write leaves that array behind, `element-storage.ts`);
 * - instanced nodes whose ordering pair is storage-capable and whose element
 *   texture three has uploaded at its current version. Mesh (an index-buffer
 *   permutation), a node the device limits cannot hold, and a node it failed to
 *   write until its next commit keep the worker.
 *
 * The sorter belongs to one `GPUDevice`. The producer follows whatever device
 * the context getter returns: on a new device (a recreated renderer after a
 * device loss) it disposes the old sorter and compiles a new one. While it
 * compiles, after the device is lost, after its pipelines fail to build, or
 * after a sort throws on it, the producer is not ready and sorts go to the
 * worker. One producer per coordinator, so each host sorts on its own renderer.
 *
 * @module rendering/depth-sort-coordinator/gpu-sort-producer
 */

import type * as THREE from 'three';

import {
  canComputeSortedIndex,
  getElementTexture,
  getElementTextureLayout,
  type DeviceOrderingWrite,
} from '../element-storage';
import { setGpuByteReservation } from '../gpu-byte-budget';
import { GpuDepthSorter, type DepthSortRequest } from '../gpu-depth-sort/gpu-depth-sorter';
import type { GpuDevice } from '../gpu-depth-sort/gpu-types';
import type { GpuSortContext } from '../gpu-depth-sort/three-webgpu-bridge';
import { hasCommittedData } from '../../types/committed-data';
import { log, Modules } from '../../utils/log';
import type { SortJob, SortOutcome, SortProducer } from './sort-producers';
import type { NodeSortState } from './state';

/** The GPU depth sort as a producer; see the module note. */
export class GpuSortProducer implements SortProducer {
  /** This producer's label in timings and profiler completions. */
  readonly route = 'gpu' as const;
  private sorter: GpuDepthSorter | null = null;
  /** The device the sorter, or its pending compile, belongs to. */
  private device: GpuDevice | null = null;
  /**
   * Bumped by every compile and release, so a compile that resolves after a
   * newer one started (or after a release) disposes its sorter instead of
   * installing it over the newer one.
   */
  private compileEpoch = 0;
  /** Devices that were lost, failed to build the pipelines, or threw: never retried. */
  private readonly failed = new WeakSet<GpuDevice>();
  /** Devices whose loss is already watched (one listener each). */
  private readonly watched = new WeakSet<GpuDevice>();
  /**
   * Nodes, by the generation at which this producer could not write their
   * order: it declines them until their next commit, so the re-sort that
   * failure arms goes to the SortWorker rather than failing here again.
   */
  private readonly declined = new WeakMap<NodeSortState, number>();

  /** @param getContext - the host's live WebGPU context, or null off native WebGPU */
  constructor(private readonly getContext: () => GpuSortContext | null) {}

  /** Ready once a native device's pipelines compiled; compiles on a new device. */
  ready(): boolean {
    const context = this.getContext();
    if (!context) {
      this.release();
      return false;
    }
    if (context.device !== this.device) this.compileFor(context.device);
    return this.sorter !== null;
  }

  /** A re-sort of an instanced node whose resources are on the GPU and fit the device. */
  accepts(job: SortJob): boolean {
    if (job.reason === 'commit' || job.state.triangleSource) return false;
    if (this.declined.get(job.state) === job.state.generation) return false;
    const geometry = job.mesh.geometry as THREE.InstancedBufferGeometry;
    const texture = getElementTexture(geometry);
    if (!texture || !getElementTextureLayout(geometry) || !canComputeSortedIndex(geometry)) {
      return false;
    }
    return !!this.getContext()?.texture(texture) && !!this.sorter?.accepts(job.state.count);
  }

  /** A device ordering whose writer sorts the node into the inactive buffer. */
  run(job: SortJob): SortOutcome {
    // Filled in when the pump runs the sort: the encode time is its kernel time.
    const timing = { route: 'gpu' as const, kernelMs: 0, boundaryMs: 0, queueMs: 0 };
    // The pump runs the sort later; snapshot the pose now (the job's matrix is
    // a shared scratch).
    const modelView = Float32Array.from(job.modelView.elements);
    const count = job.state.count;
    const write: DeviceOrderingWrite = {
      count,
      write: (target) => this.write(job, count, modelView, target, timing),
    };
    return { kind: 'device', generation: job.generation, write, timing };
  }

  /**
   * Free the sorter's scratch (16 B per element of the largest node it sorted)
   * while keeping its pipelines; the next sort allocates it again at that
   * node's size. The host calls this when its scene is torn down, so a large
   * scene's scratch does not outlive it.
   */
  releaseScratch(): void {
    this.sorter?.releaseScratch();
    setGpuByteReservation(this, 0);
  }

  /** Free the sorter's GPU resources. */
  dispose(): void {
    this.release();
  }

  /** Sort `job`'s node into `target`; false when it can no longer be written. */
  private write(
    job: SortJob,
    count: number,
    modelView: Float32Array,
    target: THREE.InstancedBufferAttribute,
    timing: { kernelMs: number }
  ): boolean {
    // A superseded population is the scheduler's business (its newer commit
    // sorts again); anything else is this producer failing the node.
    if (job.state.generation !== job.generation || !hasCommittedData(job.mesh)) return false;
    const encodeMs = this.encode(job, count, modelView, target);
    if (encodeMs === null) {
      this.declined.set(job.state, job.generation);
      return false;
    }
    timing.kernelMs = encodeMs;
    setGpuByteReservation(this, this.sorter?.scratchBytes ?? 0);
    return true;
  }

  /** Encode and submit the sort; its encode time, or null when it cannot run. */
  private encode(
    job: SortJob,
    count: number,
    modelView: Float32Array,
    target: THREE.InstancedBufferAttribute
  ): number | null {
    try {
      // Resolving the resources is inside the try too: three creates the
      // target's buffer on demand here, and a throw from that says the same
      // about the device as one from the sort.
      const request = this.request(job, count, modelView, target);
      if (!request || !this.sorter) return null;
      return this.sorter.sort(request)?.encodeMs ?? null;
    } catch (error) {
      // A throw (a resource three cannot create, a target the sorter refuses,
      // a device in a bad state) says more about the device than the node:
      // stop using it.
      log.warning(Modules.WORKER_POOL, 'GPU depth sort failed; sorting on the CPU worker', error);
      if (this.device) this.failed.add(this.device);
      this.release();
      return null;
    }
  }

  /** The sort request for the node, or null when a GPU resource is missing. */
  private request(
    job: SortJob,
    count: number,
    modelView: Float32Array,
    target: THREE.InstancedBufferAttribute
  ): DepthSortRequest | null {
    const context = this.getContext();
    const geometry = job.mesh.geometry as THREE.InstancedBufferGeometry;
    const texture = getElementTexture(geometry);
    const layout = getElementTextureLayout(geometry);
    if (!context || context.device !== this.device || !texture || !layout) return null;
    const centers = context.texture(texture);
    const buffer = context.buffer(target);
    if (!centers || !buffer) return null;
    return {
      centers,
      textureWidth: texture.image.width,
      texelsPerElement: layout.texelsPerElement,
      centerMode: layout.sortCenter,
      count,
      modelView,
      target: buffer,
    };
  }

  private compileFor(device: GpuDevice): void {
    this.release();
    this.device = device;
    const epoch = this.compileEpoch;
    if (this.failed.has(device)) return;
    if (!this.watched.has(device)) {
      this.watched.add(device);
      // A lost device runs nothing, and its sorts would leave orders stale
      // with no error: stop using it until the host hands over a new one.
      void device.lost?.then(() => {
        this.failed.add(device);
        if (this.device === device) this.release();
      });
    }
    GpuDepthSorter.create(device).then(
      (sorter) => {
        if (this.compileEpoch === epoch) this.sorter = sorter;
        else sorter.dispose();
      },
      (error: unknown) => {
        this.failed.add(device);
        log.warning(
          Modules.WORKER_POOL,
          'GPU depth sort unavailable on this device; sorting on the CPU worker',
          error
        );
      }
    );
  }

  private release(): void {
    this.compileEpoch++;
    this.sorter?.dispose();
    this.sorter = null;
    this.device = null;
    setGpuByteReservation(this, 0);
  }
}
