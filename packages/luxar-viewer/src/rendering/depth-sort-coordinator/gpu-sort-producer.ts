/**
 * The GPU depth sort as a {@link SortProducer} (#3162).
 *
 * On a native WebGPU renderer it orders instanced nodes (gsplats, points,
 * lines) with `rendering/gpu-depth-sort/`: the centers are read from the node's
 * element texture and the order is written straight into the inactive ordering
 * buffer, so a re-sort moves nothing across the CPU. Its outcome is a DEVICE
 * ordering ({@link DeviceOrderingWrite}): the element-storage pump runs the
 * sort against the inactive buffer and then flips, under the same rules as a
 * host ordering.
 *
 * It answers synchronously, so the scheduler settles the outcome in the
 * dispatching call and the per-frame pass flips it before the render.
 *
 * The sorter belongs to one `GPUDevice`. The producer follows whatever device
 * the context getter returns: on a new device (a recreated renderer after a
 * device loss) it disposes the old sorter and compiles a new one, and while it
 * compiles, or when there is no native device, it is not ready, so sorts go to
 * the SortWorker. Mesh ordering (an index-buffer permutation) and nodes the
 * device limits cannot hold stay on the worker too.
 *
 * @module rendering/depth-sort-coordinator/gpu-sort-producer
 */

import type * as THREE from 'three';

import { getElementStorage, type DeviceOrderingWrite } from '../element-storage';
import { setGpuByteReservation } from '../gpu-byte-budget';
import { GpuDepthSorter } from '../gpu-depth-sort/gpu-depth-sorter';
import type { GpuDevice } from '../gpu-depth-sort/gpu-types';
import type { GpuSortContext } from '../gpu-depth-sort/three-webgpu-bridge';
import { hasCommittedData } from '../../types/committed-data';
import { log, Modules } from '../../utils/log';
import type { SortJob, SortOutcome, SortProducer } from './sort-producers';

/** The GPU depth sort as a producer; see the module note. */
export class GpuSortProducer implements SortProducer {
  /** This producer's label in timings and profiler completions. */
  readonly route = 'gpu' as const;
  private sorter: GpuDepthSorter | null = null;
  /** The device the sorter, or its pending compile, belongs to. */
  private device: GpuDevice | null = null;
  /** Devices whose pipelines failed to build, or that were lost: never retried. */
  private readonly failed = new WeakSet<GpuDevice>();

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

  /** Instanced, texture-backed nodes the device limits can hold; never a mesh. */
  accepts(job: SortJob): boolean {
    if (job.state.triangleSource) return false;
    if (!getElementStorage(job.mesh.geometry)) return false;
    return this.sorter?.accepts(job.state.count) ?? false;
  }

  /** A device ordering whose writer sorts the node into the inactive buffer. */
  run(job: SortJob): SortOutcome {
    // Filled in when the pump runs the sort: the encode time is its kernel time.
    const timing = { route: 'gpu' as const, kernelMs: 0, boundaryMs: 0, queueMs: 0 };
    // The pump runs the sort later in this frame; snapshot the pose now (the
    // job's matrix is a shared scratch).
    const modelView = Float32Array.from(job.modelView.elements);
    const count = job.state.count;
    const write: DeviceOrderingWrite = {
      count,
      write: (target) => this.write(job, count, modelView, target, timing),
    };
    return { kind: 'device', generation: job.generation, write, timing };
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
    if (job.state.generation !== job.generation || !hasCommittedData(job.mesh)) return false;
    const context = this.getContext();
    const storage = getElementStorage(job.mesh.geometry);
    if (!context || !this.sorter || context.device !== this.device || !storage) return false;
    const centers = context.texture(storage.texture);
    const buffer = context.buffer(target);
    if (!centers || !buffer) return false;
    const result = this.sorter.sort({
      centers,
      textureWidth: storage.texture.image.width,
      texelsPerElement: storage.layout.texelsPerElement,
      centerMode: storage.layout.sortCenter,
      count,
      modelView,
      target: buffer,
    });
    if (!result) return false;
    timing.kernelMs = result.encodeMs;
    setGpuByteReservation(this, this.sorter.scratchBytes);
    return true;
  }

  private compileFor(device: GpuDevice): void {
    this.release();
    this.device = device;
    if (this.failed.has(device)) return;
    // A lost device runs nothing, and its sorts would leave orders stale with
    // no error: stop using it, so sorts fall back to the SortWorker until the
    // host hands over a new device.
    void device.lost?.then(() => {
      this.failed.add(device);
      if (this.device === device) this.release();
    });
    GpuDepthSorter.create(device).then(
      (sorter) => {
        if (this.device === device) this.sorter = sorter;
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
    this.sorter?.dispose();
    this.sorter = null;
    this.device = null;
    setGpuByteReservation(this, 0);
  }
}
