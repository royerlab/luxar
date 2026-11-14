/**
 * Who computes a node's ordering: the seam between the scheduler and the
 * machinery that sorts.
 *
 * The scheduler (`scheduler.ts`) owns everything a sort means to the
 * coordinator: the single-in-flight rule and its queued re-sort, the dispatch
 * pose, the generation and commit checks a result must pass, staging it into
 * the ordering buffers, the profiler pass, and releasing a held append draw.
 * A {@link SortProducer} owns only the computation. The SortWorker is one
 * producer ({@link workerSortProducer}): a CPU counting sort in a worker that
 * resolves later with a host array. A producer may also answer synchronously,
 * or with a GPU-side writer instead of an array (`gpuSortProducer`);
 * the scheduler settles every outcome through the same path.
 *
 * @module rendering/depth-sort-coordinator/sort-producers
 */

import type * as THREE from 'three';

import type { DeviceOrderingWrite } from '../element-storage';
import { withTimeout } from '../../workers/worker-pool/timeout/with-timeout';
import { workerHost, type CoordinatorState, type NodeSortState } from './state';
import { SORT_RPC_TIMEOUT_MS } from './worker-lifecycle';

/** One sort: one node, at the pose it was dispatched from. */
export interface SortJob {
  readonly nodeId: string;
  readonly mesh: THREE.Mesh;
  readonly state: NodeSortState;
  /** The node's generation at dispatch; a result for another is stale. */
  readonly generation: number;
  /** `inverse(camera.matrixWorld) × mesh.matrixWorld` at dispatch. */
  readonly modelView: THREE.Matrix4;
}

/** Where a sort's time went, for the profiler's 'Depth Sort' pass. */
export interface SortTiming {
  readonly route: SortRoute;
  /** Inside the worker sort kernel; unavailable for a deferred GPU write. */
  readonly kernelMs: number | null;
  /** Producer-side overhead around the kernel, when measured. */
  readonly boundaryMs: number | null;
  /** Transport and queueing between dispatch and result. */
  readonly queueMs: number | null;
}

/** Which producer computed an ordering. */
export type SortRoute = 'worker' | 'gpu';

/** What a producer hands the scheduler to stage. */
export type SortOutcome =
  | {
      readonly kind: 'host';
      readonly generation: number;
      readonly ordering: Uint32Array;
      readonly timing: SortTiming;
    }
  | {
      readonly kind: 'device';
      readonly generation: number;
      readonly write: DeviceOrderingWrite;
      readonly timing: SortTiming;
    };

/** Something that can compute orderings. */
export interface SortProducer {
  readonly route: SortRoute;
  /** Only capture and camera-motion sorts opt into this producer. */
  readonly captureOnly?: boolean;
  /** Whether it can sort at all right now (initialized, device present). */
  ready(c: CoordinatorState): boolean;
  /** Whether it can sort this particular node (its type, its size). */
  accepts(job: SortJob): boolean;
  /**
   * Sort `job`. A synchronous producer returns its outcome; an asynchronous one
   * a promise. `null` means there is nothing to apply (the producer no longer
   * holds this generation).
   */
  run(job: SortJob): SortOutcome | null | Promise<SortOutcome | null>;
}

/**
 * The SortWorker: available once its initialization has settled `ready`.
 *
 * A live `api` handle is not enough: it is wrapped BEFORE `initializeWithGuard`
 * is awaited, so between the spawn and 'ready' a `sort` would reject
 * NOT_INITIALIZED worker-side. Gating here keeps it one check for every caller
 * (the capture drain's force loop dispatches without consulting `registered`,
 * and a commit can reach the scheduler during startup warm-up).
 *
 * The RPC is raced against a timeout: a worker that CRASHES mid-session leaves
 * the Comlink call pending forever, and a stuck `inFlight` is unrecoverable
 * (the per-frame scheduler skips in-flight nodes, and later commits only set
 * `resortQueued`, which never drains). The scheduler's failure path clears
 * `inFlight` and drains the queue instead. A merely slow sort that resolves
 * after the deadline is dropped by the race, and the node re-sorts on its next
 * commit or camera trigger.
 */
export const workerSortProducer: SortProducer = {
  route: 'worker',
  ready: () => workerHost.api !== null && workerHost.workerInitState === 'ready',
  // Every geometry type: the worker sorts plain centers.
  accepts: () => true,
  async run(job: SortJob): Promise<SortOutcome | null> {
    const api = workerHost.api;
    if (!api) return null;
    const dispatchedAt = performance.now();
    const result = await withTimeout(
      'depth-sort',
      api.sort({
        nodeId: job.nodeId,
        generation: job.generation,
        modelView: new Float32Array(job.modelView.elements),
      }),
      SORT_RPC_TIMEOUT_MS
    );
    if (!result) return null;
    // kernelMs: inside the backend call (incl. wasm-bindgen boundary copies for
    // compiled WASM); boundaryMs: worker-side overhead around it; queueMs: the
    // round trip minus worker time (Comlink RPC + structured clone + event-loop
    // queueing). All durations, so mixing the worker and main clocks is safe;
    // clamped at 0 against timer granularity.
    const roundTripMs = performance.now() - dispatchedAt;
    return {
      kind: 'host',
      generation: result.generation,
      ordering: result.ordering,
      timing: {
        route: 'worker',
        kernelMs: result.kernelMs,
        boundaryMs: Math.max(0, result.workerMs - result.kernelMs),
        queueMs: Math.max(0, roundTripMs - result.workerMs),
      },
    };
  },
};

/** The first ready producer that accepts `job`, in the coordinator's preference order. */
export function selectSortProducer(
  c: CoordinatorState,
  job: SortJob,
  preferGpu = true
): SortProducer | null {
  return (
    c.sortProducers.find((p) => (!p.captureOnly || preferGpu) && p.ready(c) && p.accepts(job)) ??
    null
  );
}

/** Whether any producer could sort right now. */
export function anySortProducerReady(c: CoordinatorState): boolean {
  return c.sortProducers.some((p) => p.ready(c));
}
