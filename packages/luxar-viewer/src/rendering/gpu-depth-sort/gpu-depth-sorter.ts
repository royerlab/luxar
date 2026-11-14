/**
 * Back-to-front depth ordering on the GPU (WebGPU compute, #3162).
 *
 * The native WebGPU counterpart of the SortWorker's counting sort: the same
 * 16-bit view-depth keys (see `wgsl.ts` for the arithmetic and its tolerance),
 * sorted by a stable four-pass (4-bit digit) LSD radix sort, written into an ordering
 * attribute's GPU buffer. Nothing crosses the CPU: the centers are read from
 * the element texture the geometry already uploaded, and the order lands in
 * the buffer the vertex shader reads, so a re-sort costs no 4-byte-per-element
 * upload and no worker round trip.
 *
 * One sorter per `GPUDevice`. {@link GpuDepthSorter.create} compiles the
 * pipelines asynchronously; scratch grows to the largest element count seen.
 * Each {@link GpuDepthSorter.sort} records one command buffer and submits it,
 * so sorts of several nodes run one after another on the queue and can share
 * the scratch.
 *
 * @module rendering/gpu-depth-sort/gpu-depth-sorter
 */

import { BufferUsage, type GpuBuffer, type GpuDevice, type GpuTexture } from './gpu-types';
import {
  SCAN_BLOCK_ELEMENTS,
  packParams,
  planDispatch,
  planGrowth,
  planScratch,
  viewZRow,
  type CenterMode,
  type DispatchPlan,
} from './plan';
import { createDepthSortPipelines, type DepthSortPipelines } from './pipelines';
import { PARAMS_SLOT, SCAN_SLOT, SortScratch } from './scratch';
import type { DepthSortPass } from './wgsl';

/** One node's sort: where its centers are, and where the order goes. */
export interface DepthSortRequest {
  /** The node's RGBA32F element texture. */
  readonly centers: GpuTexture;
  /** Its width in texels (a multiple of `texelsPerElement`). */
  readonly textureWidth: number;
  readonly texelsPerElement: number;
  /** Which texels hold the sort center (a line segment sorts on its midpoint). */
  readonly centerMode: CenterMode;
  /** Elements to order: storage slots `[0, count)`. */
  readonly count: number;
  /** Column-major model-view matrix (`camera.matrixWorldInverse × mesh.matrixWorld`). */
  readonly modelView: ArrayLike<number>;
  /** Receives `count` Uint32 indices, back to front. Needs COPY_DST usage. */
  readonly target: GpuBuffer;
}

/** Initial bounds: min at the top of the ordered-u32 range, max at the bottom. */
const EMPTY_BOUNDS = new Uint32Array([0xffffffff, 0]);

export class GpuDepthSorter {
  private scratch: SortScratch | null = null;
  private disposed = false;

  private constructor(
    private readonly device: GpuDevice,
    private readonly pipelines: DepthSortPipelines
  ) {}

  /** Compile the pipelines for `device`. Rejects if the device cannot build them. */
  static async create(device: GpuDevice): Promise<GpuDepthSorter> {
    return new GpuDepthSorter(device, await createDepthSortPipelines(device));
  }

  /** Whether this device can sort `count` elements (its limits fit the scratch). */
  accepts(count: number): boolean {
    return !this.disposed && !('unsupported' in planScratch(count, this.device.limits));
  }

  /** Bytes of scratch currently held on the GPU. */
  get scratchBytes(): number {
    return this.scratch?.bytes ?? 0;
  }

  /**
   * Encode and submit one sort. Returns the CPU time spent encoding it, or null
   * when the request cannot be sorted here (see {@link accepts}).
   */
  sort(request: DepthSortRequest): { encodeMs: number } | null {
    const started = performance.now();
    if (request.target.size < request.count * 4) {
      throw new RangeError('depth sort target size is smaller than the ordering');
    }
    if ((request.target.usage & BufferUsage.COPY_DST) === 0) {
      throw new Error('depth sort target needs COPY_DST usage');
    }
    const scratch = this.scratchFor(request.count);
    if (!scratch) return null;
    const plan = planDispatch(request.count);
    this.writeUniforms(scratch, request, plan);

    const encoder = this.device.createCommandEncoder({ label: 'luxar depth sort' });
    const pass = encoder.beginComputePass({ label: 'luxar depth sort' });
    const run: RunPass = (name, bindGroup, workgroups) => {
      pass.setPipeline(this.pipelines.pipelines[name]);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroups);
    };
    const groups = scratch.bindGroups;
    run('depth', scratch.depthBindGroup(request.centers), plan.blocks);
    run('keys', groups.keys, plan.blocks);
    for (const radix of groups.passes) {
      run('histogram', radix.histogram, plan.blocks);
      encodeScan(run, groups.scan, plan);
      run(radix.scatterPass, radix.scatter, plan.blocks);
    }
    pass.end();
    encoder.copyBufferToBuffer(scratch.primary, 0, request.target, 0, request.count * 4);
    this.device.queue.submit([encoder.finish()]);
    return { encodeMs: performance.now() - started };
  }

  /** Free the scratch. The sorter refuses every later request. */
  dispose(): void {
    this.disposed = true;
    this.releaseScratch();
  }

  /** Free the scratch until the next sort, which allocates it again at its own size. */
  releaseScratch(): void {
    this.scratch?.destroy();
    this.scratch = null;
  }

  /** Scratch for `count` elements, grown when too small (see {@link planGrowth}). */
  private scratchFor(count: number): SortScratch | null {
    if (this.disposed || !Number.isInteger(count) || count < 1) return null;
    if (this.scratch && this.scratch.plan.capacity >= count) return this.scratch;
    const plan = planGrowth(count, this.scratch?.plan.capacity ?? 0, this.device.limits);
    if (!plan) return null;
    this.scratch?.destroy();
    this.scratch = new SortScratch(this.device, this.pipelines, plan);
    return this.scratch;
  }

  private writeUniforms(scratch: SortScratch, request: DepthSortRequest, plan: DispatchPlan): void {
    const queue = this.device.queue;
    const params = packParams({
      viewZ: viewZRow(request.modelView),
      count: request.count,
      blocks: plan.blocks,
      textureWidth: request.textureWidth,
      texelsPerElement: request.texelsPerElement,
      centerMode: request.centerMode,
    });
    queue.writeBuffer(scratch.uniforms, scratch.slotOffset(PARAMS_SLOT), new Uint8Array(params));
    plan.scanLevels.forEach((len, l) =>
      queue.writeBuffer(
        scratch.uniforms,
        scratch.slotOffset(SCAN_SLOT + l),
        new Uint32Array([len, 0, 0, 0])
      )
    );
    queue.writeBuffer(scratch.bounds, 0, EMPTY_BOUNDS);
  }
}

/** Records one dispatch of a pass. */
type RunPass = (name: DepthSortPass, bindGroup: unknown, workgroups: number) => void;

/**
 * Exclusive scan of the histogram: scan every level's tiles up to the one that
 * fits a workgroup, then add each coarser level's scanned totals back down. The
 * top level's own total is not needed.
 */
function encodeScan(run: RunPass, levelGroups: readonly unknown[], plan: DispatchPlan): void {
  const tiles = plan.scanLevels.map((len) => Math.ceil(len / SCAN_BLOCK_ELEMENTS));
  tiles.forEach((n, l) => run('scanBlocks', levelGroups[l], n));
  for (let l = tiles.length - 2; l >= 0; l--) run('addTotals', levelGroups[l], tiles[l]);
}
