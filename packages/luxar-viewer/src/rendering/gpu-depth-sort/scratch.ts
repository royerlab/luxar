/**
 * The GPU depth sort's scratch buffers and the bind groups over them, sized for
 * one capacity.
 *
 * Buffers (N = capacity, rounded up to whole blocks):
 *
 * | buffer      | bytes          | role                                                 |
 * | ----------- | -------------- | ---------------------------------------------------- |
 * | `primary`   | 4N             | depth, keys, odd-pass keys, then the final order     |
 * | `keys`      | 4N             | keys after the even radix passes                     |
 * | `values[2]` | 4N each        | element indices between passes (ping-pong)           |
 * | `hist`      | 4·16·N/4096    | digit-major block histograms, then offsets           |
 * | `totals[l]` | small          | each scan level's workgroup totals                   |
 * | `bounds`    | 8              | ordered-u32 min and max of in-front depth            |
 * | `uniforms`  | 256 B per slot | params, the four digit shifts, each scan length      |
 *
 * The four radix passes route keys and indices like this:
 *
 * | pass | keys in   | indices in  | keys out  | indices out |
 * | ---- | --------- | ----------- | --------- | ----------- |
 * | 0    | `primary` | (element i) | `keys`    | `values[0]` |
 * | 1    | `keys`    | `values[0]` | `primary` | `values[1]` |
 * | 2    | `primary` | `values[1]` | `keys`    | `values[0]` |
 * | 3    | `keys`    | `values[0]` | —         | `primary`   |
 *
 * `primary` ends up holding the order and is copied into the target ordering
 * attribute, so the attribute keeps three's own buffer usage. The scratch is
 * shared by every node a sorter serves: each sort writes its uniforms and
 * submits before the next one starts, and the queue runs them in order.
 *
 * @module rendering/gpu-depth-sort/scratch
 */

import { BufferUsage, type GpuBuffer, type GpuDevice, type GpuTexture } from './gpu-types';
import { PARAMS_BYTES, RADIX_BITS, RADIX_PASSES, type ScratchPlan } from './plan';
import type { DepthSortLayout, DepthSortPipelines } from './pipelines';

/** Uniform slot of each scalar block; scan level `l` uses `SCAN_SLOT + l`. */
export const PARAMS_SLOT = 0;
/** The slot holding pass `p`'s digit shift is `SHIFT_SLOT + p`. */
const SHIFT_SLOT = 1;
export const SCAN_SLOT = SHIFT_SLOT + RADIX_PASSES;
/** Byte size of a one-`u32` uniform struct (uniform structs round to 16 B). */
const SCALAR_UNIFORM_BYTES = 16;

/** One radix pass's bind groups. */
export interface RadixPassBindGroups {
  readonly histogram: unknown;
  readonly scatter: unknown;
  readonly scatterPass: 'scatterFirst' | 'scatterMiddle' | 'scatterLast';
}

/** The bind groups that do not depend on the element texture. */
export interface ScratchBindGroups {
  readonly keys: unknown;
  readonly passes: readonly RadixPassBindGroups[];
  /** One per scan level, shared by `scanBlocks` and `addTotals`. */
  readonly scan: readonly unknown[];
}

/** A uniform range bound at `binding`. */
export interface UniformEntry {
  binding: number;
  resource: { buffer: GpuBuffer; offset: number; size: number };
}

export class SortScratch {
  readonly primary: GpuBuffer;
  readonly bounds: GpuBuffer;
  readonly uniforms: GpuBuffer;
  readonly bindGroups: ScratchBindGroups;
  private readonly owned: GpuBuffer[];

  constructor(
    private readonly device: GpuDevice,
    private readonly pipelines: DepthSortPipelines,
    readonly plan: ScratchPlan
  ) {
    const buffer = (label: string, size: number, usage: number): GpuBuffer =>
      device.createBuffer({ label: `luxar depth sort: ${label}`, size, usage });
    const storage = BufferUsage.STORAGE;
    this.primary = buffer('primary', plan.elementBytes, storage | BufferUsage.COPY_SRC);
    const keys = buffer('keys', plan.elementBytes, storage);
    const values = [0, 1].map((v) => buffer(`values ${v}`, plan.elementBytes, storage));
    const hist = buffer('histogram', plan.histogramBytes, storage);
    const totals = plan.levelTotalsBytes.map((bytes, l) => buffer(`totals ${l}`, bytes, storage));
    this.bounds = buffer('bounds', 8, storage | BufferUsage.COPY_DST);
    this.uniforms = buffer(
      'uniforms',
      (SCAN_SLOT + plan.levelTotalsBytes.length) * this.slotBytes,
      BufferUsage.UNIFORM | BufferUsage.COPY_DST
    );
    this.owned = [this.primary, keys, ...values, hist, ...totals, this.bounds, this.uniforms];
    this.writeShifts();
    this.bindGroups = {
      keys: this.group(
        'keys',
        'keys',
        [this.uniformEntry(0, PARAMS_SLOT, PARAMS_BYTES)],
        [this.primary, this.bounds]
      ),
      passes: this.radixPasses(keys, values, hist),
      scan: totals.map((levelTotals, l) =>
        this.group(
          `scan ${l}`,
          'scan',
          [this.uniformEntry(0, SCAN_SLOT + l, SCALAR_UNIFORM_BYTES)],
          [l === 0 ? hist : totals[l - 1], levelTotals]
        )
      ),
    };
  }

  /** Bytes the scratch holds on the GPU. */
  get bytes(): number {
    return this.owned.reduce((sum, b) => sum + b.size, 0);
  }

  /** Byte offset of uniform slot `slot`. */
  slotOffset(slot: number): number {
    return slot * this.slotBytes;
  }

  /** The depth pass's bind group, which binds this sort's element texture. */
  depthBindGroup(centers: GpuTexture): unknown {
    return this.device.createBindGroup({
      label: 'luxar depth sort: depth',
      layout: this.pipelines.layouts.depth,
      entries: [
        this.uniformEntry(0, PARAMS_SLOT, PARAMS_BYTES),
        { binding: 1, resource: centers.createView() },
        { binding: 2, resource: { buffer: this.primary } },
        { binding: 3, resource: { buffer: this.bounds } },
      ],
    });
  }

  destroy(): void {
    for (const b of this.owned) b.destroy();
  }

  private get slotBytes(): number {
    return Math.max(256, this.device.limits.minUniformBufferOffsetAlignment);
  }

  private writeShifts(): void {
    for (let p = 0; p < RADIX_PASSES; p++) {
      this.device.queue.writeBuffer(
        this.uniforms,
        this.slotOffset(SHIFT_SLOT + p),
        new Uint32Array([p * RADIX_BITS, 0, 0, 0])
      );
    }
  }

  private uniformEntry(binding: number, slot: number, size: number): UniformEntry {
    return { binding, resource: { buffer: this.uniforms, offset: this.slotOffset(slot), size } };
  }

  /** A bind group: the uniforms first, then `buffers` at the following bindings. */
  private group(
    label: string,
    layout: DepthSortLayout,
    uniforms: UniformEntry[],
    buffers: GpuBuffer[]
  ): unknown {
    return this.device.createBindGroup({
      label: `luxar depth sort: ${label}`,
      layout: this.pipelines.layouts[layout],
      entries: [
        ...uniforms,
        ...buffers.map((buffer, i) => ({ binding: uniforms.length + i, resource: { buffer } })),
      ],
    });
  }

  /** Each pass's histogram and scatter bind groups, routed as the module note's table. */
  private radixPasses(
    keys: GpuBuffer,
    values: GpuBuffer[],
    hist: GpuBuffer
  ): RadixPassBindGroups[] {
    const passes: RadixPassBindGroups[] = [];
    for (let p = 0; p < RADIX_PASSES; p++) {
      const uniforms = [
        this.uniformEntry(0, PARAMS_SLOT, PARAMS_BYTES),
        this.uniformEntry(1, SHIFT_SLOT + p, SCALAR_UNIFORM_BYTES),
      ];
      const keysIn = p % 2 === 0 ? this.primary : keys;
      const keysOut = p % 2 === 0 ? keys : this.primary;
      const valuesIn = values[(p + 1) % 2];
      const valuesOut = values[p % 2];
      const last = p === RADIX_PASSES - 1;
      const scatterPass = p === 0 ? 'scatterFirst' : last ? 'scatterLast' : 'scatterMiddle';
      const scatterBuffers =
        p === 0
          ? [keysIn, hist, keysOut, valuesOut]
          : last
            ? [keysIn, hist, valuesIn, this.primary]
            : [keysIn, hist, valuesIn, keysOut, valuesOut];
      passes.push({
        histogram: this.group(`histogram ${p}`, 'histogram', uniforms, [keysIn, hist]),
        scatter: this.group(`scatter ${p}`, scatterPass, uniforms, scatterBuffers),
        scatterPass,
      });
    }
    return passes;
  }
}
