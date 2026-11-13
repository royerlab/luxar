/**
 * Sizes and dispatch counts for the GPU depth sort. Pure, so it is tested
 * without a device.
 *
 * The sort works in BLOCKS of {@link BLOCK_ELEMENTS} elements: one workgroup
 * of {@link WORKGROUP_SIZE} invocations handles one block, each invocation
 * {@link ELEMENTS_PER_INVOCATION} elements. The 16-bit key is sorted in
 * {@link RADIX_PASSES} passes of {@link RADIX_BITS} bits. A pass keeps one
 * {@link RADIX}-bin histogram per block, laid out digit-major
 * (`hist[digit * blocks + block]`), so one exclusive scan of that array gives
 * every block its stable write offset per digit. The scan runs in levels of {@link SCAN_BLOCK_ELEMENTS}
 * entries per workgroup; each level's workgroup totals form the next level,
 * until one workgroup holds the whole level.
 *
 * Scratch buffers are sized for a CAPACITY (the largest node seen so far) and
 * reused; each sort then dispatches only the blocks and scan levels its own
 * element count needs.
 *
 * @module rendering/gpu-depth-sort/plan
 */

import type { GpuLimits } from './gpu-types';

/** Invocations per workgroup, for every pass. */
export const WORKGROUP_SIZE = 256;
/**
 * Elements one invocation handles in a block pass. In the scatter they are
 * CONSECUTIVE, so an invocation ranks its own elements in order and one
 * workgroup scan of per-invocation digit counts ranks the block: that is what
 * makes the scatter stable without sorting inside the workgroup.
 */
export const ELEMENTS_PER_INVOCATION = 16;
/** Elements per block: one workgroup's share of a block pass. */
export const BLOCK_ELEMENTS = WORKGROUP_SIZE * ELEMENTS_PER_INVOCATION;
/**
 * Radix digit width. Four bits keep one invocation's digit counts in eight
 * registers (sixteen 16-bit fields), the size the workgroup scan carries.
 */
export const RADIX_BITS = 4;
/** Bins per radix pass. */
export const RADIX = 1 << RADIX_BITS;
/** Passes over the 16-bit depth key. */
export const RADIX_PASSES = 16 / RADIX_BITS;
/** Entries one scan workgroup sums. */
export const SCAN_BLOCK_ELEMENTS = WORKGROUP_SIZE * 4;

/** What one sort of `count` elements dispatches. */
export interface DispatchPlan {
  readonly count: number;
  /** Workgroups of every block pass. */
  readonly blocks: number;
  /**
   * Entries of each scan level, finest first. Level 0 is the histogram
   * (`RADIX * blocks`); the last level fits one workgroup.
   */
  readonly scanLevels: readonly number[];
}

/** Scratch sizes for a capacity, in bytes. */
export interface ScratchPlan {
  /** Elements the scratch holds, rounded up to whole blocks. */
  readonly capacity: number;
  /** Bytes of each of the four element-sized buffers (two key, two value). */
  readonly elementBytes: number;
  /** Bytes of the histogram. */
  readonly histogramBytes: number;
  /** Bytes of each scan level's totals, finest first. */
  readonly levelTotalsBytes: readonly number[];
  /** Everything above: four element buffers, the histogram and the totals. */
  readonly totalBytes: number;
}

/** The entries of each scan level for a histogram of `entries`, finest first. */
export function scanLevels(entries: number): number[] {
  const levels = [entries];
  let len = entries;
  while (len > SCAN_BLOCK_ELEMENTS) {
    len = Math.ceil(len / SCAN_BLOCK_ELEMENTS);
    levels.push(len);
  }
  return levels;
}

/** The dispatches for one sort of `count` elements (`count >= 1`). */
export function planDispatch(count: number): DispatchPlan {
  const blocks = Math.ceil(count / BLOCK_ELEMENTS);
  return { count, blocks, scanLevels: scanLevels(RADIX * blocks) };
}

/**
 * Scratch sizes for `capacity` elements, or the reason the device cannot hold
 * them: a single binding would exceed `maxStorageBufferBindingSize`, or a pass
 * would need more workgroups than one dispatch dimension allows. A node the
 * GPU cannot sort keeps the CPU worker.
 */
export function planScratch(
  capacity: number,
  limits: GpuLimits
): ScratchPlan | { unsupported: string } {
  if (!Number.isInteger(capacity) || capacity < 1) {
    return { unsupported: `capacity ${capacity} is not a positive integer` };
  }
  const { blocks, scanLevels: levels } = planDispatch(capacity);
  const rounded = blocks * BLOCK_ELEMENTS;
  const elementBytes = rounded * 4;
  const histogramBytes = levels[0] * 4;
  const largestBinding = Math.max(elementBytes, histogramBytes);
  if (largestBinding > limits.maxStorageBufferBindingSize) {
    return {
      unsupported: `${largestBinding} B binding exceeds maxStorageBufferBindingSize ${limits.maxStorageBufferBindingSize}`,
    };
  }
  const widestDispatch = Math.max(blocks, Math.ceil(levels[0] / SCAN_BLOCK_ELEMENTS));
  if (widestDispatch > limits.maxComputeWorkgroupsPerDimension) {
    return {
      unsupported: `${widestDispatch} workgroups exceed maxComputeWorkgroupsPerDimension ${limits.maxComputeWorkgroupsPerDimension}`,
    };
  }
  // Each level's totals hold one entry per scan workgroup of that level.
  const levelTotalsBytes = levels.map((len) => Math.ceil(len / SCAN_BLOCK_ELEMENTS) * 4);
  const totalBytes =
    4 * elementBytes + histogramBytes + levelTotalsBytes.reduce((sum, b) => sum + b, 0);
  return { capacity: rounded, elementBytes, histogramBytes, levelTotalsBytes, totalBytes };
}

/**
 * The scratch to allocate when `count` elements outgrow `capacity`: double the
 * capacity when the device allows it, so a node that keeps appending does not
 * reallocate on every commit, else exactly `count`; null when even that does
 * not fit.
 */
export function planGrowth(count: number, capacity: number, limits: GpuLimits): ScratchPlan | null {
  const doubled = planScratch(Math.max(count, capacity * 2), limits);
  if (!('unsupported' in doubled)) return doubled;
  const exact = planScratch(count, limits);
  return 'unsupported' in exact ? null : exact;
}

/** Where the element's sort center sits in its element-texture texels. */
export type CenterMode = 'texel0' | 'midpoint01';

/** The per-sort uniform: `params` in every shader. Mirrors `Params` in WGSL. */
export interface SortParams {
  /** Row 2 of the column-major model-view matrix: m2, m6, m10, m14. */
  readonly viewZ: readonly [number, number, number, number];
  readonly count: number;
  readonly blocks: number;
  readonly textureWidth: number;
  readonly texelsPerElement: number;
  readonly centerMode: CenterMode;
}

/** Byte size of the WGSL `Params` struct (vec4f + 8 u32, 16-byte aligned). */
export const PARAMS_BYTES = 48;

/** Pack {@link SortParams} into the `Params` uniform layout. */
export function packParams(p: SortParams): ArrayBuffer {
  const buffer = new ArrayBuffer(PARAMS_BYTES);
  new Float32Array(buffer, 0, 4).set(p.viewZ);
  new Uint32Array(buffer, 16, 8).set([
    p.count,
    p.blocks,
    p.textureWidth,
    p.texelsPerElement,
    p.centerMode === 'midpoint01' ? 1 : 0,
  ]);
  return buffer;
}

/** The view-z row of a column-major 4x4 model-view matrix. */
export function viewZRow(modelView: ArrayLike<number>): [number, number, number, number] {
  return [modelView[2], modelView[6], modelView[10], modelView[14]];
}
