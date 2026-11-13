/** Stable 4-bit LSD radix ordering into three's inactive vertex/storage buffer. */
import type * as THREE from 'three';
import { activeSortedIndexSlot, canComputeSortedIndex } from '../element-storage';
import type { Renderer } from '../renderer-capabilities';

const TILE = 256;
const RADIX = 16;
const PASSES = 4;
const STAGE_BINDINGS = [
  [0, 1, 3, 9],
  [3, 4, 5, 9],
  [5, 6, 7, 9],
  [7, 8],
  [0, 1, 2, 4, 6, 8, 9],
] as const;

/**
 * Workgroup histogram, parallel prefix scan, and stable scatter kernels.
 * @internal
 */
export const ORDER_WGSL = `
struct Params { count: u32, tiles: u32, blocks: u32, shift: u32, initial: u32,
  pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> source: array<u32>;
@group(0) @binding(2) var<storage, read_write> destination: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> block_totals: array<u32>;
@group(0) @binding(6) var<storage, read_write> block_offsets: array<u32>;
@group(0) @binding(7) var<storage, read_write> bin_totals: array<u32>;
@group(0) @binding(8) var<storage, read_write> bin_bases: array<u32>;
@group(0) @binding(9) var<uniform> params: Params;
var<workgroup> masks: array<atomic<u32>, 128>;
var<workgroup> scan_a: array<u32, 256>;
var<workgroup> scan_b: array<u32, 256>;

fn index_at(slot: u32) -> u32 {
  if (params.initial != 0u) { return slot; }
  return source[slot];
}
fn digit_at(index: u32) -> u32 {
  let packed = keys[index / 2u];
  let key = (packed >> ((index & 1u) * 16u)) & 65535u;
  return (key >> params.shift) & 15u;
}
fn make_masks(tile: u32, lane: u32) {
  if (lane < 128u) { atomicStore(&masks[lane], 0u); }
  workgroupBarrier();
  let slot = tile * 256u + lane;
  if (slot < params.count) {
    let digit = digit_at(index_at(slot));
    atomicOr(&masks[digit * 8u + lane / 32u], 1u << (lane & 31u));
  }
  workgroupBarrier();
}
@compute @workgroup_size(256)
fn histogram(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let tile = wg.x;
  make_masks(tile, lane);
  if (lane < 16u) {
    var n = 0u;
    for (var g = 0u; g < 8u; g++) {
      n += countOneBits(atomicLoad(&masks[lane * 8u + g]));
    }
    counts[tile * 16u + lane] = n;
  }
}
fn scan_workgroup(lane: u32) {
  workgroupBarrier();
  for (var step = 1u; step < 256u; step = step * 2u) {
    var value = scan_a[lane];
    if (lane >= step) { value += scan_a[lane - step]; }
    scan_b[lane] = value;
    workgroupBarrier();
    scan_a[lane] = scan_b[lane];
    workgroupBarrier();
  }
}
@compute @workgroup_size(256)
fn scan_tiles(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let block = wg.x;
  let bin = wg.y;
  let tile = block * 256u + lane;
  var value = 0u;
  if (tile < params.tiles) { value = counts[tile * 16u + bin]; }
  scan_a[lane] = value;
  scan_workgroup(lane);
  if (tile < params.tiles) { offsets[tile * 16u + bin] = scan_a[lane] - value; }
  if (lane == 255u) { block_totals[bin * params.blocks + block] = scan_a[lane]; }
}
@compute @workgroup_size(256)
fn scan_blocks(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let bin = wg.x;
  var value = 0u;
  if (lane < params.blocks) { value = block_totals[bin * params.blocks + lane]; }
  scan_a[lane] = value;
  scan_workgroup(lane);
  if (lane < params.blocks) { block_offsets[bin * params.blocks + lane] = scan_a[lane] - value; }
  if (lane == 255u) { bin_totals[bin] = scan_a[lane]; }
}
@compute @workgroup_size(16)
fn scan_bins(@builtin(local_invocation_index) bin: u32) {
  var base = 0u;
  for (var b = 0u; b < bin; b++) { base += bin_totals[b]; }
  bin_bases[bin] = base;
}
@compute @workgroup_size(256)
fn scatter(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  let tile = wg.x;
  make_masks(tile, lane);
  let slot = tile * 256u + lane;
  if (slot >= params.count) { return; }
  let index = index_at(slot);
  let digit = digit_at(index);
  let group = lane / 32u;
  var rank = 0u;
  for (var g = 0u; g < group; g++) {
    rank += countOneBits(atomicLoad(&masks[digit * 8u + g]));
  }
  let preceding = (1u << (lane & 31u)) - 1u;
  rank += countOneBits(atomicLoad(&masks[digit * 8u + group]) & preceding);
  let block = tile / 256u;
  let position = bin_bases[digit] + block_offsets[digit * params.blocks + block]
    + offsets[tile * 16u + digit] + rank;
  destination[position] = index;
}
`;

/** Minimal buffer surface used by the sorter. */
export interface GpuBuffer {
  destroy(): void;
}
/** Compiled compute entry point and its bind group layout. */
export interface GpuPipeline {
  getBindGroupLayout(index: number): unknown;
}
/** Command encoder pass operations needed by the radix stages. */
export interface GpuPass {
  setPipeline(pipeline: GpuPipeline): void;
  setBindGroup(index: number, group: unknown): void;
  dispatchWorkgroups(count: number, y?: number): void;
  end(): void;
}
/** Compute command encoder surface used by the sorter. */
export interface GpuEncoder {
  beginComputePass(): GpuPass;
  finish(): unknown;
}
/** Native WebGPU device operations needed by the sorter. */
export interface GpuDevice {
  limits: { maxStorageBufferBindingSize: number; maxComputeWorkgroupsPerDimension: number };
  queue: {
    writeBuffer(
      buffer: GpuBuffer,
      offset: number,
      data: ArrayBuffer,
      dataOffset: number,
      size: number
    ): void;
    submit(commands: unknown[]): void;
  };
  createBuffer(desc: { size: number; usage: number }): GpuBuffer;
  createShaderModule(desc: { code: string }): unknown;
  createComputePipelineAsync(desc: {
    layout: 'auto';
    compute: { module: unknown; entryPoint: string };
  }): Promise<GpuPipeline>;
  createBindGroup(desc: {
    layout: unknown;
    entries: { binding: number; resource: { buffer: GpuBuffer } }[];
  }): unknown;
  createCommandEncoder(): GpuEncoder;
}
/** Three's native WebGPU backend operations needed by the sorter. */
export interface GpuBackend {
  device: GpuDevice;
  get(attribute: THREE.BufferAttribute): { buffer?: GpuBuffer };
  createStorageAttribute(attribute: THREE.BufferAttribute): void;
}
/** Scratch buffers owned by one geometry and one renderer device. */
export interface GpuResources {
  keys: GpuBuffer;
  scratchA: GpuBuffer;
  scratchB: GpuBuffer;
  counts: GpuBuffer;
  offsets: GpuBuffer;
  blockTotals: GpuBuffer;
  blockOffsets: GpuBuffer;
  binTotals: GpuBuffer;
  binBases: GpuBuffer;
  params: GpuBuffer[];
  onDispose: () => void;
}

function dispatchStage(compute: GpuPass, stage: number, tiles: number, blocks: number): void {
  if (stage === 0 || stage === 4) compute.dispatchWorkgroups(tiles);
  else if (stage === 1) compute.dispatchWorkgroups(blocks, RADIX);
  else if (stage === 2) compute.dispatchWorkgroups(RADIX);
  else compute.dispatchWorkgroups(1);
}

/**
 * A renderer-scoped sorter; resources follow each geometry's dispose event.
 * @internal
 */
export class WebGpuOrderBuilder {
  private pipelines: GpuPipeline[] | null = null;
  private readonly resources = new Map<THREE.InstancedBufferGeometry, GpuResources>();
  private disabled = false;

  constructor(private readonly backend: GpuBackend) {
    const module = backend.device.createShaderModule({ code: ORDER_WGSL });
    void Promise.all(
      ['histogram', 'scan_tiles', 'scan_blocks', 'scan_bins', 'scatter'].map((entryPoint) =>
        backend.device.createComputePipelineAsync({
          layout: 'auto',
          compute: { module, entryPoint },
        })
      )
    ).then(
      (pipelines) => {
        this.pipelines = pipelines;
      },
      () => {
        this.disabled = true;
      }
    );
  }

  get ready(): boolean {
    return !this.disabled && this.pipelines !== null;
  }

  usesBackend(backend: GpuBackend): boolean {
    return this.backend === backend;
  }

  disable(): void {
    this.disabled = true;
  }

  canSort(geometry: THREE.InstancedBufferGeometry, count: number): boolean {
    const limits = this.backend.device.limits;
    const capacity = geometry.getAttribute('aSortedIndex')?.array.length ?? 0;
    const blocks = Math.ceil(Math.ceil(count / TILE) / TILE);
    return (
      this.ready &&
      canComputeSortedIndex(geometry) &&
      count > 0 &&
      count <= capacity &&
      capacity * 4 <= limits.maxStorageBufferBindingSize &&
      Math.ceil(capacity / TILE) * RADIX * 4 <= limits.maxStorageBufferBindingSize &&
      blocks <= TILE &&
      Math.ceil(count / TILE) <= limits.maxComputeWorkgroupsPerDimension
    );
  }

  /** Submit all four passes in queue order, then select the completed buffer. */
  apply(geometry: THREE.InstancedBufferGeometry, keys: Uint16Array, count: number): void {
    if (!this.canSort(geometry, count) || keys.length !== count)
      throw new Error('GPU ordering unavailable');
    const resources = this.resourcesFor(geometry);
    const packed = keys.byteLength % 4 === 0 ? keys : new Uint16Array(keys.length + 1);
    if (packed !== keys) packed.set(keys);
    this.backend.device.queue.writeBuffer(
      resources.keys,
      0,
      packed.buffer as ArrayBuffer,
      packed.byteOffset,
      packed.byteLength
    );

    const inactiveName = activeSortedIndexSlot(geometry) === 0 ? 'aSortedIndexB' : 'aSortedIndex';
    const inactive = geometry.getAttribute(inactiveName) as THREE.BufferAttribute;
    let output = this.backend.get(inactive).buffer;
    if (!output) {
      this.backend.createStorageAttribute(inactive);
      output = this.backend.get(inactive).buffer;
    }
    if (!output) throw new Error('WebGPU ordering attribute has no buffer');
    this.encodeOrdering(resources, output, count);
  }

  private resourcesFor(geometry: THREE.InstancedBufferGeometry): GpuResources {
    const cached = this.resources.get(geometry);
    if (cached) return cached;
    const device = this.backend.device;
    const usage = (globalThis as unknown as { GPUBufferUsage: Record<string, number> })
      .GPUBufferUsage;
    const capacity = geometry.getAttribute('aSortedIndex').array.length;
    const blocks = Math.ceil(Math.ceil(capacity / TILE) / TILE);
    const make = (size: number, extra = 0) =>
      device.createBuffer({ size: Math.max(4, size), usage: usage.STORAGE | extra });
    const resources: GpuResources = {
      keys: make(Math.ceil(capacity / 2) * 4, usage.COPY_DST),
      scratchA: make(capacity * 4),
      scratchB: make(capacity * 4),
      counts: make(Math.ceil(capacity / TILE) * RADIX * 4),
      offsets: make(Math.ceil(capacity / TILE) * RADIX * 4),
      blockTotals: make(blocks * RADIX * 4),
      blockOffsets: make(blocks * RADIX * 4),
      binTotals: make(RADIX * 4),
      binBases: make(RADIX * 4),
      params: Array.from({ length: PASSES }, () => make(32, usage.UNIFORM | usage.COPY_DST)),
      onDispose: () => {
        this.release(geometry);
      },
    };
    geometry.addEventListener('dispose', resources.onDispose);
    this.resources.set(geometry, resources);
    return resources;
  }

  private encodeOrdering(resources: GpuResources, output: GpuBuffer, count: number): void {
    const device = this.backend.device;
    const tiles = Math.ceil(count / TILE);
    const blocks = Math.ceil(tiles / TILE);
    const encoder = device.createCommandEncoder();
    const pipelines = this.pipelines!;
    for (let pass = 0; pass < PASSES; pass++) {
      const source = pass === 0 || pass === 2 ? resources.scratchB : resources.scratchA;
      const target = pass === 3 ? output : pass === 1 ? resources.scratchB : resources.scratchA;
      const params = new Uint32Array([count, tiles, blocks, pass * 4, pass === 0 ? 1 : 0, 0, 0, 0]);
      device.queue.writeBuffer(resources.params[pass], 0, params.buffer, 0, 32);
      const buffers = [
        resources.keys,
        source,
        target,
        resources.counts,
        resources.offsets,
        resources.blockTotals,
        resources.blockOffsets,
        resources.binTotals,
        resources.binBases,
        resources.params[pass],
      ];
      for (let stage = 0; stage < pipelines.length; stage++) {
        const bindGroup = device.createBindGroup({
          layout: pipelines[stage].getBindGroupLayout(0),
          entries: STAGE_BINDINGS[stage].map((binding) => ({
            binding,
            resource: { buffer: buffers[binding] },
          })),
        });
        const compute = encoder.beginComputePass();
        compute.setPipeline(pipelines[stage]);
        compute.setBindGroup(0, bindGroup);
        dispatchStage(compute, stage, tiles, blocks);
        compute.end();
      }
    }
    device.queue.submit([encoder.finish()]);
  }

  private release(geometry: THREE.InstancedBufferGeometry): void {
    const resources = this.resources.get(geometry);
    if (!resources) return;
    geometry.removeEventListener('dispose', resources.onDispose);
    for (const buffer of [
      resources.keys,
      resources.scratchA,
      resources.scratchB,
      resources.counts,
      resources.offsets,
      resources.blockTotals,
      resources.blockOffsets,
      resources.binTotals,
      resources.binBases,
      ...resources.params,
    ])
      buffer.destroy();
    this.resources.delete(geometry);
  }

  dispose(): void {
    this.disabled = true;
    for (const geometry of this.resources.keys()) this.release(geometry);
  }
}

let builder: WebGpuOrderBuilder | null = null;

/**
 * Bind the ordering service to the active renderer's native WebGPU device.
 * @internal
 */
export function configureWebGpuOrderBuilder(renderer: Renderer, native: boolean): void {
  const backend = (renderer as unknown as { backend?: GpuBackend }).backend;
  const computeBackend = native && backend?.device?.createComputePipelineAsync ? backend : null;
  if (computeBackend && builder?.usesBackend(computeBackend)) return;
  builder?.dispose();
  builder = computeBackend ? new WebGpuOrderBuilder(computeBackend) : null;
}

/**
 * Return the compiled sorter, or null while it is unavailable.
 * @internal
 */
export function getWebGpuOrderBuilder(): WebGpuOrderBuilder | null {
  return builder?.ready ? builder : null;
}
