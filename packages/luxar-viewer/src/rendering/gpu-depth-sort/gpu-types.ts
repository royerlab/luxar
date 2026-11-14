/**
 * The slice of the WebGPU API the depth sort uses, typed structurally.
 *
 * The viewer has no `@webgpu/types` dependency (see the same choice in
 * `scene/scene-manager/render-pipeline/webgpu-device-recovery.ts`), and unit
 * tests run in Node without a GPU, so the sorter is written against these
 * interfaces. A real `GPUDevice` satisfies them; a test fake implements only
 * what it records.
 *
 * @module rendering/gpu-depth-sort/gpu-types
 */

/** `GPUBufferUsage` flags. The values are fixed by the WebGPU specification. */
export const BufferUsage = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
} as const;

/** `GPUShaderStage.COMPUTE`, fixed by the WebGPU specification. */
export const SHADER_STAGE_COMPUTE = 0x4;

/** A GPU buffer the sorter allocates, binds, or copies into. */
export interface GpuBuffer {
  readonly size: number;
  readonly usage: number;
  destroy(): void;
}

/** A sampled texture the sorter reads element centers from. */
export interface GpuTexture {
  createView(): unknown;
}

/** One bind-group-layout entry; only the forms the sorter uses. */
export interface GpuBindGroupLayoutEntry {
  binding: number;
  visibility: number;
  buffer?: { type: 'uniform' | 'storage' | 'read-only-storage' };
  texture?: { sampleType: 'unfilterable-float' };
}

/** One bind-group entry: a buffer range or a texture view. */
export interface GpuBindGroupEntry {
  binding: number;
  resource: { buffer: GpuBuffer; offset?: number; size?: number } | unknown;
}

export interface GpuComputePassEncoder {
  setPipeline(pipeline: unknown): void;
  setBindGroup(index: number, bindGroup: unknown): void;
  dispatchWorkgroups(x: number): void;
  end(): void;
}

export interface GpuCommandEncoder {
  beginComputePass(descriptor?: { label?: string }): GpuComputePassEncoder;
  copyBufferToBuffer(
    source: GpuBuffer,
    sourceOffset: number,
    destination: GpuBuffer,
    destinationOffset: number,
    size: number
  ): void;
  finish(): unknown;
}

export interface GpuQueue {
  writeBuffer(buffer: GpuBuffer, offset: number, data: ArrayBufferView): void;
  submit(commandBuffers: unknown[]): void;
}

/** The device limits the sorter plans against. */
export interface GpuLimits {
  readonly maxStorageBufferBindingSize: number;
  readonly maxComputeWorkgroupsPerDimension: number;
  readonly minUniformBufferOffsetAlignment: number;
}

/** A `GPUDevice`, reduced to what the sorter calls. */
export interface GpuDevice {
  readonly limits: GpuLimits;
  /** Resolves when the device is lost (absent on test fakes). */
  readonly lost?: Promise<unknown>;
  readonly queue: GpuQueue;
  createBuffer(descriptor: { label?: string; size: number; usage: number }): GpuBuffer;
  createShaderModule(descriptor: { label?: string; code: string }): unknown;
  createBindGroupLayout(descriptor: {
    label?: string;
    entries: GpuBindGroupLayoutEntry[];
  }): unknown;
  createPipelineLayout(descriptor: { label?: string; bindGroupLayouts: unknown[] }): unknown;
  createComputePipelineAsync(descriptor: {
    label?: string;
    layout: unknown;
    compute: { module: unknown; entryPoint: string };
  }): Promise<unknown>;
  createBindGroup(descriptor: {
    label?: string;
    layout: unknown;
    entries: GpuBindGroupEntry[];
  }): unknown;
  createCommandEncoder(descriptor?: { label?: string }): GpuCommandEncoder;
}
