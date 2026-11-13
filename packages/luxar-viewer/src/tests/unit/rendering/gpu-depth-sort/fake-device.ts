/**
 * A recording fake WebGPU device for the GPU depth sort's unit tests: it keeps
 * every buffer, bind-group layout, pipeline, queue write and submitted command
 * stream, so a test can assert what the sorter asked of the GPU.
 */
import type {
  GpuBindGroupLayoutEntry,
  GpuBuffer,
  GpuDevice,
} from '../../../../rendering/gpu-depth-sort/gpu-types';

/** A buffer the fake device created, with its label and whether it was destroyed. */
export interface FakeBuffer extends GpuBuffer {
  label: string;
  destroyed: boolean;
}

/** One recorded compute dispatch: the pass and its workgroup count. */
export interface Dispatch {
  pass: string;
  workgroups: number;
}

export function fakeDevice(options: { failCompile?: boolean; maxBinding?: number } = {}) {
  const buffers: FakeBuffer[] = [];
  const layouts = new Map<object, GpuBindGroupLayoutEntry[]>();
  const pipelineLayouts = new Map<object, object>();
  const pipelines = new Map<object, { pass: string; code: string; layout: object }>();
  const writes: { buffer: FakeBuffer; offset: number; data: Uint8Array }[] = [];
  const submits: { dispatches: Dispatch[]; copies: [FakeBuffer, FakeBuffer, number][] }[] = [];
  const device: GpuDevice = {
    limits: {
      maxStorageBufferBindingSize: options.maxBinding ?? 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
      minUniformBufferOffsetAlignment: 256,
    },
    queue: {
      writeBuffer: (buffer, offset, data) =>
        writes.push({
          buffer: buffer as FakeBuffer,
          offset,
          data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice(),
        }),
      submit: (cbs) => submits.push(...(cbs as (typeof submits)[number][])),
    },
    createBuffer: ({ label = '', size }) => {
      const b: FakeBuffer = {
        label,
        size,
        destroyed: false,
        destroy: () => (b.destroyed = true),
      };
      buffers.push(b);
      return b;
    },
    createShaderModule: ({ code }) => ({
      code,
      getCompilationInfo: async () => ({
        messages: options.failCompile
          ? [{ type: 'error', message: 'unexpected token', lineNum: 3, linePos: 9 }]
          : [],
      }),
    }),
    createBindGroupLayout: ({ entries }) => {
      const layout = {};
      layouts.set(layout, entries);
      return layout;
    },
    createPipelineLayout: ({ bindGroupLayouts }) => {
      const layout = {};
      pipelineLayouts.set(layout, bindGroupLayouts[0] as object);
      return layout;
    },
    createComputePipelineAsync: async ({ label = '', layout, compute }) => {
      if (options.failCompile) throw new Error('invalid shader module');
      const pipeline = {};
      pipelines.set(pipeline, {
        pass: label.replace('luxar depth sort: ', ''),
        code: (compute.module as { code: string }).code,
        layout: pipelineLayouts.get(layout as object)!,
      });
      return pipeline;
    },
    createBindGroup: ({ layout, entries }) => ({ layout, entries }),
    createCommandEncoder: () => {
      const record: (typeof submits)[number] = { dispatches: [], copies: [] };
      let current: { pass: string } | null = null;
      return {
        beginComputePass: () => ({
          setPipeline: (p) => (current = { pass: pipelines.get(p as object)!.pass }),
          setBindGroup: () => {},
          dispatchWorkgroups: (x) => record.dispatches.push({ pass: current!.pass, workgroups: x }),
          end: () => {},
        }),
        copyBufferToBuffer: (src, _so, dst, _do, size) =>
          record.copies.push([src as FakeBuffer, dst as FakeBuffer, size]),
        finish: () => record,
      };
    },
  };
  return { device, buffers, layouts, pipelines, writes, submits };
}
