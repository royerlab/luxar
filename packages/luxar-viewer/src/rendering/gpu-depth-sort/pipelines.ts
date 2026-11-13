/**
 * The compute pipelines of the GPU depth sort and their bind-group layouts.
 *
 * Layouts are explicit rather than `'auto'`: the element texture is RGBA32F,
 * which binds only as `unfilterable-float`, and the two scan entry points must
 * share one layout so one bind group per scan level serves both.
 *
 * @module rendering/gpu-depth-sort/pipelines
 */

import { SHADER_STAGE_COMPUTE, type GpuBindGroupLayoutEntry, type GpuDevice } from './gpu-types';
import { DEPTH_SORT_SHADERS, type DepthSortPass } from './wgsl';

const uniform = (binding: number): GpuBindGroupLayoutEntry => ({
  binding,
  visibility: SHADER_STAGE_COMPUTE,
  buffer: { type: 'uniform' },
});
const readOnly = (binding: number): GpuBindGroupLayoutEntry => ({
  binding,
  visibility: SHADER_STAGE_COMPUTE,
  buffer: { type: 'read-only-storage' },
});
const readWrite = (binding: number): GpuBindGroupLayoutEntry => ({
  binding,
  visibility: SHADER_STAGE_COMPUTE,
  buffer: { type: 'storage' },
});
const centers = (binding: number): GpuBindGroupLayoutEntry => ({
  binding,
  visibility: SHADER_STAGE_COMPUTE,
  texture: { sampleType: 'unfilterable-float' },
});

/** A kind of bind group. */
export type DepthSortLayout =
  'depth' | 'keys' | 'histogram' | 'scan' | 'scatterFirst' | 'scatterMiddle' | 'scatterLast';

/** Layout of each kind of bind group; the binding numbers match `wgsl.ts`. */
const LAYOUTS: Record<DepthSortLayout, readonly GpuBindGroupLayoutEntry[]> = {
  depth: [uniform(0), centers(1), readWrite(2), readWrite(3)],
  keys: [uniform(0), readWrite(1), readOnly(2)],
  histogram: [uniform(0), uniform(1), readOnly(2), readWrite(3)],
  scan: [uniform(0), readWrite(1), readWrite(2)],
  scatterFirst: [uniform(0), uniform(1), readOnly(2), readOnly(3), readWrite(4), readWrite(5)],
  scatterMiddle: [
    uniform(0),
    uniform(1),
    readOnly(2),
    readOnly(3),
    readOnly(4),
    readWrite(5),
    readWrite(6),
  ],
  scatterLast: [uniform(0), uniform(1), readOnly(2), readOnly(3), readOnly(4), readWrite(5)],
};

/** The layout each pass binds. */
const PASS_LAYOUT: Record<DepthSortPass, DepthSortLayout> = {
  depth: 'depth',
  keys: 'keys',
  histogram: 'histogram',
  scanBlocks: 'scan',
  addTotals: 'scan',
  scatterFirst: 'scatterFirst',
  scatterMiddle: 'scatterMiddle',
  scatterLast: 'scatterLast',
};

/** Compiled pipelines plus the layouts their bind groups are made against. */
export interface DepthSortPipelines {
  readonly layouts: Readonly<Record<DepthSortLayout, unknown>>;
  readonly pipelines: Readonly<Record<DepthSortPass, unknown>>;
}

/**
 * Compile every pass. Asynchronous (`createComputePipelineAsync`) so pipeline
 * compilation never blocks a frame; the sorter exists only once it resolves.
 */
export async function createDepthSortPipelines(device: GpuDevice): Promise<DepthSortPipelines> {
  const layouts = {} as Record<DepthSortLayout, unknown>;
  for (const name of Object.keys(LAYOUTS) as DepthSortLayout[]) {
    layouts[name] = device.createBindGroupLayout({
      label: `luxar depth sort: ${name}`,
      entries: [...LAYOUTS[name]],
    });
  }
  const modules = new Map<string, unknown>();
  const moduleFor = (code: string, label: string): unknown => {
    let module = modules.get(code);
    if (!module) {
      module = device.createShaderModule({ label: `luxar depth sort: ${label}`, code });
      modules.set(code, module);
    }
    return module;
  };
  const passes = Object.keys(DEPTH_SORT_SHADERS) as DepthSortPass[];
  const compiled = await Promise.all(
    passes.map((pass) => {
      const { code, entryPoint } = DEPTH_SORT_SHADERS[pass];
      return device.createComputePipelineAsync({
        label: `luxar depth sort: ${pass}`,
        layout: device.createPipelineLayout({
          label: `luxar depth sort: ${pass}`,
          bindGroupLayouts: [layouts[PASS_LAYOUT[pass]]],
        }),
        compute: { module: moduleFor(code, pass), entryPoint },
      });
    })
  ).catch(async (error: unknown) => {
    throw new Error(`${String(error)}${await compilationMessages(modules)}`);
  });
  const pipelines = {} as Record<DepthSortPass, unknown>;
  passes.forEach((pass, i) => (pipelines[pass] = compiled[i]));
  return { layouts, pipelines };
}

/**
 * The compiler's error messages for every module, so a pipeline that fails to
 * build says why (a WGSL error otherwise surfaces only as "invalid module").
 */
async function compilationMessages(modules: Map<string, unknown>): Promise<string> {
  const lines: string[] = [];
  for (const module of modules.values()) {
    const info = await (
      module as { getCompilationInfo?: () => Promise<{ messages: readonly CompilationMessage[] }> }
    ).getCompilationInfo?.();
    for (const m of info?.messages ?? []) {
      if (m.type === 'error') lines.push(`${m.lineNum}:${m.linePos} ${m.message}`);
    }
  }
  return lines.length ? `\n${lines.join('\n')}` : '';
}

interface CompilationMessage {
  readonly type: string;
  readonly message: string;
  readonly lineNum: number;
  readonly linePos: number;
}
