/**
 * The GPU depth sorter's orchestration against a recording fake device: which
 * passes it dispatches and how wide, what it binds, what it copies, and how its
 * scratch grows and is freed. Whether the WGSL sorts correctly is the native
 * WebGPU E2E's job (`gpu-depth-sort.spec.ts`); this covers everything around it.
 */
import { describe, expect, it } from 'vitest';

import { GpuDepthSorter } from '../../../../rendering/gpu-depth-sort/gpu-depth-sorter';
import type { GpuBuffer } from '../../../../rendering/gpu-depth-sort/gpu-types';
import { fakeDevice } from './fake-device';
import {
  BLOCK_ELEMENTS,
  RADIX_PASSES,
  planDispatch,
} from '../../../../rendering/gpu-depth-sort/plan';
import { DEPTH_SORT_SHADERS } from '../../../../rendering/gpu-depth-sort/wgsl';

const texture = { createView: () => ({}) };

function request(count: number, target: GpuBuffer) {
  return {
    centers: texture,
    textureWidth: 4096,
    texelsPerElement: 4,
    centerMode: 'texel0' as const,
    count,
    modelView: new Float32Array(16),
    target,
  };
}

const target = (size = 64 * 1024 * 1024): GpuBuffer => ({ size, destroy: () => {} });

describe('GpuDepthSorter', () => {
  it('binds exactly the bindings each shader declares', async () => {
    const fake = fakeDevice();
    await GpuDepthSorter.create(fake.device);
    expect(fake.pipelines.size).toBe(Object.keys(DEPTH_SORT_SHADERS).length);
    for (const { pass, code, layout } of fake.pipelines.values()) {
      const declared = [...code.matchAll(/@binding\((\d+)\)/g)].map((m) => Number(m[1]));
      const bound = fake.layouts.get(layout)!.map((e) => e.binding);
      expect(bound, pass).toEqual([...new Set(declared)].sort((a, b) => a - b));
    }
  });

  it('dispatches depth, keys, then a histogram, scan and scatter per radix pass', async () => {
    const fake = fakeDevice();
    const sorter = await GpuDepthSorter.create(fake.device);
    const count = 5_000_000;
    const out = target();
    expect(sorter.sort(request(count, out))).toEqual({ encodeMs: expect.any(Number) });

    expect(fake.submits).toHaveLength(1);
    const { dispatches, copies } = fake.submits[0];
    const { blocks, scanLevels } = planDispatch(count);
    const tiles = scanLevels.map((len) => Math.ceil(len / 1024));
    const scan = [
      ...tiles.map((n) => ({ pass: 'scanBlocks', workgroups: n })),
      ...tiles
        .slice(0, -1)
        .reverse()
        .map((n) => ({ pass: 'addTotals', workgroups: n })),
    ];
    const scatter = ['scatterFirst', 'scatterMiddle', 'scatterMiddle', 'scatterLast'];
    expect(dispatches).toEqual([
      { pass: 'depth', workgroups: blocks },
      { pass: 'keys', workgroups: blocks },
      ...scatter.flatMap((pass) => [
        { pass: 'histogram', workgroups: blocks },
        ...scan,
        { pass, workgroups: blocks },
      ]),
    ]);
    expect(scatter).toHaveLength(RADIX_PASSES);
    expect(copies).toHaveLength(1);
    expect(copies[0][0].label).toBe('luxar depth sort: primary');
    expect(copies[0][1]).toBe(out);
    expect(copies[0][2]).toBe(count * 4);
  });

  it('resets the depth bounds and writes this sort’s params before submitting', async () => {
    const fake = fakeDevice();
    const sorter = await GpuDepthSorter.create(fake.device);
    sorter.sort(request(1234, target()));
    const bounds = fake.writes.filter((w) => w.buffer.label === 'luxar depth sort: bounds');
    expect(Array.from(new Uint32Array(bounds.at(-1)!.data.buffer))).toEqual([0xffffffff, 0]);
    const params = fake.writes.find(
      (w) => w.buffer.label === 'luxar depth sort: uniforms' && w.offset === 0
    )!;
    expect(new Uint32Array(params.data.buffer, 16, 2)[0]).toBe(1234);
  });

  it('reuses its scratch for smaller sorts and doubles it to grow', async () => {
    const fake = fakeDevice();
    const sorter = await GpuDepthSorter.create(fake.device);
    sorter.sort(request(BLOCK_ELEMENTS * 3, target()));
    const first = sorter.scratchBytes;
    const created = fake.buffers.length;
    sorter.sort(request(BLOCK_ELEMENTS, target()));
    expect(fake.buffers.length).toBe(created);

    sorter.sort(request(BLOCK_ELEMENTS * 4, target()));
    expect(fake.buffers.slice(0, created).every((b) => b.destroyed)).toBe(true);
    expect(sorter.scratchBytes).toBeGreaterThan(first);
    const primary = fake.buffers.filter((b) => b.label === 'luxar depth sort: primary').at(-1)!;
    expect(primary.size).toBe(BLOCK_ELEMENTS * 6 * 4);
  });

  it('refuses what the device limits cannot hold, and everything once disposed', async () => {
    const fake = fakeDevice({ maxBinding: BLOCK_ELEMENTS * 4 * 4 });
    const sorter = await GpuDepthSorter.create(fake.device);
    expect(sorter.accepts(BLOCK_ELEMENTS * 4)).toBe(true);
    expect(sorter.accepts(BLOCK_ELEMENTS * 4 + 1)).toBe(false);
    expect(sorter.sort(request(BLOCK_ELEMENTS * 4 + 1, target()))).toBeNull();
    expect(sorter.sort(request(0, target()))).toBeNull();

    sorter.sort(request(10, target()));
    sorter.dispose();
    expect(fake.buffers.every((b) => b.destroyed)).toBe(true);
    expect(sorter.scratchBytes).toBe(0);
    expect(sorter.accepts(10)).toBe(false);
    expect(sorter.sort(request(10, target()))).toBeNull();
  });

  it('reports the compiler messages when a pipeline fails to build', async () => {
    const fake = fakeDevice({ failCompile: true });
    await expect(GpuDepthSorter.create(fake.device)).rejects.toThrow(/3:9 unexpected token/);
  });
});
