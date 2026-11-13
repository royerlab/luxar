import * as THREE from 'three';
import { WebGpuOrderBuilder } from '../../../rendering/depth-sort-coordinator/webgpu-order';
import {
  attachElementStorage,
  configureSortedIndexGpuCompute,
} from '../../../rendering/element-storage';
import { POINT_TEXTURE_LAYOUT } from '../../../rendering/element-texture-layout';

interface ProbeBuffer {
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

interface ProbeDevice {
  queue: {
    writeBuffer(buffer: ProbeBuffer, offset: number, data: ArrayBufferView): void;
    submit(commands: unknown[]): void;
  };
  createBuffer(desc: { size: number; usage: number }): ProbeBuffer;
  createCommandEncoder(): {
    copyBufferToBuffer(
      source: ProbeBuffer,
      sourceOffset: number,
      target: ProbeBuffer,
      targetOffset: number,
      size: number
    ): void;
    finish(): unknown;
  };
}

export async function probe() {
  const webgpu = (
    navigator as unknown as {
      gpu: { requestAdapter(): Promise<{ requestDevice(): Promise<ProbeDevice> } | null> };
    }
  ).gpu;
  const adapter = await webgpu.requestAdapter();
  if (!adapter) throw new Error('native WebGPU adapter unavailable');
  const device = await adapter.requestDevice();
  const usage = (globalThis as unknown as { GPUBufferUsage: Record<string, number> })
    .GPUBufferUsage;
  const mapMode = (globalThis as unknown as { GPUMapMode: { READ: number } }).GPUMapMode;
  const records = new WeakMap<object, { buffer: ProbeBuffer }>();
  const backend = {
    device,
    get: (attribute: object) => records.get(attribute) ?? {},
    createStorageAttribute: (attribute: THREE.BufferAttribute) => {
      const buffer = device.createBuffer({
        size: attribute.array.byteLength,
        usage: usage.STORAGE | usage.VERTEX | usage.COPY_DST | usage.COPY_SRC,
      });
      device.queue.writeBuffer(buffer, 0, attribute.array);
      records.set(attribute, { buffer });
    },
  };
  const builder = new WebGpuOrderBuilder(backend as never);
  for (let i = 0; i < 100 && !builder.ready; i++) await new Promise((r) => setTimeout(r, 10));
  if (!builder.ready) throw new Error('builder did not compile');
  const results = [];
  for (const n of [1, 2, 255, 256, 257, 1000, 1024, 4097, 65537]) {
    configureSortedIndexGpuCompute(true);
    const geometry = new THREE.InstancedBufferGeometry();
    attachElementStorage(geometry, n, POINT_TEXTURE_LAYOUT);
    const keys = new Uint16Array(n);
    for (let j = 0; j < n; j++) {
      keys[j] = n === 1024 ? 0 : n % 2 ? (j * 7919) & 65535 : (j * 7919 + (j % 7) * 13) % 100;
    }
    builder.apply(geometry, keys, n);
    const attr = geometry.getAttribute('aSortedIndexB') as THREE.BufferAttribute;
    const output = records.get(attr)!.buffer;
    const readback = device.createBuffer({
      size: n * 4,
      usage: usage.COPY_DST | usage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(output, 0, readback, 0, n * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(mapMode.READ);
    const actual = Array.from(new Uint32Array(readback.getMappedRange()).slice(0, n));
    const expected = Array.from({ length: n }, (_, j) => j).sort(
      (a, b) => keys[a] - keys[b] || a - b
    );
    results.push({
      n,
      equal: JSON.stringify(actual) === JSON.stringify(expected),
      actual: actual.slice(0, 10),
      expected: expected.slice(0, 10),
    });
    readback.unmap();
    readback.destroy();
    geometry.dispose();
  }
  builder.dispose();
  return results;
}
