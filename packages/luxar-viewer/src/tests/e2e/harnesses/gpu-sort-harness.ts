/**
 * GPU depth-sort harness (`/gpu-sort-harness.html`).
 *
 * Runs {@link GpuDepthSorter} on a real WebGPU device against synthetic element
 * textures and checks its order against the CPU kernel the SortWorker runs
 * (`wasm/typescript/depth-sort.ts`), inside the page, so a 10M-element case
 * returns a summary rather than 40 MB of indices.
 *
 * Exposes `window.__gpuSortHarness.run(spec)`. It fails closed without a
 * hardware adapter unless the URL carries `?allowFallbackAdapter`.
 *
 * @module tests/e2e/harnesses/gpu-sort-harness
 */

import * as THREE from 'three';

import { GpuDepthSorter } from '../../../rendering/gpu-depth-sort/gpu-depth-sorter';
import type { GpuBuffer, GpuDevice, GpuTexture } from '../../../rendering/gpu-depth-sort/gpu-types';
import type { CenterMode } from '../../../rendering/gpu-depth-sort/plan';
import { forwardedAdapterLimits } from '../../../scene/scene-manager/render-pipeline/early-device-request';
import { sort_splats_by_depth } from '../../../wasm/typescript/depth-sort';

/** One harness case. */
export interface GpuSortCase {
  count: number;
  texelsPerElement: number;
  centerMode: CenterMode;
  seed: number;
  /** `cloud`: in front; `straddle`: the camera sits inside; `behind`; `flat`: one depth. */
  layout: 'cloud' | 'straddle' | 'behind' | 'flat';
  /** Timed repeats after the checked sort (0 = none). */
  repeats?: number;
}

/** What the page reports for one case. */
export interface GpuSortReport {
  /** The GPU order is a permutation of `[0, count)`. */
  permutation: boolean;
  /** Positions where the GPU and CPU orders agree exactly. */
  samePositions: number;
  /** Adjacent GPU-order pairs whose CPU keys step DOWN (should be ~0). */
  descents: number;
  /** Largest such step, in CPU key buckets. */
  maxDescent: number;
  /** Adjacent pairs with equal CPU key but decreasing index (stability). */
  tieInversions: number;
  /** Mean ms per sort over back-to-back timed repeats. */
  gpuMs: number | null;
  adapter: string;
}

declare global {
  interface Window {
    __gpuSortHarness?: { ready: Promise<void>; run(spec: GpuSortCase): Promise<GpuSortReport> };
  }
}

interface DeviceLike extends GpuDevice {
  createTexture(d: { size: [number, number]; format: string; usage: number }): GpuTexture & {
    destroy(): void;
  };
  queue: GpuDevice['queue'] & {
    writeTexture(
      dst: { texture: unknown },
      data: ArrayBufferView,
      layout: { bytesPerRow: number; rowsPerImage: number },
      size: [number, number]
    ): void;
    onSubmittedWorkDone(): Promise<void>;
  };
}

const TEXTURE_BINDING = 0x04;
const COPY_DST_TEXTURE = 0x02;
const MAX_WIDTH = 4096;

let deviceInfo: { device: DeviceLike; adapter: string } | null = null;

async function acquireDevice(): Promise<{ device: DeviceLike; adapter: string }> {
  const gpu = (navigator as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) throw new Error('navigator.gpu is unavailable');
  const adapter = (await gpu.requestAdapter()) as {
    info?: { vendor?: string; description?: string; isFallbackAdapter?: boolean };
    limits?: Parameters<typeof forwardedAdapterLimits>[0];
    requestDevice(d?: unknown): Promise<DeviceLike>;
  } | null;
  if (!adapter) throw new Error('no WebGPU adapter');
  const allowFallback = new URLSearchParams(location.search).has('allowFallbackAdapter');
  if (adapter.info?.isFallbackAdapter === true && !allowFallback) {
    throw new Error(`software fallback adapter (${adapter.info.description ?? '?'})`);
  }
  // The viewer's own device request: the adapter's buffer and texture limits
  // (a 10M-splat element texture is 9,766 rows, past the 8,192 default).
  const device = await adapter.requestDevice({
    requiredLimits: forwardedAdapterLimits(adapter.limits),
  });
  return { device, adapter: `${adapter.info?.vendor ?? '?'} ${adapter.info?.description ?? ''}` };
}

/** A deterministic PRNG (mulberry32). */
function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Element texture texels plus the CPU sort's centers, for one case. */
function buildElements(spec: GpuSortCase): {
  texels: Float32Array;
  width: number;
  height: number;
  centers3: Float32Array;
} {
  const width = Math.floor(MAX_WIDTH / spec.texelsPerElement) * spec.texelsPerElement;
  const height = Math.max(1, Math.ceil((spec.count * spec.texelsPerElement) / width));
  const texels = new Float32Array(width * height * 4);
  const centers3 = new Float32Array(spec.count * 3);
  const next = random(spec.seed);
  const coordinate = () => (spec.layout === 'flat' ? 0 : next() * 2 - 1);
  for (let i = 0; i < spec.count; i++) {
    const base = i * spec.texelsPerElement * 4;
    const a = [coordinate(), coordinate(), spec.layout === 'flat' ? 0.25 : coordinate()];
    texels.set(a, base);
    if (spec.centerMode === 'midpoint01') {
      const b = [coordinate(), coordinate(), spec.layout === 'flat' ? 0.25 : coordinate()];
      texels.set(b, base + 4);
      for (let k = 0; k < 3; k++) centers3[i * 3 + k] = (a[k] + b[k]) * 0.5;
    } else {
      centers3.set(texels.subarray(base, base + 3), i * 3);
    }
  }
  return { texels, width, height, centers3 };
}

/** A model-view matrix placing the camera for the case's layout. */
function modelViewFor(spec: GpuSortCase): Float32Array {
  const eye = {
    cloud: new THREE.Vector3(1.3, 0.8, 3.5),
    straddle: new THREE.Vector3(0.2, 0.1, 0.3),
    behind: new THREE.Vector3(0, 0, -4),
    flat: new THREE.Vector3(0, 0, 3),
  }[spec.layout];
  const target = spec.layout === 'behind' ? new THREE.Vector3(0, 0, -8) : new THREE.Vector3();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
  camera.position.copy(eye);
  camera.lookAt(target);
  camera.updateMatrixWorld(true);
  return new Float32Array(camera.matrixWorldInverse.elements);
}

/** The CPU sort's keys, recomputed with its exact arithmetic (depth-sort.ts). */
function cpuKeys(centers3: Float32Array, mv: Float32Array, count: number): Uint16Array {
  const f = Math.fround;
  const z = new Float32Array(count);
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < count; i++) {
    const b = i * 3;
    z[i] = f(
      f(f(f(mv[2] * centers3[b]) + f(mv[6] * centers3[b + 1])) + f(mv[10] * centers3[b + 2])) +
        mv[14]
    );
    if (z[i] < 0) {
      lo = Math.min(lo, z[i]);
      hi = Math.max(hi, z[i]);
    }
  }
  const keys = new Uint16Array(count);
  if (!(hi > lo)) return keys;
  const inv = f(1 / f(hi - lo));
  for (let i = 0; i < count; i++) {
    if (z[i] >= 0) continue;
    const scaled = f(f(f(z[i] - lo) * inv) * 65535);
    keys[i] = (scaled < 65535 ? scaled : 65535) | 0;
  }
  return keys;
}

function compare(
  gpu: Uint32Array,
  cpu: Uint32Array,
  keys: Uint16Array
): Omit<GpuSortReport, 'gpuMs' | 'adapter'> {
  const n = gpu.length;
  const seen = new Uint8Array(n);
  let permutation = true;
  let samePositions = 0;
  let descents = 0;
  let maxDescent = 0;
  let tieInversions = 0;
  for (let j = 0; j < n; j++) {
    const v = gpu[j];
    if (v >= n || seen[v]) permutation = false;
    else seen[v] = 1;
    if (v === cpu[j]) samePositions++;
    if (j > 0 && v < n && gpu[j - 1] < n) {
      const step = keys[v] - keys[gpu[j - 1]];
      if (step < 0) {
        descents++;
        maxDescent = Math.max(maxDescent, -step);
      } else if (step === 0 && v < gpu[j - 1]) {
        tieInversions++;
      }
    }
  }
  return { permutation, samePositions, descents, maxDescent, tieInversions };
}

async function readBack(
  device: DeviceLike,
  source: GpuBuffer,
  bytes: number
): Promise<Uint32Array> {
  const staging = device.createBuffer({ size: bytes, usage: 0x0001 | 0x0008 }) as GpuBuffer & {
    mapAsync(mode: number): Promise<void>;
    getMappedRange(): ArrayBuffer;
    unmap(): void;
  };
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(0x0001);
  const out = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return out;
}

async function run(spec: GpuSortCase): Promise<GpuSortReport> {
  deviceInfo ??= await acquireDevice();
  const { device, adapter } = deviceInfo;
  const sorter = await GpuDepthSorter.create(device);
  const { texels, width, height, centers3 } = buildElements(spec);
  const texture = device.createTexture({
    size: [width, height],
    format: 'rgba32float',
    usage: TEXTURE_BINDING | COPY_DST_TEXTURE,
  });
  device.queue.writeTexture(
    { texture },
    texels,
    { bytesPerRow: width * 16, rowsPerImage: height },
    [width, height]
  );
  const target = device.createBuffer({ size: spec.count * 4, usage: 0x0020 | 0x0008 | 0x0004 });
  const modelView = modelViewFor(spec);
  const request = {
    centers: texture,
    textureWidth: width,
    texelsPerElement: spec.texelsPerElement,
    centerMode: spec.centerMode,
    count: spec.count,
    modelView,
    target,
  };
  if (!sorter.sort(request)) throw new Error(`sorter refused ${spec.count} elements`);
  const gpu = await readBack(device, target, spec.count * 4);
  const cpu = new Uint32Array(spec.count);
  sort_splats_by_depth(centers3, modelView, cpu, spec.count);
  const report = compare(gpu, cpu, cpuKeys(centers3, modelView, spec.count));

  // Submit the repeats back to back and wait once: a single sort's submit-to-
  // idle time is dominated by how often the browser polls the queue.
  let gpuMs: number | null = null;
  const repeats = spec.repeats ?? 0;
  if (repeats > 0) {
    await device.queue.onSubmittedWorkDone();
    const t0 = performance.now();
    for (let r = 0; r < repeats; r++) sorter.sort(request);
    await device.queue.onSubmittedWorkDone();
    gpuMs = (performance.now() - t0) / repeats;
  }
  sorter.dispose();
  texture.destroy();
  target.destroy();
  return { ...report, gpuMs, adapter };
}

window.__gpuSortHarness = { ready: Promise.resolve(), run };
const status = document.getElementById('status');
if (status) status.textContent = 'gpu-sort-harness ready';
