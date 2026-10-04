/**
 * Exactness of the post-projection stage cache (#2944 B2).
 *
 * Runs the REAL projection (the shared dispatcher, in-process, on the
 * TypeScript/WASM kernel — no mocked math) and proves:
 *   1. the output committed from the cache on an S-cache revisit is
 *      byte-identical to a fresh projection of the same slice data;
 *   2. a projection-parameter change that is NOT part of the slice key
 *      (gsplats `uTruncate`, lines `extend_to_all` attrs) misses and
 *      re-projects, and the new output replaces the old one;
 *   3. an evicted / upgraded entry takes its stage output with it.
 *
 * The projection count is taken by wrapping the real in-process dispatcher.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

const calls = vi.hoisted(() => ({ gsplats: 0, lines: 0 }));

vi.mock('../../../../workers/data-worker/projection/in-process', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('../../../../workers/data-worker/projection/in-process')>();
  return {
    ...actual,
    projectGSplatsInProcess: (p: Parameters<typeof actual.projectGSplatsInProcess>[0]) => {
      calls.gsplats++;
      return actual.projectGSplatsInProcess(p);
    },
    projectLinesInProcess: (p: Parameters<typeof actual.projectLinesInProcess>[0]) => {
      calls.lines++;
      return actual.projectLinesInProcess(p);
    },
  };
});

// In-process path (no worker pool in node): the stage cache sits above the
// worker/in-process split, so this exercises the same lookup/retain logic.
vi.mock('../../../../config', () => ({
  config: { dataLoading: { performance: { useWebWorkers: false } } },
}));

import { processGSplatsData } from '../../../../data/scene-loader/process/data-processor-gsplats';
import { processLinesData } from '../../../../data/scene-loader/process/data-processor-lines';
import { SliceCache } from '../../../../cache/slice-cache';
import { perfCounters } from '../../../../profiling/perf-counters';
import {
  cloneLodSnapshot,
  restoreLadder,
  storeLadder,
} from '../../../../data/loaders/progressive/slice-cache-helper';
import type {
  GSplatsViewState,
  LoadedGSplatsData,
  ProcessedGSplatsData,
} from '../../../../types/gsplats';
import type { LinesViewState, LoadedLinesData, ProcessedLinesData } from '../../../../types/lines';

/** Deterministic PRNG (mulberry32) so the fixture is reproducible. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * 4D splats whose hidden dim 3 is CONTINUOUS, so the kernel attenuates and
 * CULLS (the projected arrays are short views of worst-case buffers — the
 * tight-copy path is exercised) — with Uint8 RGB colors (coerced) and a
 * published on-disk range (so an element-ID map is composed).
 */
function makeGSplats(n: number): LoadedGSplatsData {
  const r = rng(7);
  const ndim = 4;
  const positions = new Float32Array(n * ndim);
  const chol = new Float32Array(n * 10);
  const amplitudes = new Float32Array(n);
  const colors = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < 3; d++) positions[i * ndim + d] = r() * 100;
    positions[i * ndim + 3] = r() * 10; // hidden continuous coordinate
    // packed lower-triangular [L00, L10, L11, L20, L21, L22, L30, L31, L32, L33]
    const c = i * 10;
    chol[c] = 0.5 + r();
    chol[c + 1] = (r() - 0.5) * 0.2;
    chol[c + 2] = 0.5 + r();
    chol[c + 3] = (r() - 0.5) * 0.2;
    chol[c + 4] = (r() - 0.5) * 0.2;
    chol[c + 5] = 0.5 + r();
    chol[c + 6] = (r() - 0.5) * 0.1;
    chol[c + 7] = (r() - 0.5) * 0.1;
    chol[c + 8] = (r() - 0.5) * 0.1;
    chol[c + 9] = 0.3 + r();
    amplitudes[i] = 0.1 + r();
    colors[i * 3] = Math.floor(r() * 256);
    colors[i * 3 + 1] = Math.floor(r() * 256);
    colors[i * 3 + 2] = Math.floor(r() * 256);
  }
  return {
    positions,
    choleskyFactors: chol,
    amplitudes,
    colors,
    colorComponents: 3,
    splatCount: n,
    ndim,
    ranges: [
      { start: 100, end: 100 + n / 2 },
      { start: 5000, end: 5000 + n / 2 },
    ],
  };
}

const GS_VIEW: GSplatsViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 5],
  tolerance: [1e9, 1e9, 1e9, 2],
  dimensions: [
    { name: 'x', unit: 'px', scale: 1 },
    { name: 'y', unit: 'px', scale: 1 },
    { name: 'z', unit: 'px', scale: 1 },
    { name: 'w', unit: 'px', scale: 1 },
  ],
};

function gsplatsMesh(truncate: number): THREE.Mesh {
  const mesh = new THREE.Mesh();
  mesh.name = '/g';
  mesh.userData = { nodeType: 'gsplats', attrs: {}, visibleSplatCount: 0 };
  (mesh as unknown as { material: unknown }).material = {
    uniforms: { uTruncate: { value: truncate } },
  };
  return mesh;
}

/** Every typed-array field, as raw bytes of exactly the view (not its buffer). */
function bytesOf(obj: object): Record<string, number[]> {
  const out: Record<string, number[]> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (ArrayBuffer.isView(v)) {
      out[k] = Array.from(new Uint8Array(v.buffer, v.byteOffset, v.byteLength));
    }
  }
  return out;
}

/** Non-array fields (counts, bounds, vocabulary), minus the sharing marker. */
function scalarsOf(obj: object): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (!ArrayBuffer.isView(v) && k !== 'sharedBuffers') out[k] = v;
  }
  return out;
}

function expectIdentical(a: object, b: object): void {
  expect(Object.keys(bytesOf(a)).sort()).toEqual(Object.keys(bytesOf(b)).sort());
  expect(bytesOf(a)).toEqual(bytesOf(b));
  expect(scalarsOf(a)).toEqual(scalarsOf(b));
}

async function projectGS(
  data: LoadedGSplatsData,
  root: THREE.Group,
  view: GSplatsViewState = GS_VIEW
): Promise<ProcessedGSplatsData> {
  const staged = await processGSplatsData('/g', data, view, root, 2);
  if (!staged || staged.noop) throw new Error('expected a geometry staged commit');
  return staged.processed;
}

beforeEach(() => {
  calls.gsplats = 0;
  calls.lines = 0;
});

describe('gsplats projection stage cache — exactness', () => {
  it('commits byte-identical buffers from the cache and a fresh projection', async () => {
    const root = new THREE.Group();
    root.add(gsplatsMesh(3.0));
    const cache = new SliceCache({ maxSize: 256 * 1024 * 1024 });
    const source = makeGSplats(4000);
    storeLadder(cache, '/g', GS_VIEW, [source]);
    const restore = () => restoreLadder<LoadedGSplatsData>(cache, '/g', GS_VIEW, 1)![0];
    perfCounters.reset();

    const first = await projectGS(restore(), root); // miss: projects + attaches
    const second = await projectGS(restore(), root); // hit: no projection
    expect(calls.gsplats).toBe(1);
    expect(second).toBe(first);
    expect(second.sharedBuffers).toBe(true);
    // The fixture really culls, so the retained arrays are the tight copies.
    expect(second.splatCount).toBeGreaterThan(0);
    expect(second.splatCount).toBeLessThan(4000);
    expect(second.elementIds).toBeInstanceOf(Uint32Array);
    expect(second.centers3D.buffer.byteLength).toBe(second.centers3D.byteLength);

    // Fresh projections of the SAME content with no S-cache origin — twice,
    // so the reference is itself shown to be deterministic.
    const fresh1 = await projectGS(cloneLodSnapshot([source])[0], root);
    const fresh2 = await projectGS(cloneLodSnapshot([source])[0], root);
    expect(calls.gsplats).toBe(3);
    expect(fresh1.sharedBuffers).toBeUndefined();
    expectIdentical(fresh1, fresh2);
    expectIdentical(second, fresh1);
    expect(cache.getStats().stageHits).toBe(1);
    // The S-cache stage hit and the projection-level hit are the same event.
    expect(perfCounters.get('projection.gsplats.stageHits')).toBe(1);
    expect(perfCounters.get('projection.lines.stageHits')).toBe(0);
  });

  it('misses on a truncation change (outside the slice key) and replaces the output', async () => {
    const root = new THREE.Group();
    const mesh = gsplatsMesh(3.0);
    root.add(mesh);
    const cache = new SliceCache({ maxSize: 256 * 1024 * 1024 });
    const source = makeGSplats(4000);
    storeLadder(cache, '/g', GS_VIEW, [source]);
    const restore = () => restoreLadder<LoadedGSplatsData>(cache, '/g', GS_VIEW, 1)![0];

    const at3 = await projectGS(restore(), root);
    (
      mesh.material as unknown as { uniforms: { uTruncate: { value: number } } }
    ).uniforms.uTruncate.value = 1.5;
    const at15 = await projectGS(restore(), root);
    expect(calls.gsplats).toBe(2);
    // Truncation changes the attenuation (and so the visible set): the miss
    // was necessary, not merely conservative.
    expect(bytesOf(at15)).not.toEqual(bytesOf(at3));
    const fresh15 = await projectGS(cloneLodSnapshot([source])[0], root);
    expectIdentical(at15, fresh15);

    // The 1.5 output replaced the 3.0 one: going back to 3.0 re-projects.
    (
      mesh.material as unknown as { uniforms: { uTruncate: { value: number } } }
    ).uniforms.uTruncate.value = 3.0;
    const back = await projectGS(restore(), root);
    expect(calls.gsplats).toBe(4);
    expectIdentical(back, at3);
    // …and a repeat at 3.0 now hits.
    await projectGS(restore(), root);
    expect(calls.gsplats).toBe(4);
  });

  it('drops the stage output with its slice (eviction and upgrade)', async () => {
    const root = new THREE.Group();
    root.add(gsplatsMesh(3.0));
    const cache = new SliceCache({ maxSize: 256 * 1024 * 1024 });
    const source = makeGSplats(2000);
    storeLadder(cache, '/g', GS_VIEW, [source]);
    const restore = () => restoreLadder<LoadedGSplatsData>(cache, '/g', GS_VIEW, 1)![0];
    await projectGS(restore(), root);
    const staged = cache.getStats();
    expect(staged.stageBytes).toBeGreaterThan(0);

    // An object restored BEFORE the entry is replaced keeps no claim on it.
    const stale = restore();
    cache.clear();
    expect(cache.getStats().stageBytes).toBe(0);
    storeLadder(cache, '/g', GS_VIEW, [source]);
    await projectGS(stale, root);
    expect(calls.gsplats).toBe(2); // the stale origin no longer matches
    await projectGS(restore(), root); // new entry: projects + attaches
    await projectGS(restore(), root); // then hits
    expect(calls.gsplats).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// Lines
// ---------------------------------------------------------------------------

/**
 * 4D segments on a hidden DISCRETE axis (a timelapse: both endpoints share
 * one integer timepoint), so the membership gate really drops most of them.
 */
function makeLines(nSeg: number): LoadedLinesData {
  const r = rng(11);
  const ndim = 4;
  const nVert = nSeg * 2;
  const positions = new Float32Array(nVert * ndim);
  for (let s = 0; s < nSeg; s++) {
    const t = Math.floor(r() * 5); // timepoints 0..4; the view sits at 2
    for (const v of [2 * s, 2 * s + 1]) {
      for (let d = 0; d < 3; d++) positions[v * ndim + d] = r() * 50;
      positions[v * ndim + 3] = t;
    }
  }
  const segments = new Uint32Array(nSeg * 2);
  for (let s = 0; s < nSeg; s++) {
    segments[2 * s] = 2 * s;
    segments[2 * s + 1] = 2 * s + 1;
  }
  const widths = new Float32Array(nVert).map(() => 0.5 + r());
  const colors = new Float32Array(nVert * 3).map(() => r());
  const scalars = new Float32Array(nVert).map(() => r());
  return {
    positions,
    segments,
    widths,
    colors,
    sharpness: null,
    scalars,
    segmentCount: nSeg,
    vertexCount: nVert,
    ndim,
  };
}

const LINES_VIEW: LinesViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 2],
  tolerance: [1e9, 1e9, 1e9, 0.5],
  dimensions: [
    { name: 'x', unit: 'px', scale: 1 },
    { name: 'y', unit: 'px', scale: 1 },
    { name: 'z', unit: 'px', scale: 1 },
    { name: 'w', unit: 's', scale: 1, step: 1, discrete: true },
  ],
};

function linesMesh(extendToAll: string[] = []): THREE.Mesh {
  const mesh = new THREE.Mesh();
  mesh.name = '/l';
  mesh.userData = {
    nodeType: 'lines',
    attrs: extendToAll.length ? { extend_to_all: extendToAll } : {},
    visibleSegmentCount: 0,
  };
  return mesh;
}

async function projectLines(data: LoadedLinesData, root: THREE.Group): Promise<ProcessedLinesData> {
  const staged = await processLinesData('/l', data, LINES_VIEW, root, 2);
  if (!staged || staged.noop) throw new Error('expected a geometry staged commit');
  return staged.processed;
}

describe('lines projection stage cache — exactness', () => {
  it('commits byte-identical buffers from the cache and a fresh projection', async () => {
    const root = new THREE.Group();
    root.add(linesMesh());
    const cache = new SliceCache({ maxSize: 256 * 1024 * 1024 });
    const source = makeLines(3000);
    storeLadder(cache, '/l', LINES_VIEW, [source]);
    const restore = () => restoreLadder<LoadedLinesData>(cache, '/l', LINES_VIEW, 1)![0];
    perfCounters.reset();

    await projectLines(restore(), root);
    const hit = await projectLines(restore(), root);
    expect(calls.lines).toBe(1);
    expect(hit.sharedBuffers).toBe(true);
    expect(hit.segmentCount).toBeGreaterThan(0);
    expect(hit.segmentCount).toBeLessThan(3000);

    const fresh1 = await projectLines(cloneLodSnapshot([source])[0], root);
    const fresh2 = await projectLines(cloneLodSnapshot([source])[0], root);
    expect(calls.lines).toBe(3);
    expectIdentical(fresh1, fresh2);
    expectIdentical(hit, fresh1);
    expect(perfCounters.get('projection.lines.stageHits')).toBe(1);
    expect(perfCounters.get('projection.gsplats.stageHits')).toBe(0);
  });

  it('misses when extend_to_all (mesh attrs, outside the slice key) changes', async () => {
    const root = new THREE.Group();
    const mesh = linesMesh();
    root.add(mesh);
    const cache = new SliceCache({ maxSize: 256 * 1024 * 1024 });
    const source = makeLines(3000);
    storeLadder(cache, '/l', LINES_VIEW, [source]);
    const restore = () => restoreLadder<LoadedLinesData>(cache, '/l', LINES_VIEW, 1)![0];

    const clipped = await projectLines(restore(), root);
    (mesh.userData as { attrs: Record<string, unknown> }).attrs = { extend_to_all: ['w'] };
    const extended = await projectLines(restore(), root);
    expect(calls.lines).toBe(2);
    expect(extended.segmentCount).toBe(3000);
    expect(extended.segmentCount).toBeGreaterThan(clipped.segmentCount);
    const fresh = await projectLines(cloneLodSnapshot([source])[0], root);
    expectIdentical(extended, fresh);
  });
});
