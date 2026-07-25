/**
 * Arena-vs-reference equivalence for the progressive-ladder concat (perf
 * lever L2), across all three geometries.
 *
 * The per-geometry LadderArenas build the ladder concat INCREMENTALLY
 * (appending level k copies only level k's bytes); the exported
 * `concatenate{GSplats,Points,Lines}Data` functions remain the REFERENCE
 * full-rebuild implementations. This suite pins, for simulated 6-level
 * ladders with every geometry-specific wrinkle (white color fill, late
 * sharpness backfill, all-or-nothing drops, segment index offsetting):
 *
 *  (a) byte-equality of every incremental snapshot against the reference
 *      rebuild of the same prefix,
 *  (b) prefix stability across appends (earlier snapshots' bytes never
 *      change; later snapshots extend them in place),
 *  (c) fresh-arena isolation (a new generation's arena never disturbs an
 *      old arena's snapshots),
 *  (d) copy-work accounting — appending level k copies O(N_k), not
 *      O(N_total) (the whole point of the lever), with growth-realloc
 *      copies amortized,
 *  (e) trim-on-ladder-complete (steady-state memory equals the exact-size
 *      reference concat).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  GSplatsLadderArena,
  concatenateGSplatsData,
} from '../../../../../data/gsplats/gsplats-progressive-loader';
import {
  PointsLadderArena,
  concatenatePointsData,
} from '../../../../../data/points/points-progressive-loader';
import {
  LinesLadderArena,
  concatenateLinesData,
} from '../../../../../data/lines/lines-progressive-loader';
import type { LoadedGSplatsData } from '../../../../../types/gsplats';
import type { LoadedPointsData } from '../../../../../types/points';
import type { LoadedLinesData } from '../../../../../types/lines';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Deterministic pseudo-random fill so byte-equality is meaningful. */
function fillPattern(arr: { length: number } & Record<number, number>, seed: number): void {
  for (let i = 0; i < arr.length; i++) {
    arr[i] = ((seed * 31 + i * 7) % 251) / 251;
  }
}

function fillPatternInt(arr: { length: number } & Record<number, number>, seed: number): void {
  for (let i = 0; i < arr.length; i++) {
    arr[i] = (seed * 31 + i * 7) % 251;
  }
}

function expectSameArray(
  actual: ArrayLike<number> | null | undefined,
  expected: ArrayLike<number> | null | undefined,
  label: string
): void {
  if (expected == null || actual == null) {
    expect(actual ?? null, label).toBe(expected ?? null);
    return;
  }
  expect(actual.constructor, `${label} dtype`).toBe(expected.constructor);
  expect(actual.length, `${label} length`).toBe(expected.length);
  expect(Array.from(actual as ArrayLike<number>), label).toEqual(
    Array.from(expected as ArrayLike<number>)
  );
}

// ---------------------------------------------------------------------------
// GSplats
// ---------------------------------------------------------------------------

function makeGSplatsLod(
  splatCount: number,
  seed: number,
  opts: { colors?: boolean } = { colors: true }
): LoadedGSplatsData {
  const ndim = 3;
  const positions = new Float32Array(splatCount * ndim);
  const amplitudes = new Float32Array(splatCount);
  const choleskyFactors = new Float32Array(splatCount * 6);
  fillPattern(positions, seed);
  fillPattern(amplitudes, seed + 1);
  fillPattern(choleskyFactors, seed + 2);
  let colors: Uint8Array | null = null;
  if (opts.colors) {
    colors = new Uint8Array(splatCount * 3);
    fillPatternInt(colors, seed + 3);
  }
  return { positions, amplitudes, choleskyFactors, colors, splatCount, ndim };
}

function expectGSplatsEqual(actual: LoadedGSplatsData, expected: LoadedGSplatsData): void {
  expectSameArray(actual.positions, expected.positions, 'positions');
  expectSameArray(actual.amplitudes, expected.amplitudes, 'amplitudes');
  expectSameArray(actual.choleskyFactors, expected.choleskyFactors, 'choleskyFactors');
  expectSameArray(actual.colors, expected.colors, 'colors');
  expect(actual.colorComponents ?? 3).toBe(expected.colorComponents ?? 3);
  expect(actual.splatCount).toBe(expected.splatCount);
  expect(actual.ndim).toBe(expected.ndim);
}

describe('GSplatsLadderArena — reference equivalence (6-level ladder)', () => {
  // Level 2 is colorless (white fill); sizes vary to exercise growth.
  const parts = [
    makeGSplatsLod(100, 1),
    makeGSplatsLod(60, 2),
    makeGSplatsLod(40, 3, { colors: false }),
    makeGSplatsLod(90, 4),
    makeGSplatsLod(10, 5),
    makeGSplatsLod(70, 6),
  ];

  it('(a) every incremental snapshot is byte-identical to the reference rebuild', () => {
    const arena = new GSplatsLadderArena();
    for (let k = 2; k <= parts.length; k++) {
      arena.appendThrough(parts.slice(0, k), k === parts.length);
      expectGSplatsEqual(arena.snapshot(), concatenateGSplatsData(parts.slice(0, k)));
    }
  });

  it('(b) prefix stability: earlier snapshots keep their bytes as the ladder deepens', () => {
    const arena = new GSplatsLadderArena();
    arena.appendThrough(parts.slice(0, 2), false);
    const snap2 = arena.snapshot();
    const saved = {
      positions: snap2.positions.slice(),
      amplitudes: snap2.amplitudes.slice(),
      choleskyFactors: snap2.choleskyFactors.slice(),
      colors: snap2.colors ? snap2.colors.slice() : null,
    };
    for (let k = 3; k <= parts.length; k++) {
      arena.appendThrough(parts.slice(0, k), k === parts.length);
      const snapK = arena.snapshot();
      // Earlier snapshot's bytes are untouched...
      expectSameArray(snap2.positions, saved.positions, 'snap2 positions');
      expectSameArray(snap2.amplitudes, saved.amplitudes, 'snap2 amplitudes');
      expectSameArray(snap2.colors, saved.colors, 'snap2 colors');
      // ...and remain the newest snapshot's prefix (same values, same positions).
      expectSameArray(
        snapK.positions.subarray(0, saved.positions.length),
        saved.positions,
        'prefix positions'
      );
      expectSameArray(
        snapK.amplitudes.subarray(0, saved.amplitudes.length),
        saved.amplitudes,
        'prefix amplitudes'
      );
    }
  });

  it('(c) a fresh arena (new generation) never disturbs an old arena snapshot', () => {
    const gen1 = new GSplatsLadderArena();
    gen1.appendThrough(parts.slice(0, 3), false);
    const snap = gen1.snapshot();
    const savedPositions = snap.positions.slice();
    const gen2 = new GSplatsLadderArena();
    gen2.appendThrough([makeGSplatsLod(500, 99), makeGSplatsLod(300, 98)], true);
    expectSameArray(snap.positions, savedPositions, 'gen1 positions after gen2 fill');
  });

  it('(d) appending level k copies O(N_k) entries, not O(N_total)', () => {
    const arena = new GSplatsLadderArena();
    arena.appendThrough(parts.slice(0, 2), false);
    let prevAppended = arena.debugStats().appendedEntries;
    for (let k = 3; k <= parts.length; k++) {
      arena.appendThrough(parts.slice(0, k), k === parts.length);
      const stats = arena.debugStats();
      const n = parts[k - 1].splatCount;
      // Entries appended for level k alone: positions(3) + amplitudes(1) +
      // cholesky(6) + colors(3, real or white fill) per splat.
      expect(stats.appendedEntries - prevAppended).toBe(n * (3 + 1 + 6 + 3));
      prevAppended = stats.appendedEntries;
    }
    // Growth-realloc copies stay amortized: bounded by a small multiple of
    // the final size (1.5× growth ⇒ ≤ 3× total; the reference rebuild would
    // have copied ~k/2 × total across the ladder).
    const totalEntries = arena.debugStats().appendedEntries;
    expect(arena.debugStats().reallocCopiedEntries).toBeLessThanOrEqual(3 * totalEntries);
  });

  it('(e) the ladder-complete trim leaves zero slack (steady-state memory = exact concat)', () => {
    const arena = new GSplatsLadderArena();
    for (let k = 2; k <= parts.length; k++) {
      arena.appendThrough(parts.slice(0, k), k === parts.length);
    }
    const snap = arena.snapshot();
    expect(snap.positions.buffer.byteLength).toBe(snap.positions.byteLength);
    expect(snap.amplitudes.buffer.byteLength).toBe(snap.amplitudes.byteLength);
    expect(snap.choleskyFactors.buffer.byteLength).toBe(snap.choleskyFactors.byteLength);
    expect(snap.colors!.buffer.byteLength).toBe(snap.colors!.byteLength);
  });

  it('rejects mixed color dtypes / layouts / ndim with the reference errors, leaving the arena usable', () => {
    const arena = new GSplatsLadderArena();
    arena.appendThrough(parts.slice(0, 2), false);
    const f32Colors = makeGSplatsLod(10, 7);
    f32Colors.colors = new Float32Array(30);
    expect(() => arena.appendThrough([...parts.slice(0, 2), f32Colors], false)).toThrowError(
      /mixed color dtypes/
    );
    const wrongNdim = makeGSplatsLod(10, 7);
    (wrongNdim as { ndim: number }).ndim = 4;
    expect(() => arena.appendThrough([...parts.slice(0, 2), wrongNdim], false)).toThrowError(
      /mixed dimensionality/
    );
    // The failed batch wrote nothing: the arena still matches the reference.
    expectGSplatsEqual(arena.snapshot(), concatenateGSplatsData(parts.slice(0, 2)));
  });

  it('fails loudly on a SHRUNK ladder instead of snapshotting stale extra content', () => {
    // An arena is append-only within a reset generation (the loader only ever
    // pushes onto loadedLODs); a shorter list means a caller bug.
    const arena = new GSplatsLadderArena();
    arena.appendThrough(parts.slice(0, 3), false);
    expect(() => arena.appendThrough(parts.slice(0, 2), false)).toThrowError(/ladder shrank/);
    // Re-appending the SAME length is a legitimate no-op.
    expect(() => arena.appendThrough(parts.slice(0, 3), false)).not.toThrow();
    expectGSplatsEqual(arena.snapshot(), concatenateGSplatsData(parts.slice(0, 3)));
  });
});

// ---------------------------------------------------------------------------
// Points
// ---------------------------------------------------------------------------

function makePointsLod(
  pointCount: number,
  seed: number,
  opts: { colors?: boolean; radii?: boolean; sharpness?: boolean; scalars?: boolean } = {}
): LoadedPointsData {
  const { colors = true, radii = true, sharpness = true, scalars = true } = opts;
  const positions = new Float32Array(pointCount * 3);
  fillPattern(positions, seed);
  const bounds = new THREE.Box3(
    new THREE.Vector3(-seed, -seed, -seed),
    new THREE.Vector3(seed, seed, seed)
  );
  const result: LoadedPointsData = {
    positions,
    pointCount,
    ndim: 3,
    metadata: {
      totalPoints: 1000,
      loadedPoints: pointCount,
      bounds,
      usedSpatialIndex: true,
    },
  };
  if (colors) {
    const c = new Uint8Array(pointCount * 3);
    fillPatternInt(c, seed + 1);
    result.colors = c;
  }
  if (radii) {
    const r = new Float32Array(pointCount);
    fillPattern(r, seed + 2);
    result.radii = r;
  }
  if (sharpness) {
    const s = new Float32Array(pointCount);
    fillPattern(s, seed + 3);
    result.sharpness = s;
  }
  if (scalars) {
    const s = new Float32Array(pointCount);
    fillPattern(s, seed + 4);
    result.scalars = s;
  }
  return result;
}

function expectPointsEqual(actual: LoadedPointsData, expected: LoadedPointsData): void {
  expectSameArray(actual.positions, expected.positions, 'positions');
  expectSameArray(actual.colors ?? null, expected.colors ?? null, 'colors');
  expect(actual.colorComponents ?? null).toBe(expected.colorComponents ?? null);
  expectSameArray(actual.radii ?? null, expected.radii ?? null, 'radii');
  expectSameArray(actual.sharpness ?? null, expected.sharpness ?? null, 'sharpness');
  expectSameArray(actual.scalars ?? null, expected.scalars ?? null, 'scalars');
  expect(actual.pointCount).toBe(expected.pointCount);
  expect(actual.ndim).toBe(expected.ndim);
  expect(actual.metadata.totalPoints).toBe(expected.metadata.totalPoints);
  expect(actual.metadata.loadedPoints).toBe(expected.metadata.loadedPoints);
  expect(actual.metadata.usedSpatialIndex).toBe(expected.metadata.usedSpatialIndex);
  expect(actual.metadata.bounds.equals(expected.metadata.bounds)).toBe(true);
}

describe('PointsLadderArena — reference equivalence (6-level ladder)', () => {
  // Level 3 lacks radii (all-or-nothing drop from k=4 on) and level 4 lacks
  // scalars — exercising the drop transitions mid-ladder.
  const parts = [
    makePointsLod(80, 1),
    makePointsLod(50, 2),
    makePointsLod(120, 3),
    makePointsLod(30, 4, { radii: false }),
    makePointsLod(60, 5, { scalars: false }),
    makePointsLod(40, 6),
  ];

  it('(a) every incremental snapshot is byte-identical to the reference rebuild', () => {
    const arena = new PointsLadderArena();
    for (let k = 2; k <= parts.length; k++) {
      arena.appendThrough(parts.slice(0, k), k === parts.length);
      expectPointsEqual(arena.snapshot(), concatenatePointsData(parts.slice(0, k)));
    }
  });

  it('(b) prefix stability across appends and drops', () => {
    const arena = new PointsLadderArena();
    arena.appendThrough(parts.slice(0, 3), false);
    const snap3 = arena.snapshot();
    const savedPositions = snap3.positions.slice();
    const savedColors = snap3.colors!.slice();
    arena.appendThrough(parts.slice(0, 6), true);
    const snap6 = arena.snapshot();
    expectSameArray(snap3.positions, savedPositions, 'snap3 positions');
    expectSameArray(snap3.colors, savedColors, 'snap3 colors');
    expectSameArray(
      snap6.positions.subarray(0, savedPositions.length),
      savedPositions,
      'prefix positions'
    );
  });

  it('(c) fresh-arena isolation', () => {
    const gen1 = new PointsLadderArena();
    gen1.appendThrough(parts.slice(0, 2), false);
    const snap = gen1.snapshot();
    const saved = snap.positions.slice();
    const gen2 = new PointsLadderArena();
    gen2.appendThrough([makePointsLod(400, 50), makePointsLod(200, 51)], true);
    expectSameArray(snap.positions, saved, 'gen1 positions after gen2 fill');
  });

  it('(d) appending level k copies O(N_k) positions entries', () => {
    const arena = new PointsLadderArena();
    arena.appendThrough(parts.slice(0, 2), false);
    let prev = arena.debugStats().appendedEntries;
    for (let k = 3; k <= parts.length; k++) {
      arena.appendThrough(parts.slice(0, k), k === parts.length);
      const delta = arena.debugStats().appendedEntries - prev;
      const n = parts[k - 1].pointCount;
      // At most positions(3) + colors(3) + radii(1) + sharpness(1) +
      // scalars(1) per point — never O(N_total). (The delta can dip below
      // the per-level floor — even negative — at a drop transition, because
      // a dropped field's counter leaves the aggregate; the upper bound is
      // the O(N_k) claim.)
      expect(delta).toBeLessThanOrEqual(n * 9);
      prev = arena.debugStats().appendedEntries;
    }
  });

  it('(e) trim-on-complete leaves zero slack', () => {
    const arena = new PointsLadderArena();
    for (let k = 2; k <= parts.length; k++) {
      arena.appendThrough(parts.slice(0, k), k === parts.length);
    }
    const snap = arena.snapshot();
    expect(snap.positions.buffer.byteLength).toBe(snap.positions.byteLength);
    expect(snap.colors!.buffer.byteLength).toBe(snap.colors!.byteLength);
  });

  it('rejects a color-layout flip with the reference error, even for a dropped field', () => {
    // Level 1 lacks colors entirely → merged colors drop; a later RGBA
    // level must STILL fail the layout check (reference parity).
    const noColors = makePointsLod(10, 7, { colors: false });
    const rgba = makePointsLod(10, 8);
    rgba.colors = new Uint8Array(10 * 4);
    rgba.colorComponents = 4;
    const arena = new PointsLadderArena();
    arena.appendThrough([makePointsLod(20, 9), noColors], false);
    expect(() => arena.appendThrough([makePointsLod(20, 9), noColors, rgba], false)).toThrowError(
      /mixed color layouts/
    );
  });
});

// ---------------------------------------------------------------------------
// Lines
// ---------------------------------------------------------------------------

function makeLinesLod(
  segmentCount: number,
  seed: number,
  opts: { colors?: boolean; sharpness?: boolean; scalars?: boolean } = {}
): LoadedLinesData {
  const { colors = true, sharpness = true, scalars = true } = opts;
  const vertexCount = segmentCount * 2;
  const ndim = 4; // lines concat strides positions by the ORIGINAL ndim
  const positions = new Float32Array(vertexCount * ndim);
  fillPattern(positions, seed);
  const widths = new Float32Array(vertexCount);
  fillPattern(widths, seed + 1);
  // Local segment indices (pairs into this part's own vertex range).
  const segments = new Uint32Array(segmentCount * 2);
  for (let i = 0; i < segments.length; i++) segments[i] = i % vertexCount;
  let colorArr: Uint8Array | null = null;
  if (colors) {
    colorArr = new Uint8Array(vertexCount * 3);
    fillPatternInt(colorArr, seed + 2);
  }
  let sharpnessArr: Float32Array | null = null;
  if (sharpness) {
    sharpnessArr = new Float32Array(vertexCount);
    fillPattern(sharpnessArr, seed + 3);
  }
  const result: LoadedLinesData = {
    positions,
    segments,
    widths,
    colors: colorArr,
    ...(colorArr ? { colorComponents: 3 as const } : {}),
    sharpness: sharpnessArr,
    segmentCount,
    vertexCount,
    ndim,
  };
  if (scalars) {
    const s = new Float32Array(vertexCount);
    fillPattern(s, seed + 4);
    result.scalars = s;
  }
  return result;
}

function expectLinesEqual(actual: LoadedLinesData, expected: LoadedLinesData): void {
  expectSameArray(actual.positions, expected.positions, 'positions');
  expectSameArray(actual.segments, expected.segments, 'segments');
  expectSameArray(actual.widths, expected.widths, 'widths');
  expectSameArray(actual.colors, expected.colors, 'colors');
  expect(actual.colorComponents ?? null).toBe(expected.colorComponents ?? null);
  expectSameArray(actual.sharpness, expected.sharpness, 'sharpness');
  expectSameArray(actual.scalars ?? null, expected.scalars ?? null, 'scalars');
  expect(actual.segmentCount).toBe(expected.segmentCount);
  expect(actual.vertexCount).toBe(expected.vertexCount);
  expect(actual.ndim).toBe(expected.ndim);
}

describe('LinesLadderArena — reference equivalence (6-level ladder)', () => {
  // Level 1 is colorless (white fill), level 0 lacks sharpness (late
  // 0.5-backfill when level 1 carries it), level 4 lacks scalars (drop).
  const parts = [
    makeLinesLod(40, 1, { sharpness: false }),
    makeLinesLod(25, 2, { colors: false }),
    makeLinesLod(60, 3),
    makeLinesLod(15, 4),
    makeLinesLod(30, 5, { scalars: false }),
    makeLinesLod(20, 6),
  ];

  it('(a) every incremental snapshot is byte-identical to the reference rebuild (incl. segment offsets)', () => {
    const arena = new LinesLadderArena();
    for (let k = 2; k <= parts.length; k++) {
      arena.appendThrough(parts.slice(0, k), k === parts.length);
      expectLinesEqual(arena.snapshot(), concatenateLinesData(parts.slice(0, k)));
    }
  });

  it('(b) prefix stability across appends (segments included)', () => {
    const arena = new LinesLadderArena();
    arena.appendThrough(parts.slice(0, 2), false);
    const snap2 = arena.snapshot();
    const savedSegments = snap2.segments.slice();
    const savedPositions = snap2.positions.slice();
    arena.appendThrough(parts.slice(0, 6), true);
    const snap6 = arena.snapshot();
    expectSameArray(snap2.segments, savedSegments, 'snap2 segments');
    expectSameArray(snap2.positions, savedPositions, 'snap2 positions');
    expectSameArray(
      snap6.segments.subarray(0, savedSegments.length),
      savedSegments,
      'prefix segments'
    );
  });

  it('(c) fresh-arena isolation', () => {
    const gen1 = new LinesLadderArena();
    gen1.appendThrough(parts.slice(0, 2), false);
    const snap = gen1.snapshot();
    const saved = snap.positions.slice();
    const gen2 = new LinesLadderArena();
    gen2.appendThrough([makeLinesLod(200, 50), makeLinesLod(100, 51)], true);
    expectSameArray(snap.positions, saved, 'gen1 positions after gen2 fill');
  });

  it('(d) appending level k copies O(N_k) entries', () => {
    const arena = new LinesLadderArena();
    arena.appendThrough(parts.slice(0, 2), false);
    let prev = arena.debugStats().appendedEntries;
    for (let k = 3; k <= parts.length; k++) {
      arena.appendThrough(parts.slice(0, k), k === parts.length);
      const delta = arena.debugStats().appendedEntries - prev;
      const v = parts[k - 1].vertexCount;
      const s = parts[k - 1].segmentCount;
      // positions(ndim=4) + widths(1) + colors(3) + sharpness(1) +
      // scalars(≤1) per vertex, + 2 per segment.
      expect(delta).toBeLessThanOrEqual(v * 10 + s * 2);
      expect(delta).toBeGreaterThanOrEqual(v * 4 + s * 2);
      prev = arena.debugStats().appendedEntries;
    }
  });

  it('(e) trim-on-complete leaves zero slack', () => {
    const arena = new LinesLadderArena();
    for (let k = 2; k <= parts.length; k++) {
      arena.appendThrough(parts.slice(0, k), k === parts.length);
    }
    const snap = arena.snapshot();
    expect(snap.positions.buffer.byteLength).toBe(snap.positions.byteLength);
    expect(snap.segments.buffer.byteLength).toBe(snap.segments.byteLength);
  });

  it('rejects mixed ndim with the reference error', () => {
    const arena = new LinesLadderArena();
    arena.appendThrough(parts.slice(0, 2), false);
    const wrong = makeLinesLod(5, 9);
    (wrong as { ndim: number }).ndim = 3;
    expect(() => arena.appendThrough([...parts.slice(0, 2), wrong], false)).toThrowError(
      /mixed dimensionality/
    );
  });
});
