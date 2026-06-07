/**
 * Synthetic large-scene generators for the perf bench.
 *
 * Builds an `InstancedLinesMeshConfig` (or, in future commits, a
 * points / gsplats equivalent) at configurable instance counts and
 * pushes the resulting mesh into the live scene via the existing
 * `node-factory` + `material-manager` pipeline. Used exclusively by
 * the perf-bench harness to exercise the bandwidth-bound regime
 * (millions of segments) without needing a real on-disk zarr file.
 *
 * Not loaded in production builds — the only consumers are the
 * `__luxarDebug.injectSyntheticScene(...)` debug API and the
 * `line-perf-bench.spec.ts` Playwright spec that calls it.
 *
 * @module scene/synthetic-scene
 */

import type * as THREE from 'three';

import type { InstancedLinesMeshConfig } from '../rendering/line-geometry';

export type SyntheticSceneType = 'lines';

export interface SyntheticSceneSpec {
  type: SyntheticSceneType;
  /** Number of line segments to generate. */
  count: number;
  /**
   * Bounds for the random walk that generates segments. Larger
   * bounds → more on-screen spread. Defaults to [-100, 100].
   */
  bounds?: number;
  /**
   * Deterministic seed so the same `(type, count, seed)` produces
   * the same scene byte-for-byte. Defaults to 1.
   */
  seed?: number;
}

/**
 * Mulberry32 PRNG — small, fast, deterministic. Sufficient for
 * filler synthetic data; no cryptographic claims.
 */
function mulberry32(seed: number): () => number {
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
 * Generate a line-segments scene of the requested size. Segments form
 * a random walk through a cube of side `2 * bounds`, with per-segment
 * widths and sharpness at the dataset defaults (1.0 and 0.5; the
 * normalized sharpness knob's Gaussian midpoint, beta = 2).
 *
 * Memory cost — these are *source* arrays only; the actual peak
 * during a `?renderer=…` bench run is higher because
 * `packInterleavedAttributes` allocates another `count * stride`
 * Float32Array and Three's GPU upload double-buffers in driver
 * memory until the first frame submits.
 *
 *   source arrays:   count × 17 Float32 (positions×2=6, colors×2=6,
 *                       widths×2=2, sharpness×2=2, length×1=1) × 4 B
 *                    + count × 2 Uint8  (clipped flags) × 1 B
 *                  = count × 70 B
 *   interleaved buf: count × stride × 4 B, stride matches the per-
 *                    instance set above (widened to Float32) plus
 *                    any alignment padding
 *   ≈ 2× the source-array figure as a working JS heap estimate.
 *
 * For 10 M segments: ~700 MB of source arrays → ~1.4 GB peak JS
 * heap during construction + packing. The bench machine needs the
 * RAM headroom; on developer laptops, prefer smaller counts (the
 * synthetic scenarios list in `line-perf-bench.spec.ts` is a good
 * starting point to scale down).
 */
export function generateSyntheticLines(spec: SyntheticSceneSpec): InstancedLinesMeshConfig {
  const count = spec.count;
  const bounds = spec.bounds ?? 100;
  const rand = mulberry32(spec.seed ?? 1);

  const startPositions = new Float32Array(count * 3);
  const endPositions = new Float32Array(count * 3);
  const startColors = new Float32Array(count * 3);
  const endColors = new Float32Array(count * 3);
  const startWidths = new Float32Array(count);
  const endWidths = new Float32Array(count);
  const startSharpness = new Float32Array(count);
  const endSharpness = new Float32Array(count);
  const segmentLengths = new Float32Array(count);
  const startClipped = new Uint8Array(count);
  const endClipped = new Uint8Array(count);

  // Random-walk anchor for segment continuity — visually more
  // interesting than disconnected random pairs and matches what real
  // streamline / trajectory datasets look like.
  let px = (rand() * 2 - 1) * bounds;
  let py = (rand() * 2 - 1) * bounds;
  let pz = (rand() * 2 - 1) * bounds;

  const stepScale = bounds * 0.01;

  for (let i = 0; i < count; i++) {
    const i3 = i * 3;

    // Occasional reset to avoid the random walk drifting outside the
    // viewable bounds.
    if (i % 64 === 0) {
      px = (rand() * 2 - 1) * bounds;
      py = (rand() * 2 - 1) * bounds;
      pz = (rand() * 2 - 1) * bounds;
    }

    startPositions[i3] = px;
    startPositions[i3 + 1] = py;
    startPositions[i3 + 2] = pz;

    px += (rand() * 2 - 1) * stepScale;
    py += (rand() * 2 - 1) * stepScale;
    pz += (rand() * 2 - 1) * stepScale;

    endPositions[i3] = px;
    endPositions[i3 + 1] = py;
    endPositions[i3 + 2] = pz;

    // Random vivid colours so colour interpolation is meaningful in
    // the fragment stage.
    startColors[i3] = rand();
    startColors[i3 + 1] = rand();
    startColors[i3 + 2] = rand();
    endColors[i3] = rand();
    endColors[i3 + 1] = rand();
    endColors[i3 + 2] = rand();

    startWidths[i] = 1.0;
    endWidths[i] = 1.0;
    // Default 0.5 — the normalized sharpness knob's Gaussian midpoint
    // (beta = 2^(6·0.5 − 2) = 2), the production-default dataset value.
    startSharpness[i] = 0.5;
    endSharpness[i] = 0.5;

    const dx = endPositions[i3] - startPositions[i3];
    const dy = endPositions[i3 + 1] - startPositions[i3 + 1];
    const dz = endPositions[i3 + 2] - startPositions[i3 + 2];
    segmentLengths[i] = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // Both endpoints unclipped — synthetic data has no slice clipping.
    startClipped[i] = 0;
    endClipped[i] = 0;
  }

  return {
    startPositions,
    endPositions,
    startColors,
    endColors,
    startWidths,
    endWidths,
    startSharpness,
    endSharpness,
    segmentLengths,
    startClipped,
    endClipped,
    segmentCount: count,
  };
}

/**
 * Result of injecting a synthetic scene. Returned to the caller so
 * the perf bench can read back the actual segment count and pass it
 * to its result JSON.
 */
export interface SyntheticInjectionResult {
  type: SyntheticSceneType;
  segmentCount: number;
  /** The Mesh that was added to the scene. */
  mesh: THREE.Mesh;
}
