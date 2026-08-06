/**
 * Synthetic large-scene generators for the perf bench.
 *
 * Builds an `InstancedLinesMeshConfig`, an `InstancedGSplatsMeshConfig`,
 * or a points payload at configurable instance counts; the debug
 * injector (`core/app/debug/debug-interface.ts`) pushes the resulting
 * mesh into the live scene via the existing `node-factory` +
 * `material-manager` pipeline. Used exclusively by the perf-bench
 * harness to exercise the bandwidth-bound regime (millions of
 * elements) without needing a real on-disk zarr file.
 *
 * Not loaded in production builds — the only consumers are the
 * `__luxarDebug.injectSyntheticScene(...)` debug API and the
 * perf-bench Playwright specs that call it.
 *
 * Generators are PURE (typed arrays in/out, seeded PRNG, no THREE
 * runtime import) so they unit-test headlessly. The lines generator's
 * positions/colors/lengths are kept byte-for-byte as originally shipped
 * (pinned by the `line-perf-bench.spec.ts` 10 M-segment scenario
 * contract; the joint-code arrays are derived from the same walk
 * without consuming PRNG draws, so the pinned arrays are untouched); the
 * points/gsplats generators share the seeded-RNG + gaussian-cluster
 * sampling scaffolding below.
 *
 * @module scene/synthetic-scene
 */

import type * as THREE from 'three';

import type { InstancedLinesMeshConfig } from '../rendering/line-geometry';
import type { InstancedGSplatsMeshConfig } from '../rendering/gsplat-geometry';
import { JOINT_FREE_END } from '../wasm/typescript/lines-clipping';

export type SyntheticSceneType = 'lines' | 'points' | 'gsplats';

export interface SyntheticSceneSpec {
  type: SyntheticSceneType;
  /** Number of elements (segments / points / splats) to generate. */
  count: number;
  /**
   * Half-extent of the generation volume. Lines: bounds of the random
   * walk. Points/gsplats: cluster centers are drawn uniformly from
   * `[-bounds, bounds]^3`. Larger bounds → more on-screen spread.
   * Defaults to [-100, 100].
   */
  bounds?: number;
  /**
   * Deterministic seed so the same `(type, count, seed, clusters)`
   * produces the same scene byte-for-byte. Defaults to 1.
   */
  seed?: number;
  /**
   * Number of gaussian blobs the points/gsplats samplers draw from
   * (ignored by 'lines', which is a random walk). Defaults to
   * {@link DEFAULT_CLUSTERS}.
   */
  clusters?: number;
  /**
   * Blending mode for the injected node — consumed by the debug
   * injector (not the generators). Defaults: 'additive' for lines
   * (the historical bench contract), 'normal' for points/gsplats so
   * the depth-sort subsystem engages.
   */
  blending?: string;
}

/** Default gaussian-blob count for the clustered samplers. */
export const DEFAULT_CLUSTERS = 256;

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
 * during a `?renderer=…` bench run is higher because the line texture
 * (`writeLineTexels`) allocates another `count × 24` Float32Array
 * backing store and Three's GPU upload double-buffers in driver
 * memory until the first frame submits.
 *
 *   source arrays:   count × 17 Float32 (positions×2=6, colors×2=6,
 *                       widths×2=2, sharpness×2=2, length×1=1) × 4 B
 *                    + count × 2 Float32 (joint codes) × 4 B
 *                  = count × 76 B
 *   line texture:    count × 24 floats × 4 B = count × 96 B
 *                    (6 texels/segment RGBA32F — see line-geometry.ts)
 *   ≈ 2.3× the source-array figure as a working JS heap estimate.
 *
 * For 10 M segments: ~760 MB of source arrays → ~1.75 GB peak JS
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
  const startJointCode = new Float32Array(count);
  const endJointCode = new Float32Array(count);

  // Random-walk anchor for segment continuity — visually more
  // interesting than disconnected random pairs and matches what real
  // streamline / trajectory datasets look like.
  let px = (rand() * 2 - 1) * bounds;
  let py = (rand() * 2 - 1) * bounds;
  let pz = (rand() * 2 - 1) * bounds;

  const stepScale = bounds * 0.01;

  // Previous segment's length, to detect a degenerate neighbour when emitting
  // joint codes (see the block at the bottom of the loop).
  let prevLen = 0;

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
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    segmentLengths[i] = len;

    // Faithful joint codes from the walk's own topology (this path never runs
    // through `compute_joint_codes`, so emit here exactly what the kernel would
    // emit for connected geometry). The walk is a chain, so segment i-1's END
    // meets segment i's START: i-1 names segment i at its start (+(i + 1)) and
    // i names segment i-1 at its end (−((i − 1) + 3)). Note this is purely
    // topological — unlike the angle-derived scalar it replaces, it does not
    // depend on the segment directions at all. Chain breaks (the i % 64 reset
    // above) and free ends stay JOINT_FREE_END, keeping the soft cap;
    // zero-length segments do too, matching the kernel's degenerate fallback.
    startJointCode[i] = JOINT_FREE_END;
    endJointCode[i] = JOINT_FREE_END;
    if (i % 64 !== 0 && prevLen > 0 && len > 0) {
      endJointCode[i - 1] = i + 1;
      startJointCode[i] = -(i - 1 + 3);
    }
    prevLen = len;
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
    startJointCode,
    endJointCode,
    segmentCount: count,
  };
}

/**
 * Gaussian sampler over a seeded uniform PRNG (Box–Muller, spare-value
 * cached). Deterministic: the same `rand` stream yields the same
 * normal stream.
 */
function makeGaussian(rand: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    // rand() ∈ [0, 1); shift u1 away from 0 so log() stays finite.
    const u1 = 1 - rand();
    const u2 = rand();
    const mag = Math.sqrt(-2 * Math.log(u1));
    spare = mag * Math.sin(2 * Math.PI * u2);
    return mag * Math.cos(2 * Math.PI * u2);
  };
}

/**
 * Sample `count` 3D positions from `clusters` gaussian blobs inside a
 * `[-bounds, bounds]^3` volume — the shared spatial scaffold for the
 * points and gsplats generators (real microscopy / astronomy point
 * sets are clumpy, and clustered depth structure is what makes
 * depth-sorted blending measurably order-dependent).
 *
 * Cluster centers are uniform in the volume; per-cluster σ is
 * 2–8 % of `bounds`. Deterministic for a given `(rand-stream, count,
 * clusters, bounds)`.
 *
 * Also returns the empirical min/max corner of the sampled positions
 * (gaussians have unbounded tails, so consumers needing a bounding box
 * must use the measured one, not `±bounds`).
 */
export function sampleClusteredPositions(
  rand: () => number,
  count: number,
  clusters: number,
  bounds: number
): { positions: Float32Array; min: [number, number, number]; max: [number, number, number] } {
  const gauss = makeGaussian(rand);
  const k = Math.max(1, Math.floor(clusters));

  // Cluster table first (fixed PRNG-stream prefix, so the same seed
  // gives the same blobs regardless of count).
  const centers = new Float32Array(k * 3);
  const sigmas = new Float32Array(k);
  for (let c = 0; c < k; c++) {
    centers[c * 3] = (rand() * 2 - 1) * bounds;
    centers[c * 3 + 1] = (rand() * 2 - 1) * bounds;
    centers[c * 3 + 2] = (rand() * 2 - 1) * bounds;
    sigmas[c] = bounds * (0.02 + 0.06 * rand());
  }

  const positions = new Float32Array(count * 3);
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < count; i++) {
    const c = Math.min(k - 1, Math.floor(rand() * k));
    const s = sigmas[c];
    for (let d = 0; d < 3; d++) {
      const v = centers[c * 3 + d] + s * gauss();
      positions[i * 3 + d] = v;
      // Read back the Float32-rounded value so min/max bound the array
      // EXACTLY (v is float64 here).
      const stored = positions[i * 3 + d];
      if (stored < min[d]) min[d] = stored;
      if (stored > max[d]) max[d] = stored;
    }
  }
  if (count === 0) {
    min[0] = min[1] = min[2] = 0;
    max[0] = max[1] = max[2] = 0;
  }
  return { positions, min, max };
}

/**
 * Payload for a synthetic points node — plain typed arrays (Float32
 * throughout, so the production widen/normalize path is an identity)
 * plus the measured bounds the injector turns into the geometry's
 * `THREE.Box3`.
 */
export interface SyntheticPointsConfig {
  /** Positions (pointCount * 3). */
  positions: Float32Array;
  /** RGB colors in [0, 1) (pointCount * 3). */
  colors: Float32Array;
  /** Per-point world-unit radii (pointCount). */
  radii: Float32Array;
  /** Per-point sharpness knob in [0, 1] (pointCount). */
  sharpness: Float32Array;
  pointCount: number;
  /** Max world-unit radius — the geometry's footprint expansion. */
  maxRadius: number;
  /** Empirical min corner of `positions`. */
  boundsMin: [number, number, number];
  /** Empirical max corner of `positions`. */
  boundsMax: [number, number, number];
}

/**
 * Generate a clustered points scene: positions from
 * {@link sampleClusteredPositions}, vivid random colors, per-point
 * radii varied over [0.2 %, 1 %] of `bounds`, and sharpness at the
 * dataset default 0.5 (Gaussian midpoint, beta = 2 — matches the
 * lines generator).
 *
 * Memory: `count × 8` Float32 source floats (32 B/point) + the
 * RGBA32F point texture at 3 texels/point (48 B/point) during
 * injection.
 */
export function generateSyntheticPoints(spec: SyntheticSceneSpec): SyntheticPointsConfig {
  const count = spec.count;
  const bounds = spec.bounds ?? 100;
  const clusters = spec.clusters ?? DEFAULT_CLUSTERS;
  const rand = mulberry32(spec.seed ?? 1);

  const { positions, min, max } = sampleClusteredPositions(rand, count, clusters, bounds);

  const colors = new Float32Array(count * 3);
  const radii = new Float32Array(count);
  const sharpness = new Float32Array(count);
  let maxRadius = 0;
  for (let i = 0; i < count; i++) {
    colors[i * 3] = rand();
    colors[i * 3 + 1] = rand();
    colors[i * 3 + 2] = rand();
    radii[i] = bounds * (0.002 + 0.008 * rand());
    if (radii[i] > maxRadius) maxRadius = radii[i];
    sharpness[i] = 0.5;
  }

  return {
    positions,
    colors,
    radii,
    sharpness,
    pointCount: count,
    // 0.5 mirrors createPointsGeometry's no-radii footprint default.
    maxRadius: count > 0 ? maxRadius : 0.5,
    boundsMin: min,
    boundsMax: max,
  };
}

/**
 * Generate a clustered gsplats scene as an `InstancedGSplatsMeshConfig`.
 *
 * Each splat gets a lower-triangular Cholesky factor L with strictly
 * positive diagonal — by construction a valid factor of the SPD
 * covariance Σ = L·Lᵀ:
 * - characteristic size log-uniform over one decade
 *   ([0.2 %, 2 %] of `bounds`) → scale varies;
 * - per-axis diagonal jitter ×[0.4, 1.6] → anisotropy varies;
 * - off-diagonals ±0.8 × the row's diagonal → orientation/correlation
 *   varies (Σ gains substantial off-diagonal terms).
 * Scale AND orientation therefore differ across splats, so
 * depth-sorted 'normal' blending is measurably order-dependent.
 *
 * Amplitudes are positive and varied ([0.2, 1.0)); colors vivid random
 * RGB.
 *
 * Packing matches `SplatTexelSource.choleskyFactors`: 6-stride
 * row-major [L00, L10, L11, L20, L21, L22] per splat.
 */
export function generateSyntheticGSplats(spec: SyntheticSceneSpec): InstancedGSplatsMeshConfig {
  const count = spec.count;
  const bounds = spec.bounds ?? 100;
  const clusters = spec.clusters ?? DEFAULT_CLUSTERS;
  const rand = mulberry32(spec.seed ?? 1);

  const { positions: centers } = sampleClusteredPositions(rand, count, clusters, bounds);

  const choleskyFactors = new Float32Array(count * 6);
  const amplitudes = new Float32Array(count);
  const colors = new Float32Array(count * 3);

  const sizeFloor = bounds * 0.002;
  for (let i = 0; i < count; i++) {
    // Characteristic size: log-uniform over [sizeFloor, 10·sizeFloor].
    const s = sizeFloor * Math.pow(10, rand());
    // Per-axis diagonals: positive, anisotropic (×[0.4, 1.6]).
    const d0 = s * (0.4 + 1.2 * rand());
    const d1 = s * (0.4 + 1.2 * rand());
    const d2 = s * (0.4 + 1.2 * rand());
    // Off-diagonals relative to the row diagonal — any lower-triangular
    // L with positive diagonal is a valid Cholesky factor, so no
    // further constraint is needed for positive-definiteness.
    const L10 = (rand() * 2 - 1) * 0.8 * d1;
    const L20 = (rand() * 2 - 1) * 0.8 * d2;
    const L21 = (rand() * 2 - 1) * 0.8 * d2;

    const c6 = i * 6;
    choleskyFactors[c6] = d0; // L00
    choleskyFactors[c6 + 1] = L10; // L10
    choleskyFactors[c6 + 2] = d1; // L11
    choleskyFactors[c6 + 3] = L20; // L20
    choleskyFactors[c6 + 4] = L21; // L21
    choleskyFactors[c6 + 5] = d2; // L22

    amplitudes[i] = 0.2 + 0.8 * rand();
    colors[i * 3] = rand();
    colors[i * 3 + 1] = rand();
    colors[i * 3 + 2] = rand();
  }

  return {
    centers,
    choleskyFactors,
    amplitudes,
    colors,
    splatCount: count,
  };
}

/**
 * Result of injecting a synthetic scene. Returned to the caller so
 * the perf bench can read back the actual element count and pass it
 * to its result JSON. `elementCount` is uniform across types (and is
 * the CLAMPED drawn count — see the per-node texture capacity clamps);
 * the per-type aliases (`segmentCount` / `pointCount` / `splatCount`)
 * carry the REQUESTED count and preserve the original lines shape.
 */
export type SyntheticInjectionResult =
  | { type: 'lines'; segmentCount: number; elementCount: number; mesh: THREE.Mesh }
  | { type: 'points'; pointCount: number; elementCount: number; mesh: THREE.Mesh }
  | { type: 'gsplats'; splatCount: number; elementCount: number; mesh: THREE.Mesh };
