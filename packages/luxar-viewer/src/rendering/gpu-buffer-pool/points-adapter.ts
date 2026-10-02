/**
 * Points buffer-pool adapter.
 *
 * Owns the points-specific pool state (per-capacity buckets) and
 * implements acquire/release/update for Points geometries. Shared
 * state (activeBuffers, stats, commit counter) is read from the
 * GPUBufferPool reference passed at construction.
 *
 * Per-point data lives in the RGBA32F point texture attached at
 * creation (3 texels/point — see `../point-geometry.ts` for the layout
 * and `../element-texture-layout.ts` for the addressing math); the only
 * per-instance attribute is `aSortedIndex`. The layout is FIXED
 * regardless of which optional fields the dataset carries, so — unlike
 * the interleaved era's dtype/spec bucketing — ANY pooled points
 * geometry fits ANY points node (mirroring the gsplats adapter): reuse
 * keys on capacity alone. Dtype normalization happens at upload time
 * with the same `widenToFloat32` calls (and the same fallback fills)
 * the interleaved path used, so texel values are bit-identical to what
 * the old per-attribute writes produced.
 *
 * The acquire/release/dispose lifecycle is shared with the lines and
 * gsplats adapters (`texture-backed-adapter.ts`); this file owns the
 * Points texel layout hook and the update path.
 */

import * as THREE from 'three';
import { widenToFloat32 } from '../widen-to-float32';
import {
  attachPointStorage,
  getPointTexture,
  pointsNormalizationDivisor,
  stampPointPresenceFlags,
  writePointTexels,
  type PointTexelSource,
} from '../point-geometry';
import { clampPointCapacity } from '../element-texture-layout';
import { DEFAULT_POINT_RADIUS } from '../../config/constants';
import type { LoadedPointsData } from '../../data/data-loader-types';
import type { FreeBucketMap } from './byte-tracked-maps';
import {
  TextureBackedAdapter,
  prepareInstancedQuadForDraw,
  writeInstancedCommitOrdering,
  type InstancedOrderingOptions,
  type PoolAdapterHost,
  type TextureBackedLayout,
} from './texture-backed-adapter';

/**
 * Points: 3 texels/point, so a per-node bound of width × maxTextureSize / 3
 * (5.59M points on a 4096-class device).
 */
const POINTS_LAYOUT: TextureBackedLayout = {
  type: 'points',
  clampCapacity: clampPointCapacity,
  attachStorage: attachPointStorage,
};

export class PointsBufferAdapter extends TextureBackedAdapter {
  constructor(host: PoolAdapterHost) {
    super(host, POINTS_LAYOUT);
  }

  /** @internal — pool-bucketed reusable point geometries. */
  get pointBuffers(): FreeBucketMap {
    return this.buffers;
  }

  updateGeometry(
    geometry: THREE.InstancedBufferGeometry,
    data: LoadedPointsData,
    count: number,
    options?: InstancedOrderingOptions
  ): void {
    const instanced = geometry;
    const texture = getPointTexture(instanced);
    if (!texture) {
      throw new Error(
        'PointsBufferAdapter.updateGeometry: geometry has no point texture — ' +
          'was it acquired from the pool?'
      );
    }
    // Append fast path (Phase 4 Stage 2): the commit layer sets
    // `fromInstance` to the prefix count already on the GPU when this
    // commit only extends it, so the fused writer + ranged upload touch
    // just the `[fromInstance, count)` suffix (see writePointTexels).
    // 0 means a full write.
    const fromInstance = options?.fromInstance ?? 0;

    // Widen each field EXACTLY as the interleaved era did — the same
    // widenToFloat32 calls on the same source slices with the same
    // normalization divisors and the same fallback fills — so the
    // floats written to texels are bit-identical to what the old
    // per-attribute writes uploaded.
    const positionsF32 =
      data.positions instanceof Float32Array
        ? data.positions
        : widenToFloat32(data.positions as ArrayLike<number>);

    // Color layout: 3 (RGB) or 4 (RGBA — alpha = per-point opacity).
    // Strides the staged slice and the writer's per-point reads; the
    // absent-colors fill stays RGB (the writer stamps the 1.0 opaque
    // identity into texel2.y unconditionally either way).
    const colorK: 3 | 4 = data.colors ? (data.colorComponents ?? 3) : 3;
    let colorsF32: Float32Array;
    if (data.colors) {
      colorsF32 = widenToFloat32(
        data.colors.subarray(0, count * colorK) as ArrayLike<number>,
        pointsNormalizationDivisor(data.colors, /*normalized=*/ true)
      );
    } else {
      colorsF32 = new Float32Array(count * 3);
      colorsF32.fill(1.0); // white default
    }

    let radiiF32: Float32Array;
    if (data.radii) {
      radiiF32 = widenToFloat32(
        data.radii.subarray(0, count) as ArrayLike<number>,
        pointsNormalizationDivisor(data.radii, /*normalized=*/ true)
      );
    } else {
      radiiF32 = new Float32Array(count);
      radiiF32.fill(DEFAULT_POINT_RADIUS);
    }

    let sharpnessF32: Float32Array;
    if (data.sharpness) {
      sharpnessF32 = widenToFloat32(
        data.sharpness.subarray(0, count) as ArrayLike<number>,
        pointsNormalizationDivisor(data.sharpness, /*normalized=*/ true)
      );
    } else {
      sharpnessF32 = new Float32Array(count);
      sharpnessF32.fill(0.5); // default sharpness knob -> beta=2 (Gaussian)
    }

    // Scalars: absent ⇒ omitted from the source, and the writer stamps
    // the 0.0 identity into texel2.x unconditionally (the fixed-layout
    // analog of the interleaved era's zero-fill of a bound aScalar).
    const scalarsF32 = data.scalars
      ? widenToFloat32(
          data.scalars.subarray(0, count) as ArrayLike<number>,
          pointsNormalizationDivisor(data.scalars, /*normalized=*/ true)
        )
      : undefined;

    const texelSrc: PointTexelSource = {
      positions: positionsF32,
      colors: colorsF32,
      colorComponents: colorK,
      radii: radiiF32,
      sharpness: sharpnessF32,
      scalars: scalarsF32,
    };
    // One fused pass over the staged arrays into the texel layout
    // (replaces the five per-attribute strided writes), then identity
    // ordering. The writer clamps to the texture capacity; mirror that
    // clamp in instanceCount so a bound-clamped node never draws
    // instances whose texels were not written.
    count = writePointTexels(texture, texelSrc, count, { fromPoint: fromInstance });
    writeInstancedCommitOrdering(instanced, count, options);

    prepareInstancedQuadForDraw(instanced, count);

    // Presence stamps — shared chokepoint with the node factory; see
    // `stampPointPresenceFlags` for what each flag carries and why they
    // refresh on EVERY update (pool tenant flips).
    //
    // Deliberately stamped AFTER the texel write above: writePointTexels
    // throws only at its pre-loop length guard, so a throwing write
    // leaves the texture's PREVIOUS content fully intact — and the
    // un-reached stamps stay consistent with it (old texels + old
    // stamps). Stamping before the write would instead pair NEW stamps
    // with OLD texels on that path (reviewed and kept; the lines
    // adapter's stampLinePresenceFlags follows the same ordering).
    stampPointPresenceFlags(instanced, data, colorK);

    if (data.metadata.bounds) {
      instanced.boundingBox = data.metadata.bounds.clone();
    } else {
      const box = new THREE.Box3();
      const v = new THREE.Vector3();
      const positions = data.positions;
      for (let i = 0; i < count; i++) {
        v.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
        box.expandByPoint(v);
      }
      instanced.boundingBox = box;
    }
    instanced.boundingSphere = new THREE.Sphere();
    instanced.boundingBox.getBoundingSphere(instanced.boundingSphere);
  }
}
