/**
 * Lines buffer-pool adapter.
 *
 * Owns the lines-specific pool state (per-capacity buckets) and
 * implements acquire/release/update for Line geometries. Shared state
 * (activeBuffers, stats, commit counter) is read from the GPUBufferPool
 * reference passed at construction.
 *
 * Per-segment data lives in the RGBA32F line texture attached at
 * creation (6 texels/segment — see `../line-geometry.ts` for the layout
 * and `../element-texture-layout.ts` for the addressing math); the only
 * per-instance attribute is `aSortedIndex`. The layout is FIXED
 * regardless of which optional fields the dataset carries, so — unlike
 * the interleaved era's colormap-scalar spec bucketing — ANY pooled
 * lines geometry fits ANY lines node (mirroring the points/gsplats
 * adapters): reuse keys on capacity alone.
 *
 * The acquire/release/dispose lifecycle is shared with the points and
 * gsplats adapters (`texture-backed-adapter.ts`); this file owns the
 * Lines texel layout hook and the update path.
 */

import type * as THREE from 'three';
import {
  attachLineStorage,
  computeLineBounds,
  getLineTexture,
  stampLinePresenceFlags,
  writeLineTexels,
} from '../line-geometry';
import { clampLineCapacity } from '../element-texture-layout';
import type { ProcessedLinesData } from '../../types/lines';
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
 * Lines: 6 texels/segment, so a per-node bound of width × maxTextureSize / 6
 * (2.79M segments on a 4096-class device).
 */
const LINES_LAYOUT: TextureBackedLayout = {
  type: 'lines',
  clampCapacity: clampLineCapacity,
  attachStorage: attachLineStorage,
};

export class LinesBufferAdapter extends TextureBackedAdapter {
  constructor(host: PoolAdapterHost) {
    super(host, LINES_LAYOUT);
  }

  /** @internal — pool-bucketed reusable line geometries. */
  get lineBuffers(): FreeBucketMap {
    return this.buffers;
  }

  updateGeometry(
    geometry: THREE.InstancedBufferGeometry,
    data: ProcessedLinesData,
    count: number,
    options?: InstancedOrderingOptions
  ): void {
    const instanced = geometry;
    const texture = getLineTexture(instanced);
    if (!texture) {
      throw new Error(
        'LinesBufferAdapter.updateGeometry: geometry has no line texture — ' +
          'was it acquired from the pool?'
      );
    }
    // Append fast path (Phase 4 Stage 2): the commit layer sets
    // `fromInstance` to the prefix count already on the GPU when this
    // commit only extends it, so the fused writer + ranged upload touch
    // just the `[fromInstance, count)` suffix (see writeLineTexels).
    // 0 means a full write.
    const fromInstance = options?.fromInstance ?? 0;

    // One fused pass over the staged arrays into the texel layout
    // (replaces the 11–13 per-attribute strided writes; the writer's
    // fail-loud guard runs before ANY store, retiring the interleaved
    // era's separate pre-flight torn-write sweep), then the commit
    // ordering. The writer clamps to the texture capacity; mirror that
    // clamp in instanceCount so a bound-clamped node never draws
    // instances whose texels were not written.
    count = writeLineTexels(texture, data, count, { fromSegment: fromInstance });
    const drawCount = writeInstancedCommitOrdering(instanced, count, options);

    prepareInstancedQuadForDraw(instanced, drawCount);

    // Scalar presence stamp (drives `supportsScalarColormap`) — refreshed
    // on EVERY update; pool geometries are reused across tenants.
    stampLinePresenceFlags(instanced, data);

    // CRITICAL: Recompute bounding box after position updates, expanded
    // by max width so the rendered footprint is covered by frustum
    // culling (shared helper — see line-geometry.ts).
    computeLineBounds(instanced, data, count);
  }
}
