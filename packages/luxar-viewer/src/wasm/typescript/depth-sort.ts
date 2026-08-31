/**
 * Depth sorting for order-dependent gsplat blending.
 *
 * TypeScript reference implementation matching depth_sort.rs — see that
 * file (and `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md` §5) for the
 * algorithm rationale. Produces a back-to-front permutation for the
 * `aSortedIndex` instance attribute: instance `j` renders splat
 * `ordering[j]`, so `ordering[0]` is the farthest splat.
 *
 * Every floating-point step goes through `Math.fround` in the exact
 * operation order of the Rust kernel, so the two backends produce
 * IDENTICAL uint16 keys — the parity contract for a sorting kernel is
 * exact-permutation equality, not almost-equality (a one-bucket key
 * difference would reorder splats).
 */

/** Number of counting-sort buckets (full uint16 key range). */
const DEPTH_SORT_BUCKETS = 1 << 16;

/** Maximum key value as f32 (matches Rust's `DEPTH_KEY_MAX`). */
const DEPTH_KEY_MAX = Math.fround(DEPTH_SORT_BUCKETS - 1);

/** Stand-in for the shard-bounds outputs when no bounds were requested. */
const EMPTY_BOUNDS = new Float32Array(0);

/**
 * Local-space AABB of the elements drawn by each contiguous equal-population
 * range of `ordering` — the twin of Rust's `write_shard_bounds`. See that
 * function (and `docs/guides/specs/CROSS_NODE_DEPTH_ORDERING_SPEC.md` §3.3
 * decision 3) for the rationale.
 *
 * Parity here is exact WITHOUT any `Math.fround` work, unlike the depth-key math
 * below: min/max over f32 values never rounds, so both backends see identical
 * bytes as long as the comparison SHAPE matches. That shape is load-bearing —
 * plain `<` / `>` comparisons are false for NaN, so a NaN center is EXCLUDED
 * from its shard's box rather than poisoning it. `Math.min(NaN, x)` is NaN and
 * would diverge from Rust's `f32::min`, so do not "simplify" this to Math.min.
 * ±Infinity, by contrast, compares successfully and so does reach the box.
 */
function writeShardBounds(
  centers3: Float32Array,
  ordering: Uint32Array,
  count: number,
  shardCount: number,
  shardBoundsMin: Float32Array,
  shardBoundsMax: Float32Array
): void {
  if (shardCount === 0) {
    return;
  }
  for (let i = 0; i < shardCount * 3; i++) {
    shardBoundsMin[i] = Infinity;
    shardBoundsMax[i] = -Infinity;
  }
  if (count === 0) {
    return;
  }

  // Equal-population shards: the same `ceil(count / shardCount)` the main thread
  // uses to size each shard's draw.
  const shardSize = Math.ceil(count / shardCount);
  for (let s = 0; s < shardCount; s++) {
    const lo = s * shardSize;
    if (lo >= count) break;
    const hi = Math.min(lo + shardSize, count);
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (let j = lo; j < hi; j++) {
      const base = ordering[j] * 3;
      const x = centers3[base];
      const y = centers3[base + 1];
      const z = centers3[base + 2];
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }
    const out = s * 3;
    shardBoundsMin[out] = minX;
    shardBoundsMin[out + 1] = minY;
    shardBoundsMin[out + 2] = minZ;
    shardBoundsMax[out] = maxX;
    shardBoundsMax[out + 1] = maxY;
    shardBoundsMax[out + 2] = maxZ;
  }
}

/**
 * Sort splats back-to-front by camera-space depth.
 *
 * @param centers3 - Projected 3D splat centers [count * 3] (x, y, z triplets)
 * @param modelView - Column-major 4x4 model-view matrix [16]
 *   (`camera.matrixWorldInverse × mesh.matrixWorld`, THREE.js layout)
 * @param ordering - Output permutation [count] — `ordering[j]` is the
 *   original splat index drawn at instance slot `j` (slot 0 = farthest)
 * @param count - Number of splats
 * @param shardCount - Number of contiguous equal-population ranges to report
 *   bounds for; 0 skips that work and leaves the two arrays untouched
 * @param shardBoundsMin - Output local-space AABB minima [shardCount * 3]
 * @param shardBoundsMax - Output local-space AABB maxima [shardCount * 3]
 * @returns Number of splats placed via depth keys, or 0 when the identity
 *   fallback was taken (degenerate depth range — ordering is still written, and
 *   the shard bounds are still valid boxes of it, but they carry no DEPTH
 *   meaning, so the caller must not merge them as depth intervals)
 */
export function sort_splats_by_depth(
  centers3: Float32Array,
  modelView: Float32Array,
  ordering: Uint32Array,
  count: number,
  shardCount = 0,
  shardBoundsMin?: Float32Array,
  shardBoundsMax?: Float32Array
): number {
  // Both slices are present whenever shardCount > 0 (the WASM signature makes
  // them required); default to a throwaway so the optional TS arity stays safe.
  const boundsMin = shardBoundsMin ?? EMPTY_BOUNDS;
  const boundsMax = shardBoundsMax ?? EMPTY_BOUNDS;

  if (count === 0) {
    // Still stamp the empty-box sentinels: a caller that reads the arrays
    // without checking the return value must not see a stale previous frame.
    writeShardBounds(centers3, ordering, 0, shardCount, boundsMin, boundsMax);
    return 0;
  }

  // Column-major view-z row: z_view = m2·x + m6·y + m10·z + m14.
  const m2 = modelView[2];
  const m6 = modelView[6];
  const m10 = modelView[10];
  const m14 = modelView[14];

  // Pass 1: camera-space z per splat + min/max over in-front splats
  // (view space looks down -z, so in-front splats have z < 0). The
  // Float32Array scratch rounds the stored value; the intermediate sums
  // are frounded explicitly to mirror Rust's left-associative f32 chain.
  const zScratch = new Float32Array(count);
  let zMin = Infinity;
  let zMax = -Infinity;
  for (let i = 0; i < count; i++) {
    const base = i * 3;
    const z = Math.fround(
      Math.fround(
        Math.fround(Math.fround(m2 * centers3[base]) + Math.fround(m6 * centers3[base + 1])) +
          Math.fround(m10 * centers3[base + 2])
      ) + m14
    );
    zScratch[i] = z;
    if (z < 0.0) {
      zMin = Math.min(zMin, z);
      zMax = Math.max(zMax, z);
    }
  }

  // Degenerate depth range (single depth plane, <=1 in-front splat, or
  // everything behind the camera): identity ordering. `!(zMax > zMin)`
  // rather than `zMax <= zMin` so NaN centers also fall through to
  // identity (matches the Rust kernel).
  if (!(zMax > zMin)) {
    for (let j = 0; j < count; j++) {
      ordering[j] = j;
    }
    writeShardBounds(centers3, ordering, count, shardCount, boundsMin, boundsMax);
    return 0;
  }

  // Pass 2: normalized uint16 keys + histogram. zMin (farthest) -> key 0,
  // zMax (nearest in-front) -> key 65535; behind-camera -> far bucket 0.
  const invRange = Math.fround(1.0 / Math.fround(zMax - zMin));
  const keys = new Uint16Array(count);
  const histogram = new Uint32Array(DEPTH_SORT_BUCKETS);
  for (let i = 0; i < count; i++) {
    const z = zScratch[i];
    let key = 0;
    if (!(z >= 0.0)) {
      const scaled = Math.fround(Math.fround(Math.fround(z - zMin) * invRange) * DEPTH_KEY_MAX);
      // Truncating cast with a 65535 clamp. NOT Math.min: Rust's
      // `f32::min(NaN, 65535.0)` returns 65535 (min yields the OTHER
      // operand on NaN — the saturating `as u16` is never reached with
      // NaN), while `Math.min(NaN, x)` is NaN. The `<` comparison is
      // false for NaN, so a NaN z keys to 65535 exactly like Rust.
      key = (scaled < DEPTH_KEY_MAX ? scaled : DEPTH_KEY_MAX) | 0;
    }
    keys[i] = key;
    histogram[keys[i]]++;
  }

  // Prefix sum: bucket k's write cursor starts after all lower (farther)
  // buckets — ascending keys scatter back-to-front.
  let cursor = 0;
  for (let k = 0; k < DEPTH_SORT_BUCKETS; k++) {
    const bucketCount = histogram[k];
    histogram[k] = cursor;
    cursor += bucketCount;
  }

  // Pass 3: stable scatter (equal keys keep their input order).
  for (let i = 0; i < count; i++) {
    const bucket = keys[i];
    ordering[histogram[bucket]] = i;
    histogram[bucket]++;
  }

  writeShardBounds(centers3, ordering, count, shardCount, boundsMin, boundsMax);

  return count;
}
