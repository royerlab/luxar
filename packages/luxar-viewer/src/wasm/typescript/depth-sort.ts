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

/**
 * Sort splats back-to-front by camera-space depth.
 *
 * @param centers3 - Projected 3D splat centers [count * 3] (x, y, z triplets)
 * @param modelView - Column-major 4x4 model-view matrix [16]
 *   (`camera.matrixWorldInverse × mesh.matrixWorld`, THREE.js layout)
 * @param ordering - Output permutation [count] — `ordering[j]` is the
 *   original splat index drawn at instance slot `j` (slot 0 = farthest)
 * @param count - Number of splats
 * @returns Number of splats placed via depth keys, or 0 when the identity
 *   fallback was taken (degenerate depth range — ordering is still written)
 */
export function sort_splats_by_depth(
  centers3: Float32Array,
  modelView: Float32Array,
  ordering: Uint32Array,
  count: number
): number {
  if (count === 0) {
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

  return count;
}
