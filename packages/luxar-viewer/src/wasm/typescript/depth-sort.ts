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
 *
 * Two entry points, one algorithm (mirrors depth_sort.rs):
 * - {@link sort_splats_by_depth} — stateless free function (per-call
 *   scratch allocations); the raw-kernel parity/bench pin.
 * - {@link DepthSorter} / {@link create_depth_sorter} — the stateful
 *   registered-node lifecycle mirroring the WASM-resident `DepthSorter`
 *   class (one centers copy at construction, reused scratch per sort,
 *   explicit `free()`).
 */

/** Number of counting-sort buckets (full uint16 key range). */
const DEPTH_SORT_BUCKETS = 1 << 16;

/** Maximum key value as f32 (matches Rust's `DEPTH_KEY_MAX`). */
const DEPTH_KEY_MAX = Math.fround(DEPTH_SORT_BUCKETS - 1);

/**
 * Shared three-pass counting-sort core — the single home of the
 * algorithm; both the stateless function and the stateful sorter
 * delegate here (mirrors Rust's `sort_by_depth_core`).
 *
 * Scratch contract: `zScratch`/`keys` need `length >= count` (contents
 * ignored — fully overwritten); `histogram` needs
 * `length >= DEPTH_SORT_BUCKETS` and is zeroed HERE (callers may hand
 * back a dirty buffer from the previous sort). `ordering[0..count)` is
 * always fully written.
 */
function sortByDepthCore(
  centers3: Float32Array,
  modelView: Float32Array,
  zScratch: Float32Array,
  keys: Uint16Array,
  histogram: Uint32Array,
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
  // The histogram may carry the previous sort's cursors — zero it first.
  histogram.fill(0, 0, DEPTH_SORT_BUCKETS);
  const invRange = Math.fround(1.0 / Math.fround(zMax - zMin));
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
  // buckets — ascending keys scatter back-to-front. Reuses the histogram
  // array in place as the write-cursor array.
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

/**
 * Sort splats back-to-front by camera-space depth (stateless form —
 * per-call scratch allocations; hot-path callers should register a
 * {@link DepthSorter} instead).
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
  return sortByDepthCore(
    centers3,
    modelView,
    new Float32Array(count),
    new Uint16Array(count),
    new Uint32Array(DEPTH_SORT_BUCKETS),
    ordering,
    count
  );
}

/**
 * Stateful depth-sorter — the TypeScript twin of the WASM-resident
 * `DepthSorter` class in depth_sort.rs (perf lever L3). Same lifecycle,
 * same fround semantics, exact-permutation-identical outputs:
 * - construction copies the centers ONCE (like the wasm-bindgen boundary
 *   copy) and allocates all scratch;
 * - {@link DepthSorter.sort} reuses every buffer — zero allocations;
 * - {@link DepthSorter.read_ordering_into} is a single memcpy out;
 * - {@link DepthSorter.free} releases the buffers (idempotent). Any use
 *   after `free()` throws, mirroring wasm-bindgen's freed-class
 *   "null pointer passed to rust" error.
 *
 * Before the first `sort`, the stored ordering is the identity (matches
 * the Rust constructor).
 */
export class DepthSorter {
  private centers3: Float32Array;
  private readonly n: number;
  private zScratch: Float32Array;
  private keys: Uint16Array;
  private histogram: Uint32Array;
  private ordering: Uint32Array;
  private freed = false;

  constructor(centers3: Float32Array, count: number) {
    // Same capacity clamp as the Rust factory (and the worker's
    // registerNode clamp).
    const n = Math.min(count, Math.floor(centers3.length / 3));
    this.n = n;
    // Copy-in at registration: mutating the caller's buffer afterwards
    // must not change results (the WASM twin inherently copies).
    this.centers3 = centers3.slice(0, n * 3);
    this.zScratch = new Float32Array(n);
    this.keys = new Uint16Array(n);
    this.histogram = new Uint32Array(DEPTH_SORT_BUCKETS);
    this.ordering = new Uint32Array(n);
    for (let i = 0; i < n; i++) this.ordering[i] = i;
  }

  /** Mirror wasm-bindgen's post-free behavior: methods throw. */
  private assertLive(): void {
    if (this.freed) {
      throw new Error('DepthSorter used after free()');
    }
  }

  /**
   * Recompute the back-to-front ordering for the given column-major 4x4
   * model-view. Returns the number of splats placed via depth keys
   * (0 = identity fallback; ordering still fully written).
   */
  sort(modelView: Float32Array): number {
    this.assertLive();
    return sortByDepthCore(
      this.centers3,
      modelView,
      this.zScratch,
      this.keys,
      this.histogram,
      this.ordering,
      this.n
    );
  }

  /** Number of splats (after the construction-time capacity clamp). */
  count(): number {
    this.assertLive();
    return this.n;
  }

  /** Copy the current ordering into `target` (`length >= count()`). */
  read_ordering_into(target: Uint32Array): void {
    this.assertLive();
    target.set(this.ordering);
  }

  /**
   * Release the buffers (idempotent, like wasm-bindgen's `free()`).
   * Buffers are re-pointed at empty arrays so a retained-but-freed
   * sorter cannot pin multi-MB data against GC.
   */
  free(): void {
    if (this.freed) return;
    this.freed = true;
    this.centers3 = new Float32Array(0);
    this.zScratch = new Float32Array(0);
    this.keys = new Uint16Array(0);
    this.histogram = new Uint32Array(0);
    this.ordering = new Uint32Array(0);
  }
}

/**
 * Factory mirroring the Rust `create_depth_sorter` export — the
 * `WasmModule` interface types this as the stateful entry point for both
 * backends.
 */
export function create_depth_sorter(centers3: Float32Array, count: number): DepthSorter {
  return new DepthSorter(centers3, count);
}
