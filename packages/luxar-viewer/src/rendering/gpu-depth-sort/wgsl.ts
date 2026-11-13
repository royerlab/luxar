/**
 * WGSL for the GPU depth sort, one shader module per pass.
 *
 * The passes, in dispatch order (see `gpu-depth-sorter.ts`):
 *
 * 1. `computeDepth` — each element's view-space z from its element-texture
 *    center, plus the min/max over in-front elements (z < 0), reduced per
 *    workgroup and merged with atomics on an order-preserving u32 encoding.
 * 2. `computeKeys` — the 16-bit back-to-front key, written over the depth.
 * 3. `histogram` — per block, the count of each 4-bit digit at a shift.
 * 4. `scanBlocks` / `addTotals` — the multi-level exclusive scan of the
 *    digit-major histogram, giving each block its write offset per digit.
 * 5. `scatterFirst` / `scatterMiddle` / `scatterLast` — the STABLE scatter of
 *    one pass (see `scatter` below). Steps 3-5 repeat for each of the four
 *    4-bit digits, least significant first: the first pass reads the keys and
 *    the implicit element index, the last writes the final element indices.
 *
 * The key arithmetic is the CPU kernel's (`wasm/typescript/depth-sort.ts`):
 * view z = m2·x + m6·y + m10·z + m14; zMin/zMax over z < 0; a degenerate range
 * (`!(zMax > zMin)`) keys everything 0, which the stable sort turns into the
 * identity order, like the CPU's identity fallback; otherwise
 * `key = min(trunc((z − zMin) · (1 / (zMax − zMin)) · 65535), 65535)` for an
 * element in front of the camera and 0 behind it. WGSL may contract a
 * multiply-add into an FMA and its division is accurate to 2.5 ULP (refined
 * here by one Newton step), so a key near a bucket edge can land one bucket
 * from the CPU's. The sort of the GPU's own keys is exact and stable.
 *
 * @module rendering/gpu-depth-sort/wgsl
 */

import { BLOCK_ELEMENTS, ELEMENTS_PER_INVOCATION, RADIX, WORKGROUP_SIZE } from './plan';

/** Mirrors `SortParams` / `packParams` in `plan.ts`. */
const PARAMS_STRUCT = /* wgsl */ `
struct Params {
  viewZ: vec4f,
  count: u32,
  blocks: u32,
  textureWidth: u32,
  texelsPerElement: u32,
  centerMode: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}
`;

/** The digit shift of one radix pass. */
const PASS_STRUCT = /* wgsl */ `
struct Pass {
  shift: u32,
}
`;

/** The entry count of one scan level. */
const SCAN_STRUCT = /* wgsl */ `
struct Scan {
  len: u32,
}
`;

const CONSTANTS = /* wgsl */ `
const WG: u32 = ${WORKGROUP_SIZE}u;
const EPT: u32 = ${ELEMENTS_PER_INVOCATION}u;
const BLOCK: u32 = ${BLOCK_ELEMENTS}u;
const RADIX: u32 = ${RADIX}u;
const F32_MAX: f32 = 0x1.fffffep+127f;
`;

/**
 * Order-preserving f32 <-> u32: a larger float maps to a larger u32, so
 * `atomicMin` / `atomicMax` on the encoding reduce floats.
 */
const ORDERED_BITS = /* wgsl */ `
fn orderedBits(f: f32) -> u32 {
  let u = bitcast<u32>(f);
  return select(u | 0x80000000u, ~u, (u & 0x80000000u) != 0u);
}

fn fromOrderedBits(u: u32) -> f32 {
  return bitcast<f32>(select(~u, u & 0x7fffffffu, (u & 0x80000000u) != 0u));
}
`;

/** Inclusive scan over the workgroup; call from uniform control flow. */
const WORKGROUP_SCAN = /* wgsl */ `
var<workgroup> scanScratch: array<u32, ${WORKGROUP_SIZE}>;

fn inclusiveScan(lid: u32, value: u32) -> u32 {
  scanScratch[lid] = value;
  workgroupBarrier();
  for (var d = 1u; d < WG; d = d << 1u) {
    var x = scanScratch[lid];
    if (lid >= d) {
      x = x + scanScratch[lid - d];
    }
    workgroupBarrier();
    scanScratch[lid] = x;
    workgroupBarrier();
  }
  return scanScratch[lid];
}
`;

const DEPTH = /* wgsl */ `
${CONSTANTS}${PARAMS_STRUCT}${ORDERED_BITS}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var centers: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> depth: array<u32>;
@group(0) @binding(3) var<storage, read_write> bounds: array<atomic<u32>, 2>;

var<workgroup> lows: array<f32, ${WORKGROUP_SIZE}>;
var<workgroup> highs: array<f32, ${WORKGROUP_SIZE}>;

fn loadCenter(i: u32) -> vec3f {
  let texel = i * params.texelsPerElement;
  let coord = vec2u(texel % params.textureWidth, texel / params.textureWidth);
  let a = textureLoad(centers, coord, 0).xyz;
  if (params.centerMode == 1u) {
    // A line segment sorts on its midpoint; its texels never straddle a row.
    let b = textureLoad(centers, coord + vec2u(1u, 0u), 0).xyz;
    return (a + b) * 0.5;
  }
  return a;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn computeDepth(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_index) lid: u32
) {
  var lo = F32_MAX;
  var hi = -F32_MAX;
  for (var k = 0u; k < EPT; k = k + 1u) {
    let i = wg.x * BLOCK + k * WG + lid;
    if (i < params.count) {
      let c = loadCenter(i);
      let v = params.viewZ;
      let z = ((v.x * c.x + v.y * c.y) + v.z * c.z) + v.w;
      depth[i] = bitcast<u32>(z);
      if (z < 0.0) {
        lo = min(lo, z);
        hi = max(hi, z);
      }
    }
  }
  lows[lid] = lo;
  highs[lid] = hi;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (lid < s) {
      lows[lid] = min(lows[lid], lows[lid + s]);
      highs[lid] = max(highs[lid], highs[lid + s]);
    }
    workgroupBarrier();
  }
  // lows <= highs only when the block holds an in-front element.
  if (lid == 0u && lows[0] <= highs[0]) {
    atomicMin(&bounds[0], orderedBits(lows[0]));
    atomicMax(&bounds[1], orderedBits(highs[0]));
  }
}
`;

const KEYS = /* wgsl */ `
${CONSTANTS}${PARAMS_STRUCT}${ORDERED_BITS}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> depthKeys: array<u32>;
@group(0) @binding(2) var<storage, read> bounds: array<u32, 2>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn computeKeys(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_index) lid: u32
) {
  // No in-front element leaves the bounds at their initial values, which
  // decode to NaN: the range is then degenerate, as on the CPU.
  let zMin = fromOrderedBits(bounds[0]);
  let zMax = fromOrderedBits(bounds[1]);
  let ranged = zMax > zMin;
  let range = zMax - zMin;
  var inv = 1.0 / range;
  inv = inv * (2.0 - range * inv);
  for (var k = 0u; k < EPT; k = k + 1u) {
    let i = wg.x * BLOCK + k * WG + lid;
    if (i < params.count) {
      let z = bitcast<f32>(depthKeys[i]);
      var key = 0u;
      if (ranged && !(z >= 0.0)) {
        let scaled = (z - zMin) * inv * 65535.0;
        key = select(65535u, u32(scaled), scaled < 65535.0);
      }
      depthKeys[i] = key;
    }
  }
}
`;

const HISTOGRAM = /* wgsl */ `
${CONSTANTS}${PARAMS_STRUCT}${PASS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> pass_: Pass;
@group(0) @binding(2) var<storage, read> keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> hist: array<u32>;

var<workgroup> bins: array<atomic<u32>, ${RADIX}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn histogram(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_index) lid: u32
) {
  if (lid < RADIX) {
    atomicStore(&bins[lid], 0u);
  }
  workgroupBarrier();
  for (var k = 0u; k < EPT; k = k + 1u) {
    let i = wg.x * BLOCK + k * WG + lid;
    if (i < params.count) {
      atomicAdd(&bins[(keys[i] >> pass_.shift) & (RADIX - 1u)], 1u);
    }
  }
  workgroupBarrier();
  if (lid < RADIX) {
    hist[lid * params.blocks + wg.x] = atomicLoad(&bins[lid]);
  }
}
`;

const SCAN = /* wgsl */ `
${CONSTANTS}${SCAN_STRUCT}${WORKGROUP_SCAN}
@group(0) @binding(0) var<uniform> scan: Scan;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> totals: array<u32>;

// Exclusive scan of one 1024-entry tile in place; its total goes to totals.
@compute @workgroup_size(${WORKGROUP_SIZE})
fn scanBlocks(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_index) lid: u32
) {
  let base = wg.x * WG * 4u + lid * 4u;
  var prefix: array<u32, 4>;
  var run = 0u;
  for (var k = 0u; k < 4u; k = k + 1u) {
    prefix[k] = run;
    if (base + k < scan.len) {
      run = run + data[base + k];
    }
  }
  let inclusive = inclusiveScan(lid, run);
  let before = inclusive - run;
  for (var k = 0u; k < 4u; k = k + 1u) {
    if (base + k < scan.len) {
      data[base + k] = prefix[k] + before;
    }
  }
  if (lid == WG - 1u) {
    totals[wg.x] = inclusive;
  }
}

// Add each tile's scanned total (from the coarser level) to its entries.
@compute @workgroup_size(${WORKGROUP_SIZE})
fn addTotals(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_index) lid: u32
) {
  let add = totals[wg.x];
  let base = wg.x * WG * 4u + lid * 4u;
  for (var k = 0u; k < 4u; k = k + 1u) {
    if (base + k < scan.len) {
      data[base + k] = data[base + k] + add;
    }
  }
}
`;

/** Packed counter fields: two 16-bit digit counts per u32. */
const PACKED_FIELDS = RADIX / 2;
/**
 * The count scan's segments: one invocation per (field, segment) pair, so the
 * workgroup splits into PACKED_FIELDS × SCAN_SEGMENTS of them.
 */
const SCAN_SEGMENTS = WORKGROUP_SIZE / PACKED_FIELDS;
const SCAN_SEGMENT_LENGTH = WORKGROUP_SIZE / SCAN_SEGMENTS;

/**
 * Sixteen 16-bit digit counters packed in eight u32s, and their workgroup scan.
 * A field never overflows: a block holds 4096 elements.
 */
const PACKED_COUNTS = /* wgsl */ `
const PACKED: u32 = RADIX / 2u;

fn countOf(c: ptr<function, array<u32, PACKED>>, d: u32) -> u32 {
  return ((*c)[d >> 1u] >> ((d & 1u) * 16u)) & 0xffffu;
}

fn bump(c: ptr<function, array<u32, PACKED>>, d: u32) {
  (*c)[d >> 1u] = (*c)[d >> 1u] + (1u << ((d & 1u) * 16u));
}

var<workgroup> countScan: array<array<u32, PACKED>, ${WORKGROUP_SIZE}>;
var<workgroup> segmentTotals: array<array<u32, PACKED>, ${SCAN_SEGMENTS}>;

// Exclusive scan of every invocation's counters across the workgroup, in three
// barriers: each of the ${WORKGROUP_SIZE} invocations scans one packed field of
// one segment of ${SCAN_SEGMENT_LENGTH} invocations; the first PACKED
// invocations scan the segment totals; each invocation adds its segment's
// prefix. Field-wise sums never carry into a neighbour, since totals stay
// <= 4096.
fn countsBefore(lid: u32, own: array<u32, PACKED>) -> array<u32, PACKED> {
  countScan[lid] = own;
  workgroupBarrier();
  let field = lid % PACKED;
  let segment = lid / PACKED;
  var run = 0u;
  for (var t = 0u; t < ${SCAN_SEGMENT_LENGTH}u; t = t + 1u) {
    let at = segment * ${SCAN_SEGMENT_LENGTH}u + t;
    let c = countScan[at][field];
    countScan[at][field] = run;
    run = run + c;
  }
  segmentTotals[segment][field] = run;
  workgroupBarrier();
  if (lid < PACKED) {
    var total = 0u;
    for (var g = 0u; g < ${SCAN_SEGMENTS}u; g = g + 1u) {
      let c = segmentTotals[g][lid];
      segmentTotals[g][lid] = total;
      total = total + c;
    }
  }
  workgroupBarrier();
  var before = countScan[lid];
  let base = segmentTotals[lid / ${SCAN_SEGMENT_LENGTH}u];
  for (var j = 0u; j < PACKED; j = j + 1u) {
    before[j] = before[j] + base[j];
  }
  return before;
}
`;

/** Which radix pass a scatter shader serves. */
type ScatterMode = 'first' | 'middle' | 'last';

/**
 * The stable scatter of one radix pass. Each invocation owns ELEMENTS_PER_
 * INVOCATION consecutive elements of its block and ranks them in order, so an
 * element's destination is its block's offset for its digit, plus that digit's
 * count in earlier invocations of the block, plus its count among the
 * invocation's earlier elements.
 *
 * `first` sorts the key buffer and emits keys and element indices; `middle`
 * carries keys and indices on; `last` emits the indices only.
 */
function scatter(mode: ScatterMode): string {
  const valuesIn =
    mode === 'first' ? '' : '@group(0) @binding(4) var<storage, read> valuesIn4: array<vec4u>;';
  const outBase = mode === 'first' ? 4 : 5;
  const keysOut =
    mode === 'last'
      ? ''
      : `@group(0) @binding(${outBase}) var<storage, read_write> keysOut: array<u32>;`;
  const valuesOutBinding = mode === 'last' ? 5 : outBase + 1;
  const value = mode === 'first' ? 'i' : 'values[k]';
  const loadValues =
    mode === 'first'
      ? ''
      : /* wgsl */ `
  var values: array<u32, ${ELEMENTS_PER_INVOCATION}>;
  for (var q = 0u; q < EPT / 4u; q = q + 1u) {
    let v = valuesIn4[first / 4u + q];
    values[q * 4u] = v.x;
    values[q * 4u + 1u] = v.y;
    values[q * 4u + 2u] = v.z;
    values[q * 4u + 3u] = v.w;
  }`;
  const writeKey = mode === 'last' ? '' : 'keysOut[dest] = key;';
  return /* wgsl */ `
${CONSTANTS}${PARAMS_STRUCT}${PASS_STRUCT}${PACKED_COUNTS}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> pass_: Pass;
@group(0) @binding(2) var<storage, read> keysIn4: array<vec4u>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
${valuesIn}
${keysOut}
@group(0) @binding(${valuesOutBinding}) var<storage, read_write> valuesOut: array<u32>;

var<workgroup> blockOffset: array<u32, ${RADIX}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_index) lid: u32
) {
  if (lid < RADIX) {
    blockOffset[lid] = offsets[lid * params.blocks + wg.x];
  }
  let first = wg.x * BLOCK + lid * EPT;
  // The invocation's keys, read once as 16-byte loads (a block never splits a
  // vec4: BLOCK and EPT are multiples of 4).
  var keys: array<u32, ${ELEMENTS_PER_INVOCATION}>;
  for (var q = 0u; q < EPT / 4u; q = q + 1u) {
    let v = keysIn4[first / 4u + q];
    keys[q * 4u] = v.x;
    keys[q * 4u + 1u] = v.y;
    keys[q * 4u + 2u] = v.z;
    keys[q * 4u + 3u] = v.w;
  }
  var own: array<u32, PACKED>;
  var rank: array<u32, ${ELEMENTS_PER_INVOCATION}>;
  for (var k = 0u; k < EPT; k = k + 1u) {
    if (first + k < params.count) {
      let d = (keys[k] >> pass_.shift) & (RADIX - 1u);
      rank[k] = countOf(&own, d);
      bump(&own, d);
    }
  }
  var before = countsBefore(lid, own);${loadValues}
  for (var k = 0u; k < EPT; k = k + 1u) {
    let i = first + k;
    if (i < params.count) {
      let key = keys[k];
      let d = (key >> pass_.shift) & (RADIX - 1u);
      let dest = blockOffset[d] + countOf(&before, d) + rank[k];
      ${writeKey}
      valuesOut[dest] = ${value};
    }
  }
}
`;
}

/** Every shader module, by pass, with its entry point. */
export const DEPTH_SORT_SHADERS = {
  depth: { code: DEPTH, entryPoint: 'computeDepth' },
  keys: { code: KEYS, entryPoint: 'computeKeys' },
  histogram: { code: HISTOGRAM, entryPoint: 'histogram' },
  scanBlocks: { code: SCAN, entryPoint: 'scanBlocks' },
  addTotals: { code: SCAN, entryPoint: 'addTotals' },
  scatterFirst: { code: scatter('first'), entryPoint: 'main' },
  scatterMiddle: { code: scatter('middle'), entryPoint: 'main' },
  scatterLast: { code: scatter('last'), entryPoint: 'main' },
} as const;

/** A pass of the sort. */
export type DepthSortPass = keyof typeof DEPTH_SORT_SHADERS;
