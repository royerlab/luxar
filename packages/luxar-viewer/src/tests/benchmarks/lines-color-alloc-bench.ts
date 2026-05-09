/**
 * Lines TS-path color-alloc audit.
 *
 * Measures the cost of `coerceColorsToFloat32` (upfront alloc + scale)
 * vs an inline per-vertex-pair scaler closure for a representative
 * Lines dataset, ~95% culled (worst case — alloc/scale work that's
 * mostly thrown away).
 *
 * Threshold: 1ms per call. Below that the upfront alloc is a
 * theoretical concern only; above that the closure approach is
 * worth the added per-vertex multiplication.
 *
 * Recorded results (Apple Silicon, Node 22):
 *   |  segments | current (ms) | inline (ms) | delta (ms) |
 *   |-----------|--------------|-------------|------------|
 *   |    10_000 |        0.075 |       0.003 |      0.072 |
 *   |   100_000 |        0.490 |       0.021 |      0.469 |
 *   |   500_000 |        1.914 |       0.071 |      1.843 |
 *
 * For typical Lines workloads (10k-100k segments) the alloc cost is
 * sub-1ms — well below the threshold. It only crosses at 500k+
 * segments, AND only when input is Uint8/Uint16. Float32 inputs
 * (the dominant production HDR pipeline) pass through with zero
 * alloc. Won't-fix as long as 500k+ segment workloads aren't
 * routine; revisit when they are.
 *
 * Run with: npx tsx src/tests/benchmarks/lines-color-alloc-bench.ts
 *           npx tsx src/tests/benchmarks/lines-color-alloc-bench.ts 100000
 */

// Allow scenario sweep via CLI arg: `... -- 10000` or `... -- 100000`.
const SEGMENTS = Number(process.argv[2] ?? 10_000);
const VERTICES = SEGMENTS * 2;
const VISIBLE_FRACTION = 0.05; // 95% culled — worst case for the alloc/scale audit
const ITERATIONS = 50;
const WARMUP = 5;

function makeUint8Colors(): Uint8Array {
  const out = new Uint8Array(VERTICES * 3);
  for (let i = 0; i < out.length; i++) out[i] = i & 0xff;
  return out;
}

function coerceColorsToFloat32(colors: Uint8Array): Float32Array {
  const norm = 1 / 255;
  const out = new Float32Array(colors.length);
  for (let i = 0; i < colors.length; i++) out[i] = colors[i] * norm;
  return out;
}

/** Current approach: pre-normalize whole buffer up front. */
function currentApproach(colors: Uint8Array): number {
  const colorsF32 = coerceColorsToFloat32(colors);
  let sum = 0;
  // Simulate visible-segment reads (5% of segments touch 6 colors each).
  const visible = Math.floor(SEGMENTS * VISIBLE_FRACTION);
  for (let i = 0; i < visible; i++) {
    const v0 = (i * 11) % VERTICES;
    const v1 = (i * 11 + 7) % VERTICES;
    sum +=
      colorsF32[v0 * 3] +
      colorsF32[v0 * 3 + 1] +
      colorsF32[v0 * 3 + 2] +
      colorsF32[v1 * 3] +
      colorsF32[v1 * 3 + 1] +
      colorsF32[v1 * 3 + 2];
  }
  return sum;
}

/** Alternative: read raw bytes + scale per visible vertex pair. */
function inlineApproach(colors: Uint8Array): number {
  const norm = 1 / 255;
  let sum = 0;
  const visible = Math.floor(SEGMENTS * VISIBLE_FRACTION);
  for (let i = 0; i < visible; i++) {
    const v0 = (i * 11) % VERTICES;
    const v1 = (i * 11 + 7) % VERTICES;
    sum +=
      colors[v0 * 3] * norm +
      colors[v0 * 3 + 1] * norm +
      colors[v0 * 3 + 2] * norm +
      colors[v1 * 3] * norm +
      colors[v1 * 3 + 1] * norm +
      colors[v1 * 3 + 2] * norm;
  }
  return sum;
}

function timeFn(fn: () => number, label: string): number {
  for (let i = 0; i < WARMUP; i++) fn();
  const t0 = performance.now();
  let acc = 0;
  for (let i = 0; i < ITERATIONS; i++) acc += fn();
  const t1 = performance.now();
  const avg = (t1 - t0) / ITERATIONS;
  // Use acc to prevent dead-code elimination
  if (Number.isNaN(acc)) console.error('unreachable');
  console.log(`  ${label}: ${avg.toFixed(3)} ms/op (${ITERATIONS} iters)`);
  return avg;
}

function main(): void {
  const colors = makeUint8Colors();
  console.log(
    `Lines color-alloc bench: ${SEGMENTS} segments, ${VERTICES} vertices, ${(
      VISIBLE_FRACTION * 100
    ).toFixed(0)}% visible (worst case for upfront-alloc cost)`
  );
  console.log('Input dtype: Uint8Array (the only case where alloc is needed)');
  console.log();
  const cur = timeFn(() => currentApproach(colors), 'current (coerceColorsToFloat32)');
  const inl = timeFn(() => inlineApproach(colors), 'inline (per-vertex scale)');
  console.log();
  console.log(`Delta: ${(cur - inl).toFixed(3)} ms/op (current - inline)`);
  console.log('Threshold for concern: > 1.000 ms/op');
  console.log(
    `Decision: ${cur - inl < 1.0 ? "WON'T FIX (sub-1ms; theoretical concern)" : 'FIX (meaningful)'}`
  );
}

main();
