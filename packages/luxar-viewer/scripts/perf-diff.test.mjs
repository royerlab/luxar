import { describe, it, expect } from 'vitest';
import { buildPerfDiff } from './perf-diff.mjs';

/**
 * Slice the markdown from a section heading to the next `## ` heading (or
 * end), so assertions target one section and don't accidentally match the
 * frame-timing table (whose rows also start with the scenario key).
 */
function section(md, heading) {
  const start = md.indexOf(heading);
  if (start === -1) return '';
  const rest = md.slice(start + heading.length);
  const nextIdx = rest.indexOf('\n## ');
  return heading + (nextIdx === -1 ? rest : rest.slice(0, nextIdx));
}

/** Split a markdown table row `| a | b | c |` into trimmed cells. */
function cellsOf(line) {
  return line
    .split('|')
    .slice(1, -1)
    .map((c) => c.trim());
}

/**
 * Extract the Δ-column cell for a given field label from a section, for
 * the row keyed by `rowKey`. Pins the exact delta cell so a fabricated
 * delta (NaN%, -100.0%, +Infinity%) cannot hide behind a value cell that
 * independently renders `—`.
 */
function deltaCell(sec, rowKey, label) {
  const lines = sec.split('\n');
  const header = lines.find((l) => l.startsWith('| Scenario / backend |'));
  if (!header) throw new Error(`no header in section for label "${label}"`);
  const idx = cellsOf(header).indexOf(`Δ ${label}`);
  if (idx === -1) throw new Error(`no "Δ ${label}" column in section`);
  const row = lines.find((l) => l.startsWith(`| ${rowKey} |`));
  if (!row) throw new Error(`no row for "${rowKey}" in section`);
  return cellsOf(row)[idx];
}

/** Assert a Δ cell is exactly `—` with no fabricated numeric delta. */
function expectMissingDelta(cell) {
  expect(cell).toBe('—');
  expect(cell).not.toContain('%');
  expect(cell).not.toContain('NaN');
  expect(cell).not.toContain('Infinity');
  expect(cell).not.toContain('🟢');
  expect(cell).not.toContain('🔴');
}

/**
 * Minimal PerfRunResult wrapper around a list of scenarios.
 */
function run(scenarios, overrides = {}) {
  return {
    capturedAt: '2026-01-01T00:00:00.000Z',
    commit: 'abc1234',
    sampleWindowMs: 2000,
    warmupFrames: 30,
    scenarios,
    ...overrides,
  };
}

/**
 * A gsplat-shaped scenario. Pass partial overrides for the fields under
 * test; everything else gets sane defaults so the frame table renders.
 */
function scn(overrides = {}) {
  return {
    scenarioId: 's1',
    scenarioLabel: 'Scenario 1',
    backend: 'webgpu',
    actualApi: 'webgpu',
    isWebGLBackend: false,
    visibleSegments: 1000,
    elementCount: 1000,
    frameMs: { count: 100, median: 5, p95: 8, p99: 9, mean: 5, min: 4, max: 10 },
    firstRenderMs: 12,
    depthSort: null,
    notes: [],
    skipped: false,
    ...overrides,
  };
}

/**
 * A line-bench-shaped scenario: no depthSort key at all, no gsplat
 * campaign metrics. Used to prove the new sections stay omitted.
 */
function lineScn(overrides = {}) {
  return {
    scenarioId: 'line1',
    backend: 'webgl',
    actualApi: 'webgl',
    isWebGLBackend: false,
    visibleSegments: 5000,
    frameMs: { count: 100, median: 3, p95: 4, p99: 5, mean: 3, min: 2, max: 6 },
    skipped: false,
    ...overrides,
  };
}

describe('buildPerfDiff — frame timing (baseline behaviour)', () => {
  it('renders the JS frame timing table for a normal input', () => {
    const base = run([scn({ frameMs: { median: 5, p95: 8 } })]);
    const next = run([scn({ frameMs: { median: 5, p95: 8 } })]);
    const md = buildPerfDiff(base, next);
    expect(md).toContain('## JS frame timing');
    expect(md).toContain('s1/webgpu');
    // Legend preserved.
    expect(md).toContain('🟢 = ≥5% faster');
  });

  it('marks the api column (sw) when either side ran on a software rasterizer', () => {
    const base = run([scn({ softwareRenderer: true })]);
    const next = run([scn({ softwareRenderer: false })]);
    const md = buildPerfDiff(base, next);
    const row = md.split('\n').find((l) => l.startsWith('| s1/webgpu |'));
    expect(row).toBeDefined();
    expect(cellsOf(row)[1]).toBe('webgpu (sw)');
  });

  it('keeps the api column unmarked when softwareRenderer is absent (line bench)', () => {
    const md = buildPerfDiff(run([lineScn()]), run([lineScn()]));
    const row = md.split('\n').find((l) => l.startsWith('| line1/webgl |'));
    expect(row).toBeDefined();
    expect(cellsOf(row)[1]).toBe('webgl');
  });

  it('omits the new sections for a pure line-bench input', () => {
    const base = run([lineScn()]);
    const next = run([lineScn()]);
    const md = buildPerfDiff(base, next);
    expect(md).toContain('## JS frame timing');
    expect(md).not.toContain('## Depth-sort stages');
    expect(md).not.toContain('## L8 sort-tail');
    expect(md).not.toContain('## Ladder load');
    expect(md).not.toContain('## GPU pass time');
  });
});

describe('buildPerfDiff — depth-sort stages', () => {
  const fields = [
    ['kernelMsMedian', 'kernel'],
    ['queueMsMedian', 'queue'],
    ['boundaryMsMedian', 'boundary'],
    ['sortLatencyMedianMs', 'sortLat med'],
    ['sortLatencyP95Ms', 'sortLat p95'],
  ];

  for (const [key, label] of fields) {
    it(`flags a 10x regression in depthSort.${key} with 🔴`, () => {
      const base = run([scn({ depthSort: { [key]: 1 } })]);
      const next = run([scn({ depthSort: { [key]: 10 } })]);
      const md = buildPerfDiff(base, next);
      expect(md).toContain('## Depth-sort stages');
      expect(md).toContain(`base ${label}`);
      expect(md).toContain('🔴');
      expect(md).toContain('+900.0%');
    });
  }

  it('renders — (not a fabricated delta) when a field is present on base but absent on next', () => {
    const base = run([scn({ depthSort: { kernelMsMedian: 2 } })]);
    const next = run([scn({ depthSort: { queueMsMedian: 3 } })]);
    const md = buildPerfDiff(base, next);
    const sec = section(md, '## Depth-sort stages');
    expect(sec).not.toBe('');
    // kernel: base=2, new absent → Δ must be —, never NaN%/-100%/0.0%.
    expectMissingDelta(deltaCell(sec, 's1/webgpu', 'kernel'));
    // queue: base absent, new=3 → Δ must be — as well.
    expectMissingDelta(deltaCell(sec, 's1/webgpu', 'queue'));
  });

  it('renders — for a null sort-latency without a phantom delta', () => {
    const base = run([scn({ depthSort: { sortLatencyMedianMs: 4, kernelMsMedian: 1 } })]);
    const next = run([scn({ depthSort: { sortLatencyMedianMs: null, kernelMsMedian: 1 } })]);
    const md = buildPerfDiff(base, next);
    const sec = section(md, '## Depth-sort stages');
    expect(sec).not.toBe('');
    // kernel is equal (1 vs 1) → 0.0% is legitimate there, but the
    // sortLat column must not fabricate a delta for the null side.
    expectMissingDelta(deltaCell(sec, 's1/webgpu', 'sortLat med'));
    // Sanity: the genuine equal-value delta is untouched.
    expect(deltaCell(sec, 's1/webgpu', 'kernel')).toBe('0.0%');
  });
});

describe('buildPerfDiff — L8 sort-tail p99', () => {
  it('flags a 10x regression in sortAdjacentP99Ms with 🔴', () => {
    const base = run([scn({ sortAdjacentP99Ms: 2 })]);
    const next = run([scn({ sortAdjacentP99Ms: 20 })]);
    const md = buildPerfDiff(base, next);
    expect(md).toContain('## L8 sort-tail (p99)');
    expect(md).toContain('🔴');
    expect(md).toContain('+900.0%');
  });

  it('flags a 10x regression in idleOrbitP99Ms with 🔴', () => {
    const base = run([scn({ idleOrbitP99Ms: 1.5 })]);
    const next = run([scn({ idleOrbitP99Ms: 15 })]);
    const md = buildPerfDiff(base, next);
    expect(md).toContain('## L8 sort-tail (p99)');
    expect(md).toContain('🔴');
    expect(md).toContain('+900.0%');
  });

  it('omits the section when every scenario has null/absent p99s', () => {
    const base = run([scn({ sortAdjacentP99Ms: null, idleOrbitP99Ms: null })]);
    const next = run([scn({ sortAdjacentP99Ms: null })]);
    const md = buildPerfDiff(base, next);
    expect(md).not.toContain('## L8 sort-tail');
  });

  it('only renders the row for the scenario that carries the p99s', () => {
    const base = run([
      scn({ scenarioId: 's10m', sortAdjacentP99Ms: 3 }),
      scn({ scenarioId: 'other' }),
    ]);
    const next = run([
      scn({ scenarioId: 's10m', sortAdjacentP99Ms: 3 }),
      scn({ scenarioId: 'other' }),
    ]);
    const md = buildPerfDiff(base, next);
    const section = md.slice(md.indexOf('## L8 sort-tail'));
    expect(section).toContain('s10m/webgpu');
    expect(section).not.toContain('other/webgpu');
  });
});

describe('buildPerfDiff — ladder load', () => {
  it('flags a 10x regression in wallMsToLadderComplete with 🔴', () => {
    const base = run([
      scn({
        ladder: { wallMsToLadderComplete: 100, observedGrowth: true, loadWindowFrameMs: null },
      }),
    ]);
    const next = run([
      scn({
        ladder: { wallMsToLadderComplete: 1000, observedGrowth: true, loadWindowFrameMs: null },
      }),
    ]);
    const md = buildPerfDiff(base, next);
    expect(md).toContain('## Ladder load');
    expect(md).toContain('🔴');
    expect(md).toContain('+900.0%');
  });

  it('annotates observedGrowth:false with (lb) and emits no delta', () => {
    const base = run([
      scn({
        ladder: { wallMsToLadderComplete: 100, observedGrowth: true, loadWindowFrameMs: null },
      }),
    ]);
    const next = run([
      scn({
        ladder: { wallMsToLadderComplete: 50, observedGrowth: false, loadWindowFrameMs: null },
      }),
    ]);
    const md = buildPerfDiff(base, next);
    const sec = section(md, '## Ladder load');
    expect(sec).not.toBe('');
    const line = sec.split('\n').find((l) => l.startsWith('| s1/webgpu |'));
    expect(line).toBeDefined();
    expect(line).toContain('(lb)');
    // A misleading delta must not be computed when a side is a lower bound.
    expect(line).not.toContain('%');
    expect(line).toContain('—');
  });

  it('omits the section when no scenario carries a ladder object', () => {
    const base = run([scn()]);
    const next = run([scn()]);
    const md = buildPerfDiff(base, next);
    expect(md).not.toContain('## Ladder load');
  });
});

describe('buildPerfDiff — missing side handling', () => {
  it('renders — in the Δ cell for an absent metric on either side', () => {
    const base = run([scn({ depthSort: { kernelMsMedian: 2 }, sortAdjacentP99Ms: 3 })]);
    const next = run([scn({ depthSort: null })]);
    const md = buildPerfDiff(base, next);
    // Depth-sort + L8 columns exist (base carried them) but the new side
    // is missing — the Δ cell must be —, never a fabricated delta.
    const depth = section(md, '## Depth-sort stages');
    const l8 = section(md, '## L8 sort-tail (p99)');
    expect(depth).not.toBe('');
    expect(l8).not.toBe('');
    expectMissingDelta(deltaCell(depth, 's1/webgpu', 'kernel'));
    expectMissingDelta(deltaCell(l8, 's1/webgpu', 'sortAdjacent p99'));
  });

  it('renders — (not Infinity%) in the Δ cell when the baseline is zero', () => {
    const base = run([scn({ depthSort: { kernelMsMedian: 0 } })]);
    const next = run([scn({ depthSort: { kernelMsMedian: 5 } })]);
    const md = buildPerfDiff(base, next);
    const sec = section(md, '## Depth-sort stages');
    expect(sec).not.toBe('');
    // base=0, new=5: a percent change is undefined — Δ must be —, never
    // +Infinity% (division by a zero baseline).
    expectMissingDelta(deltaCell(sec, 's1/webgpu', 'kernel'));
  });
});
