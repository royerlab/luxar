import { describe, expect, it } from 'vitest';

import {
  aggregate,
  evaluate,
  isGlobKey,
  matchesGlob,
  parseArgs,
  relativizeSummary,
} from './check-coverage-slack.mjs';

const ROOT = '/repo/packages/luxar-viewer';

/** Build a coverage-summary entry with the given covered/total per metric. */
function entry(covered, total) {
  const m = { covered, total, skipped: 0, pct: total ? (100 * covered) / total : 100 };
  return { lines: m, statements: m, functions: m, branches: m };
}

describe('matchesGlob', () => {
  it('matches nested files under a dir/** key', () => {
    expect(matchesGlob('src/ui/layers/layers-panel.ts', 'src/ui/**')).toBe(true);
  });

  it('does not match a sibling directory sharing a prefix', () => {
    expect(matchesGlob('src/uikit/thing.ts', 'src/ui/**')).toBe(false);
  });

  it('does not match a file outside the subtree', () => {
    expect(matchesGlob('src/data/loader.ts', 'src/ui/**')).toBe(false);
  });
});

describe('isGlobKey', () => {
  it('treats the four metric names as global, not globs', () => {
    for (const m of ['lines', 'statements', 'functions', 'branches']) {
      expect(isGlobKey(m)).toBe(false);
    }
  });

  it('treats a path key as a glob', () => {
    expect(isGlobKey('src/ui/**')).toBe(true);
  });
});

describe('aggregate', () => {
  it('sums absolute counts rather than averaging percentages', () => {
    // 1/1 and 0/99 average to 50% but aggregate to 1%. Averaging would let a
    // large badly-covered file hide behind a tiny perfect one.
    const { pct } = aggregate([entry(1, 1), entry(0, 99)], 'lines');
    expect(pct).toBeCloseTo(1, 5);
  });

  it('reports null for an empty group rather than a number', () => {
    expect(aggregate([], 'lines').pct).toBeNull();
  });
});

describe('relativizeSummary', () => {
  it('strips the viewer root and drops the total entry', () => {
    const files = relativizeSummary(
      { total: entry(1, 1), [`${ROOT}/src/a.ts`]: entry(1, 2) },
      ROOT
    );
    expect([...files.keys()]).toEqual(['src/a.ts']);
  });
});

describe('parseArgs', () => {
  it('accepts a finite max-slack override', () => {
    expect(parseArgs(['--max-slack', '1.5']).maxSlack).toBe(1.5);
  });

  it.each([
    ['missing', ['--max-slack']],
    ['non-numeric', ['--max-slack', 'nope']],
  ])('rejects a %s max-slack value', (_label, argv) => {
    expect(() => parseArgs(argv)).toThrow(/requires a finite number/);
  });
});

describe('evaluate', () => {
  const summary = {
    total: entry(90, 100),
    [`${ROOT}/src/ui/a.ts`]: entry(90, 100),
  };

  it('passes when every floor sits inside the slack budget', () => {
    const { failures } = evaluate(summary, { lines: 88 }, ROOT, 3);
    expect(failures).toEqual([]);
  });

  it('fails a floor that has decayed further than the budget', () => {
    const { failures } = evaluate(summary, { lines: 71 }, ROOT, 3);
    expect(failures).toHaveLength(1);
    expect(failures[0]).toMatch(/floor 71 is 19.0 pts under measured 90.00/);
  });

  it('accepts a floor exactly at the slack budget', () => {
    const { failures } = evaluate(summary, { lines: 87 }, ROOT, 3);
    expect(failures).toEqual([]);
  });

  it('fails a glob that matched zero files', () => {
    // The hole vitest leaves open: an empty group reports pct "Unknown", and
    // "Unknown" < 90 is false, so vitest passes it silently.
    const { failures } = evaluate(summary, { 'src/gone/**': { lines: 90 } }, ROOT, 3);
    expect(failures).toHaveLength(1);
    expect(failures[0]).toMatch(/matched 0 files/);
  });

  it('fails a glob metric that has zero countable items', () => {
    const zeroFunctions = {
      ...entry(90, 100),
      functions: { covered: 0, total: 0, skipped: 0, pct: 100 },
    };
    const noFunctionSummary = {
      total: zeroFunctions,
      [`${ROOT}/src/constants/a.ts`]: zeroFunctions,
    };
    const { failures, rows } = evaluate(
      noFunctionSummary,
      { 'src/constants/**': { functions: 90 } },
      ROOT,
      3
    );
    expect(rows).toHaveLength(0);
    expect(failures).toHaveLength(1);
    expect(failures[0]).toMatch(/functions has 0 countable items/);
  });

  it('reports a floor the measurement has fallen below', () => {
    const { failures } = evaluate(summary, { lines: 95 }, ROOT, 3);
    expect(failures.some((f) => /is BELOW floor 95/.test(f))).toBe(true);
  });

  it('scopes a glob floor to its own subtree', () => {
    const mixed = {
      total: entry(50, 100),
      [`${ROOT}/src/ui/a.ts`]: entry(99, 100),
      [`${ROOT}/src/rendering/b.ts`]: entry(1, 100),
    };
    const { rows } = evaluate(mixed, { 'src/ui/**': { lines: 97 } }, ROOT, 3);
    expect(rows).toHaveLength(1);
    expect(rows[0].measured).toBeCloseTo(99, 5);
  });
});
