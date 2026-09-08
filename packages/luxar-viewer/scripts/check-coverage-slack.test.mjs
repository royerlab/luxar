import { describe, expect, it } from 'vitest';

import { COVERAGE_RECORDED, COVERAGE_THRESHOLDS } from '../coverage-thresholds.mjs';

import {
  aggregate,
  emitWarnings,
  evaluate,
  formatRecordedBaselines,
  isGlobKey,
  matchesGlob,
  parseArgs,
  relativizeSummary,
  validateRecorded,
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

  it('enables paste-ready baseline output', () => {
    expect(parseArgs(['--print']).print).toBe(true);
  });
});

describe('evaluate', () => {
  const summary = {
    total: entry(90, 100),
    [`${ROOT}/src/ui/a.ts`]: entry(90, 100),
  };

  it('passes when every floor sits inside the slack budget', () => {
    const { failures } = evaluate(
      summary,
      { lines: 88 },
      { lines: 90 },
      {
        viewerRoot: ROOT,
        maxSlack: 3,
        maxErosion: 1,
      }
    );
    expect(failures).toEqual([]);
  });

  it('fails a floor that has decayed further than the budget', () => {
    const { failures } = evaluate(
      summary,
      { lines: 71 },
      { lines: 90 },
      {
        viewerRoot: ROOT,
        maxSlack: 3,
        maxErosion: 1,
      }
    );
    expect(failures).toHaveLength(1);
    expect(failures[0]).toMatch(/floor 71 is 19.0 pts under measured 90.00/);
    expect(failures[0]).toMatch(/set COVERAGE_RECORDED\.lines to 90\.$/);
  });

  it('accepts a floor exactly at the slack budget', () => {
    const { failures } = evaluate(
      summary,
      { lines: 87 },
      { lines: 90 },
      {
        viewerRoot: ROOT,
        maxSlack: 3,
        maxErosion: 1,
      }
    );
    expect(failures).toEqual([]);
  });

  it('fails a glob that matched zero files', () => {
    // The hole vitest leaves open: an empty group reports pct "Unknown", and
    // "Unknown" < 90 is false, so vitest passes it silently.
    const { failures } = evaluate(
      summary,
      { 'src/gone/**': { lines: 90 } },
      { 'src/gone/**': { lines: 90 } },
      { viewerRoot: ROOT, maxSlack: 3, maxErosion: 1 }
    );
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
      { 'src/constants/**': { functions: 90 } },
      { viewerRoot: ROOT, maxSlack: 3, maxErosion: 1 }
    );
    expect(rows).toHaveLength(0);
    expect(failures).toHaveLength(1);
    expect(failures[0]).toMatch(/functions has 0 countable items/);
  });

  it('reports a floor the measurement has fallen below', () => {
    const { failures } = evaluate(
      summary,
      { lines: 95 },
      { lines: 90 },
      {
        viewerRoot: ROOT,
        maxSlack: 3,
        maxErosion: 1,
      }
    );
    expect(failures.some((f) => /is BELOW floor 95/.test(f))).toBe(true);
  });

  it('scopes a glob floor to its own subtree', () => {
    const mixed = {
      total: entry(50, 100),
      [`${ROOT}/src/ui/a.ts`]: entry(99, 100),
      [`${ROOT}/src/rendering/b.ts`]: entry(1, 100),
    };
    const { rows } = evaluate(
      mixed,
      { 'src/ui/**': { lines: 97 } },
      { 'src/ui/**': { lines: 99 } },
      { viewerRoot: ROOT, maxSlack: 3, maxErosion: 1 }
    );
    expect(rows).toHaveLength(1);
    expect(rows[0].measured).toBeCloseTo(99, 5);
  });

  it('warns when measurement erodes beyond the recorded budget', () => {
    const { failures, warnings } = evaluate(
      summary,
      { lines: 88 },
      { lines: 91.01 },
      {
        viewerRoot: ROOT,
        maxSlack: 3,
        maxErosion: 1,
      }
    );
    expect(failures).toEqual([]);
    expect(warnings).toEqual([
      'lines: measured 90.00 is 1.01 pts below recorded 91.01 (budget 1).',
    ]);
  });

  it('warns when the recorded baseline is stale-low', () => {
    const { warnings } = evaluate(
      summary,
      { lines: 88 },
      { lines: 88.99 },
      {
        viewerRoot: ROOT,
        maxSlack: 3,
        maxErosion: 1,
      }
    );
    expect(warnings).toEqual([
      'lines: measured 90.00 is 1.01 pts above recorded 88.99 (budget 1); refresh the baseline.',
    ]);
  });

  it('does not warn at exactly the erosion budget', () => {
    const { warnings } = evaluate(
      summary,
      { lines: 88 },
      { lines: 91 },
      {
        viewerRoot: ROOT,
        maxSlack: 3,
        maxErosion: 1,
      }
    );
    expect(warnings).toEqual([]);
  });
});

describe('formatRecordedBaselines', () => {
  it('prints a paste-ready block in threshold order', () => {
    expect(
      formatRecordedBaselines([
        { key: 'lines', metric: 'lines', measured: 90 },
        { key: 'src/ui/**', metric: 'lines', measured: 91.234 },
        { key: 'src/ui/**', metric: 'branches', measured: 79.1 },
      ])
    ).toBe(`/** Last accepted coverage measurements for each floor. */
export const COVERAGE_RECORDED = {
  lines: 90,
  'src/ui/**': { lines: 91.23, branches: 79.1 },
};`);
  });
});

describe('emitWarnings', () => {
  it('adds escaped GitHub annotations while retaining stderr warnings', () => {
    const output = [];
    emitWarnings(['lines: down 1%\nrefresh'], {
      githubActions: true,
      warn: (message) => output.push(message),
    });
    expect(output).toEqual([
      'Coverage erosion warnings:\n  - lines: down 1%\nrefresh',
      '::warning title=Coverage erosion::lines: down 1%25%0Arefresh',
    ]);
  });
});

describe('validateRecorded', () => {
  it('keeps the configured threshold and recorded maps in lockstep', () => {
    expect(validateRecorded(COVERAGE_THRESHOLDS, COVERAGE_RECORDED)).toEqual([]);
  });

  it('accepts matching top-level and metric keys', () => {
    expect(
      validateRecorded(
        { lines: 88, 'src/ui/**': { lines: 89, branches: 77 } },
        { lines: 89.03, 'src/ui/**': { lines: 91.53, branches: 79.11 } }
      )
    ).toEqual([]);
  });

  it('fails closed when a row or metric baseline is missing', () => {
    const failures = validateRecorded(
      { lines: 88, 'src/ui/**': { lines: 89, branches: 77 } },
      { 'src/ui/**': { lines: 91.53 } }
    );
    expect(failures).toEqual([
      'coverage recorded baselines must have exactly the threshold keys: missing lines',
      'src/ui/** recorded baselines must have exactly the threshold metrics: missing branches',
    ]);
  });

  it('fails closed when a recorded baseline has the wrong shape', () => {
    expect(
      validateRecorded(
        { lines: 88, 'src/ui/**': { lines: 89 } },
        { lines: Number.NaN, 'src/ui/**': 91.53 }
      )
    ).toEqual([
      'lines recorded baseline must be a finite number',
      'src/ui/** recorded baseline must be a metric map',
    ]);
  });
});
