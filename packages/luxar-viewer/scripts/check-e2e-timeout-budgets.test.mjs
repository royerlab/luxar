import { describe, expect, it } from 'vitest';

import {
  analyzeSpec,
  compareViolationsToExceptions,
  compareViolationCountsToBaseline,
  isDefaultConfigSpec,
  projectTestTimeout,
} from './check-e2e-timeout-budgets.mjs';

describe('analyzeSpec', () => {
  it('flags a long explicit wait without a test budget', () => {
    const source = `
      import { test } from '@playwright/test';

      test('loads eventually', async ({ page }) => {
        await page.waitForFunction(() => window.ready, undefined, { timeout: 45_000 });
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([
      {
        deadlineMs: 45_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 4,
        test: 'loads eventually',
      },
    ]);
  });

  it('accepts test and enclosing describe budgets', () => {
    const source = `
      import { test } from '@playwright/test';

      test.describe('slow group', () => {
        test.describe.configure({ timeout: 120_000 });

        test('inherits the budget', async ({ page }) => {
          await page.waitForTimeout(45_000);
        });
      });

      test('owns its budget', async ({ page }) => {
        test.slow();
        await page.waitForFunction(() => window.ready, undefined, { timeout: 60_000 });
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([]);
  });

  it('accepts a file-level describe budget', () => {
    const source = `
      import { test } from '@playwright/test';

      test.describe.configure({ timeout: 120_000 });

      test('inherits the file budget', async ({ page }) => {
        await page.waitForFunction(() => window.ready, undefined, { timeout: 60_000 });
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([]);
  });

  it('does not treat mode-only describe configuration as a budget', () => {
    const source = `
      import { test } from '@playwright/test';

      test.describe.configure({ mode: 'serial' });

      test('still needs a budget', async ({ page }) => {
        await page.waitForTimeout(45_000);
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toHaveLength(1);
  });

  it('rejects a declared budget that does not exceed the deadline', () => {
    const source = `
      import { test } from '@playwright/test';

      test('needs actual headroom', async ({ page }) => {
        test.setTimeout(45_000);
        await page.waitForTimeout(45_000);
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toHaveLength(1);
  });

  it('lets a direct test budget override an enclosing describe budget', () => {
    const source = `
      import { test } from '@playwright/test';

      test.describe('group', () => {
        test.describe.configure({ timeout: 120_000 });

        test('narrows its own budget', async ({ page }) => {
          test.setTimeout(45_000);
          await page.waitForTimeout(45_000);
        });
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toHaveLength(1);
  });

  it('accepts an unbounded test timeout', () => {
    const source = `
      import { test } from '@playwright/test';

      test('opts out of the timeout', async ({ page }) => {
        test.setTimeout(0);
        await page.waitForTimeout(90_000);
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([]);
  });

  it('resolves local helper defaults and module deadline constants', () => {
    const source = `
      import { test } from '@playwright/test';

      const SWEEP_DEADLINE_MS = 90_000;

      async function waitForCommitted(page, timeoutMs = 45_000) {
        await page.waitForFunction(() => window.committed, undefined, { timeout: timeoutMs });
      }

      test('uses helper default', async ({ page }) => {
        await waitForCommitted(page);
      });

      test('passes a deadline into the page', async ({ page }) => {
        await page.evaluate(({ deadlineMs }) => deadlineMs, {
          deadlineMs: SWEEP_DEADLINE_MS,
        });
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([
      {
        deadlineMs: 45_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 10,
        test: 'uses helper default',
      },
      {
        deadlineMs: 90_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 14,
        test: 'passes a deadline into the page',
      },
    ]);
  });

  it('ignores the half-budget boundary and computed short values', () => {
    const source = `
      import { test } from '@playwright/test';

      const SHORT_TIMEOUT_MS = 15_000 * 2;

      test('stays within half the project budget', async ({ page }) => {
        await page.waitForFunction(() => window.ready, undefined, {
          timeout: SHORT_TIMEOUT_MS,
        });
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([]);
  });

  it('uses the supplied project-derived threshold', () => {
    const source = `
      import { test } from '@playwright/test';

      test('crosses a smaller project threshold', async ({ page }) => {
        await page.waitForTimeout(25_000);
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts', 20_000)).toHaveLength(1);
  });

  it('reads explicit deadline arguments passed to shared wait helpers', () => {
    const source = `
      import { test } from '@playwright/test';
      import { waitForPointsLoaded } from './helpers';

      test('loads a large scene', async ({ page }) => {
        await waitForPointsLoaded(page, 100, 60_000);
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([
      {
        deadlineMs: 60_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 5,
        test: 'loads a large scene',
      },
    ]);
  });

  it('resolves aliased defaults from exported shared helpers', () => {
    const source = `
      import { test } from '@playwright/test';
      import { waitForLuxarReady as waitForReady } from './helpers';

      test('uses the imported default', async ({ page }) => {
        await waitForReady(page);
      });

      test('overrides the imported default', async ({ page }) => {
        await waitForReady(page, 20_000);
      });
    `;
    const helpers = `
      const DEFAULT_TIMEOUT = 45_000;
      export async function waitForLuxarReady(page, timeout = DEFAULT_TIMEOUT) {}
      async function privateWait(page, timeout = 90_000) {}
    `;

    expect(
      analyzeSpec(
        source,
        'src/tests/e2e/example.spec.ts',
        30_000,
        60_000,
        new Map([['./helpers', helpers]])
      )
    ).toEqual([
      {
        deadlineMs: 45_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 5,
        test: 'uses the imported default',
      },
    ]);
  });

  it('ignores unrelated and unresolved helper imports', () => {
    const source = `
      import { test } from '@playwright/test';
      import { waitForReady } from '../support';
      import { missingHelper } from './helpers/missing';

      test('has no modeled helper', async ({ page }) => {
        await waitForReady(page);
        await missingHelper(page);
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([]);
  });

  it('resolves helper submodule defaults', () => {
    const source = `
      import { test } from '@playwright/test';
      import { waitForScene } from './helpers/scene';

      test('uses a helper submodule', async ({ page }) => {
        await waitForScene(page);
      });
    `;
    const helpers = `
      export const waitForScene = async (page, deadlineMs = 45_000) => {};
    `;

    expect(
      analyzeSpec(
        source,
        'src/tests/e2e/example.spec.ts',
        30_000,
        60_000,
        new Map([['./helpers/scene', helpers]])
      )
    ).toHaveLength(1);
  });

  it('includes long deadlines from enclosing hooks', () => {
    const source = `
      import { test } from '@playwright/test';

      test.describe('group', () => {
        test.beforeEach(async ({ page }) => {
          await page.waitForFunction(() => window.ready, undefined, { timeout: 45_000 });
        });

        test('inherits the hook deadline', async () => {});
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([
      {
        deadlineMs: 45_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 9,
        test: 'inherits the hook deadline',
      },
    ]);
  });
});

describe('compareViolationsToExceptions', () => {
  const violation = {
    deadlineMs: 45_000,
    file: 'src/tests/e2e/example.spec.ts',
    line: 4,
    test: 'loads eventually',
  };

  it('requires a non-empty reason for every exception', () => {
    expect(() =>
      compareViolationsToExceptions(
        [violation],
        [{ file: violation.file, test: violation.test, reason: '   ' }]
      )
    ).toThrow(/non-empty reason/);
  });

  it('reports new violations and stale exceptions', () => {
    expect(
      compareViolationsToExceptions(
        [violation],
        [
          {
            file: 'src/tests/e2e/old.spec.ts',
            line: 8,
            test: 'fixed test',
            reason: 'Legacy helper is bounded elsewhere.',
          },
        ]
      )
    ).toEqual({
      newViolations: [violation],
      staleExceptions: [
        {
          file: 'src/tests/e2e/old.spec.ts',
          line: 8,
          test: 'fixed test',
          reason: 'Legacy helper is bounded elsewhere.',
        },
      ],
    });
  });

  it('keys dynamic-title exceptions by source line', () => {
    const dynamicViolation = { ...violation, test: '<dynamic title>' };
    const secondViolation = { ...dynamicViolation, line: 8 };

    expect(
      compareViolationsToExceptions(
        [dynamicViolation, secondViolation],
        [
          {
            file: dynamicViolation.file,
            line: dynamicViolation.line,
            test: dynamicViolation.test,
            reason: 'Legacy wait.',
          },
        ]
      )
    ).toEqual({ newViolations: [secondViolation], staleExceptions: [] });
  });
});

describe('compareViolationCountsToBaseline', () => {
  const violations = [
    { deadlineMs: 45_000, file: 'a.spec.ts', line: 1, test: 'one' },
    { deadlineMs: 45_000, file: 'a.spec.ts', line: 2, test: 'two' },
  ];

  it('reports count regressions and stale over-declarations', () => {
    expect(
      compareViolationCountsToBaseline(violations, { 'a.spec.ts': 1, 'old.spec.ts': 2 })
    ).toEqual({
      regressions: [{ allowed: 1, file: 'a.spec.ts', found: 2 }],
      improvements: [{ allowed: 2, file: 'old.spec.ts', found: 0 }],
    });
  });

  it('accepts exact counts and rejects invalid counts', () => {
    expect(compareViolationCountsToBaseline(violations, { 'a.spec.ts': 2 })).toEqual({
      regressions: [],
      improvements: [],
    });
    expect(() => compareViolationCountsToBaseline(violations, { 'a.spec.ts': -1 })).toThrow(
      /non-negative integer/
    );
  });
});

describe('isDefaultConfigSpec', () => {
  it('excludes specs owned by the mobile and perf configs', () => {
    expect(isDefaultConfigSpec('src/tests/e2e/example.spec.ts')).toBe(true);
    expect(isDefaultConfigSpec('src/tests/e2e/mobile/touch.spec.ts')).toBe(false);
    expect(isDefaultConfigSpec('src/tests/e2e/rendering-perf-bench.spec.ts')).toBe(false);
  });
});

describe('projectTestTimeout', () => {
  it('reads the test timeout without confusing web-server or expect timeouts', () => {
    const source = `
      const TEST_TIMEOUT = 60_000;

      export default defineConfig({
        webServer: { timeout: 15_000 },
        timeout: TEST_TIMEOUT,
        expect: { timeout: 10_000 },
      });
    `;

    expect(projectTestTimeout(source)).toBe(60_000);
  });
});
