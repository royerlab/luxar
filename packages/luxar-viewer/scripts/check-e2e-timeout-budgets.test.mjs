import { describe, expect, it } from 'vitest';

import {
  analyzeSpec,
  compareViolationsToExceptions,
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
          test: 'fixed test',
          reason: 'Legacy helper is bounded elsewhere.',
        },
      ],
    });
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
