import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  analyzeSpec,
  compareViolationsToExceptions,
  isDefaultConfigSpec,
  projectTestTimeout,
  specFiles,
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

  it('scales test.slow() by the enclosing describe budget', () => {
    const source = `
      import { test } from '@playwright/test';

      test.describe('slow group', () => {
        test.describe.configure({ timeout: 600_000 });

        test('triples the resolved slot timeout', async ({ page }) => {
          test.slow();
          await page.waitForTimeout(300_000);
        });
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([]);
  });

  it('triples a suite budget whatever the order of slow and configure', () => {
    // At suite scope nothing executes: the declared timeout always wins and the
    // static `slow` annotation triples it afterwards, across levels.
    const source = `
      import { test } from '@playwright/test';

      test.describe('configure first', () => {
        test.describe.configure({ timeout: 100_000 });
        test.slow();

        test('inherits 300 s', async ({ page }) => {
          await page.waitForTimeout(200_000);
        });
      });

      test.describe('slow first', () => {
        test.slow();
        test.describe.configure({ timeout: 100_000 });

        test('inherits 300 s either way', async ({ page }) => {
          await page.waitForTimeout(200_000);
        });
      });

      test.describe('slow outside', () => {
        test.slow();

        test.describe('configured inside', () => {
          test.describe.configure({ timeout: 100_000 });

          test('inherits 300 s across levels', async ({ page }) => {
            await page.waitForTimeout(200_000);
          });
        });
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([]);
  });

  it('walks a test body in statement order', () => {
    const source = `
      import { test } from '@playwright/test';

      test.describe('group', () => {
        test.describe.configure({ timeout: 300_000 });

        test('narrows before slowing down', async ({ page }) => {
          test.setTimeout(40_000);
          test.slow();
          await page.waitForTimeout(200_000);
        });

        test('narrows after slowing down', async ({ page }) => {
          test.slow();
          test.setTimeout(400_000);
          await page.waitForTimeout(200_000);
        });
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([
      {
        deadlineMs: 200_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 7,
        test: 'narrows before slowing down',
      },
    ]);
  });

  it('ignores a conditional test.slow()', () => {
    const source = `
      import { test } from '@playwright/test';

      test.describe('suite condition', () => {
        test.slow(process.platform === 'darwin');

        test('has no declared budget', async ({ page }) => {
          await page.waitForTimeout(120_000);
        });
      });

      test('condition in the body', async ({ page, isMobile }) => {
        test.slow(isMobile, 'slower on mobile');
        await page.waitForTimeout(120_000);
      });

      test('callback condition', async ({ page }) => {
        test.slow(() => true);
        await page.waitForTimeout(120_000);
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([
      {
        deadlineMs: 120_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 7,
        test: 'has no declared budget',
      },
      {
        deadlineMs: 120_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 12,
        test: 'condition in the body',
      },
      {
        deadlineMs: 120_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 17,
        test: 'callback condition',
      },
    ]);
  });

  it('applies test.slow() at most once per test', () => {
    const source = `
      import { test } from '@playwright/test';

      test.describe('slow group', () => {
        test.slow();

        test('is not slowed twice', async ({ page }) => {
          test.slow();
          await page.waitForTimeout(200_000);
        });
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([
      {
        deadlineMs: 200_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 7,
        test: 'is not slowed twice',
      },
    ]);
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

  it('lets an inner describe budget override the outer one', () => {
    const source = `
      import { test } from '@playwright/test';

      test.describe('outer', () => {
        test.describe.configure({ timeout: 120_000 });

        test.describe('inner', () => {
          test.describe.configure({ timeout: 300_000 });

          test('uses the inner budget', async ({ page }) => {
            await page.waitForTimeout(200_000);
          });
        });

        test('outruns the inherited outer budget', async ({ page }) => {
          await page.waitForTimeout(200_000);
        });
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([
      {
        deadlineMs: 200_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 15,
        test: 'outruns the inherited outer budget',
      },
    ]);
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

  it('does not count a named budget constant as its own deadline', () => {
    const source = `
      import { test } from '@playwright/test';

      const MESH_COMMIT_TIMEOUT_MS = 120_000;

      test('declares its budget through a constant', async ({ page }) => {
        test.setTimeout(MESH_COMMIT_TIMEOUT_MS);
        await page.waitForTimeout(45_000);
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([]);
  });

  it('still flags a named budget constant without headroom', () => {
    const source = `
      import { test } from '@playwright/test';

      const MESH_COMMIT_TIMEOUT_MS = 45_000;

      test('declares an undersized budget', async ({ page }) => {
        test.setTimeout(MESH_COMMIT_TIMEOUT_MS);
        await page.waitForTimeout(45_000);
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toHaveLength(1);
  });

  it('keeps an enclosing budget when a direct timeout cannot be resolved', () => {
    const source = `
      import { test } from '@playwright/test';

      const BACKENDS = ['webgl', 'webgpu'];

      test.describe('group', () => {
        test.describe.configure({ timeout: 300_000 });

        test('scales with the backend axis', async ({ page }) => {
          test.setTimeout(Math.max(300_000, 90_000 * BACKENDS.length));
          await page.waitForTimeout(90_000);
        });
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([]);
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

  it('includes long deadlines from file-level hooks in a test.only', () => {
    const source = `
      import { test } from '@playwright/test';

      test.beforeEach(async ({ page }) => {
        await page.waitForFunction(() => window.ready, undefined, { timeout: 45_000 });
      });

      test.only('inherits the file hook deadline', async () => {});
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([
      {
        deadlineMs: 45_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 8,
        test: 'inherits the file hook deadline',
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

  it('rejects two exceptions keyed to the same test', () => {
    expect(() =>
      compareViolationsToExceptions(
        [violation],
        [
          { ...violation, reason: 'Legacy helper is bounded elsewhere.' },
          { ...violation, reason: 'Duplicated by a bad merge.' },
        ]
      )
    ).toThrow(/Duplicate timeout-budget exception/);
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

describe('isDefaultConfigSpec', () => {
  it('excludes specs owned by the mobile and perf configs', () => {
    expect(isDefaultConfigSpec('src/tests/e2e/example.spec.ts')).toBe(true);
    expect(isDefaultConfigSpec('src/tests/e2e/mobile/touch.spec.ts')).toBe(false);
    expect(isDefaultConfigSpec('src/tests/e2e/rendering-perf-bench.spec.ts')).toBe(false);
  });
});

describe('specFiles', () => {
  it('keeps a checkout under a mobile ancestor directory in scope', () => {
    const fixture = mkdtempSync(join(tmpdir(), 'luxar-timeout-budgets-'));
    // The bug was matching the exclusions against the ABSOLUTE path, which a
    // checkout under /home/mobile/... drags its own ancestors through.
    const root = join(fixture, 'mobile', 'ci', 'pkg');
    const specRoot = join(root, 'src/tests/e2e');
    try {
      mkdirSync(join(specRoot, 'mobile'), { recursive: true });
      writeFileSync(join(specRoot, 'basic.spec.ts'), '');
      writeFileSync(join(specRoot, 'rendering-perf-bench.spec.ts'), '');
      writeFileSync(join(specRoot, 'mobile/touch.spec.ts'), '');

      expect(specFiles(specRoot, root).map((path) => relative(root, path))).toEqual([
        'src/tests/e2e/basic.spec.ts',
      ]);
    } finally {
      rmSync(fixture, { recursive: true, force: true });
    }
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
