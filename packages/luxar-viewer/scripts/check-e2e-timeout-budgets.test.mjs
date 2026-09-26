import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  analyzeSpec,
  compareViolationsToExceptions,
  compareViolationCountsToBaseline,
  helperSourcesForSpec,
  isDefaultConfigSpec,
  parseArgs,
  projectTestTimeout,
  saveBaseline,
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

    expect(
      analyzeSpec(
        source,
        'src/tests/e2e/example.spec.ts',
        30_000,
        60_000,
        new Map([
          [
            './helpers',
            'export async function waitForPointsLoaded(page, count, timeout = 45_000) {}',
          ],
        ])
      )
    ).toEqual([
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

  it('resolves helpers exposed through a local export list', () => {
    const source = `
      import { test } from '@playwright/test';
      import { waitForReady } from './helpers';

      test('uses the exported alias', async ({ page }) => {
        await waitForReady(page);
      });
    `;
    const helpers = `
      async function wait(page, timeout = 45_000) {}
      export { wait as waitForReady };
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
        test: 'uses the exported alias',
      },
    ]);
  });

  it('ignores unrelated imports and unknown named helper exports', () => {
    const source = `
      import { test } from '@playwright/test';
      import { waitForReady } from '../support';
      import { missingHelper } from './helpers/missing';

      test('has no modelled helper', async ({ page }) => {
        await waitForReady(page);
        await missingHelper(page);
      });
    `;

    expect(
      analyzeSpec(
        source,
        'src/tests/e2e/example.spec.ts',
        30_000,
        60_000,
        new Map([['./helpers/missing', 'export const unrelated = 1;']])
      )
    ).toEqual([]);
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

  it('propagates hardcoded deadlines from local helper bodies', () => {
    const source = `
      import { test } from '@playwright/test';

      async function loadScene(page) {
        await page.waitForFunction(() => window.ready, undefined, { timeout: 45_000 });
      }

      test('loads through a local helper', async ({ page }) => {
        await loadScene(page);
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([
      {
        deadlineMs: 45_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 8,
        test: 'loads through a local helper',
      },
    ]);
  });

  it('propagates imported helper defaults through a local helper body', () => {
    const source = `
      import { test } from '@playwright/test';
      import { waitForLuxarReady, waitForPointsLoaded } from './helpers';

      async function loadScene(page) {
        await waitForLuxarReady(page);
        await waitForPointsLoaded(page);
      }

      test('loads through imported helpers', async ({ page }) => {
        await loadScene(page);
      });
    `;
    const helpers = `
      export async function waitForLuxarReady(page, timeout = 45_000) {}
      export async function waitForPointsLoaded(page, count = 1, timeout = 45_000) {}
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
        line: 10,
        test: 'loads through imported helpers',
      },
    ]);
  });

  it('propagates deadlines from imported helper bodies', () => {
    const source = `
      import { test } from '@playwright/test';
      import { openPanel } from './helpers';

      test('opens through an imported wrapper', async ({ page }) => {
        await openPanel(page);
      });
    `;
    const helpers = `
      async function waitForPanel(page) {
        await page.waitForFunction(() => window.panelOpen, undefined, { timeout: 45_000 });
      }

      export async function openPanel(page) {
        await waitForPanel(page);
      }
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
        test: 'opens through an imported wrapper',
      },
    ]);
  });

  it('propagates caller timeout arguments through nested helper bodies', () => {
    const source = `
      import { test } from '@playwright/test';

      async function waitForScene(page, timeout = 45_000) {
        await page.waitForFunction(() => window.ready, undefined, { timeout });
      }

      async function prepareScene(page, deadline = 45_000) {
        await waitForScene(page, deadline);
      }

      test('overrides the nested helper deadline', async ({ page }) => {
        await prepareScene(page, 75_000);
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([
      {
        deadlineMs: 75_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 12,
        test: 'overrides the nested helper deadline',
      },
    ]);
  });

  it('guards mutually recursive helper bodies', () => {
    const source = `
      import { test } from '@playwright/test';

      async function loadScene(page) {
        await retryLoad(page);
      }

      async function retryLoad(page) {
        await loadScene(page);
        await page.waitForTimeout(45_000);
      }

      test('loads through recursive helpers', async ({ page }) => {
        await loadScene(page);
      });
    `;

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toEqual([
      {
        deadlineMs: 45_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 13,
        test: 'loads through recursive helpers',
      },
    ]);
  });

  it('fails closed on deeply nested helper-body traversal', () => {
    const helperChain = Array.from({ length: 20 }, (_, index) => {
      const next = index === 19 ? 'page.waitForTimeout(45_000)' : `helper${index + 1}(page)`;
      return `async function helper${index}(page) { await ${next}; }`;
    }).join('\n');
    const source = `
      import { test } from '@playwright/test';
      ${helperChain}

      test('does not traverse an unbounded helper chain', async ({ page }) => {
        await helper0(page);
      });
    `;

    expect(() => analyzeSpec(source, 'src/tests/e2e/example.spec.ts')).toThrow(
      /src\/tests\/e2e\/example.spec.ts: Helper body traversal exceeded 10 nested calls/
    );
  });

  it('handles a wide helper graph without spreading every call path', () => {
    const levels = Array.from({ length: 10 }, (_, depth) =>
      Array.from({ length: 4 }, (_, index) =>
        depth === 9
          ? `async function h${depth}_${index}(page) { await page.waitForTimeout(${index === 3 ? '75_000' : '45_000'}); }`
          : `async function h${depth}_${index}(page) { ${Array.from({ length: 4 }, (_, child) => `await h${depth + 1}_${child}(page);`).join(' ')} }`
      ).join('\n')
    ).join('\n');
    const source = `${levels}\ntest('wide graph', async ({ page }) => { await h0_0(page); });`;

    expect(analyzeSpec(source, 'src/tests/e2e/wide.spec.ts')).toEqual([
      { deadlineMs: 75_000, file: 'src/tests/e2e/wide.spec.ts', line: 41, test: 'wide graph' },
    ]);
  });

  it('follows a helper imported by another helper module', () => {
    const source = `import { outer } from './helpers';
test('nested import', async ({ page }) => { await outer(page); });`;
    const helpers = new Map([
      [
        './helpers',
        `import { inner as renamed } from './helpers/inner';
export async function outer(page) { await renamed(page); }`,
      ],
      [
        './helpers/inner',
        'export async function inner(page) { await page.waitForTimeout(45_000); }',
      ],
    ]);

    expect(analyzeSpec(source, 'src/tests/e2e/example.spec.ts', 30_000, 60_000, helpers)).toEqual([
      { deadlineMs: 45_000, file: 'src/tests/e2e/example.spec.ts', line: 2, test: 'nested import' },
    ]);
  });

  it('fails closed on unsupported shared-helper import forms', () => {
    const helpers = new Map([
      ['./helpers', 'export async function wait(page, timeout = 45_000) {}'],
    ]);

    expect(() =>
      analyzeSpec(
        "import * as helpers from './helpers';",
        'src/tests/e2e/example.spec.ts',
        30_000,
        60_000,
        helpers
      )
    ).toThrow(/namespace import/);
    expect(() =>
      analyzeSpec(
        "import helpers from './helpers';",
        'src/tests/e2e/example.spec.ts',
        30_000,
        60_000,
        helpers
      )
    ).toThrow(/default import/);
    expect(() =>
      analyzeSpec("import { wait } from './helpers';", 'src/tests/e2e/example.spec.ts')
    ).toThrow(/Could not resolve shared helper module/);
  });

  it('fails closed on shared-helper re-exports', () => {
    const source = "import { wait } from './helpers';";

    expect(() =>
      analyzeSpec(
        source,
        'src/tests/e2e/example.spec.ts',
        30_000,
        60_000,
        new Map([['./helpers', "export * from './waits';"]])
      )
    ).toThrow(/re-export/);
    expect(() =>
      analyzeSpec(
        source,
        'src/tests/e2e/example.spec.ts',
        30_000,
        60_000,
        new Map([['./helpers', "export { wait } from './waits';"]])
      )
    ).toThrow(/re-export/);
  });

  it('ignores type-only shared-helper imports and re-exports', () => {
    const runtimeImport = `
      import { test } from '@playwright/test';
      import { wait } from './helpers';

      test('uses the runtime helper', async ({ page }) => {
        await wait(page);
      });
    `;
    const helperWithTypeReExport = `
      export type { HelperOptions } from './types';
      export async function wait(page, timeout = 45_000) {}
    `;

    expect(
      analyzeSpec(
        runtimeImport,
        'src/tests/e2e/example.spec.ts',
        30_000,
        60_000,
        new Map([['./helpers', helperWithTypeReExport]])
      )
    ).toEqual([
      {
        deadlineMs: 45_000,
        file: 'src/tests/e2e/example.spec.ts',
        line: 5,
        test: 'uses the runtime helper',
      },
    ]);
    expect(
      analyzeSpec("import type Helpers from './helpers';", 'src/tests/e2e/example.spec.ts')
    ).toEqual([]);
    expect(
      analyzeSpec("import type * as Helpers from './helpers';", 'src/tests/e2e/example.spec.ts')
    ).toEqual([]);
    expect(
      analyzeSpec("import { type HelperOptions } from './helpers';", 'example.spec.ts')
    ).toEqual([]);
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

describe('helperSourcesForSpec', () => {
  it('loads relative imports inside helper modules for analysis', () => {
    const fixture = mkdtempSync(join(tmpdir(), 'luxar-timeout-helpers-'));
    try {
      const specPath = join(fixture, 'example.spec.ts');
      const source =
        "import { outer } from './helpers';\ntest('nested', async ({ page }) => { await outer(page); });";
      writeFileSync(specPath, source);
      writeFileSync(
        join(fixture, 'helpers.ts'),
        "import { inner } from './inner'; export async function outer(page) { await inner(page); }"
      );
      writeFileSync(
        join(fixture, 'inner.ts'),
        'export async function inner(page) { await page.waitForTimeout(45_000); }'
      );

      const sources = helperSourcesForSpec(source, specPath);
      expect([...sources.keys()]).toEqual(['./helpers', join(fixture, 'inner.ts')]);
      expect(analyzeSpec(source, 'example.spec.ts', 30_000, 60_000, sources)).toEqual([
        { deadlineMs: 45_000, file: 'example.spec.ts', line: 2, test: 'nested' },
      ]);
    } finally {
      rmSync(fixture, { recursive: true, force: true });
    }
  });

  it('keeps nested imports distinct when a helper resolves through index.ts', () => {
    const fixture = mkdtempSync(join(tmpdir(), 'luxar-timeout-helpers-'));
    try {
      const specPath = join(fixture, 'example.spec.ts');
      const source =
        "import { a } from './helpers';\nimport { b } from './helpers/other';\ntest('nested', async ({ page }) => { await a(page); await b(page); });";
      mkdirSync(join(fixture, 'helpers'));
      writeFileSync(specPath, source);
      writeFileSync(
        join(fixture, 'helpers/index.ts'),
        "import { w } from './wait'; export async function a(page) { await w(page); }"
      );
      writeFileSync(
        join(fixture, 'helpers/other.ts'),
        "import { w } from '../wait'; export async function b(page) { await w(page); }"
      );
      writeFileSync(
        join(fixture, 'helpers/wait.ts'),
        'export async function w(page) { await page.waitForTimeout(1_000); }'
      );
      writeFileSync(
        join(fixture, 'wait.ts'),
        'export async function w(page) { await page.waitForTimeout(45_000); }'
      );

      const sources = helperSourcesForSpec(source, specPath);
      expect(analyzeSpec(source, 'example.spec.ts', 30_000, 60_000, sources)).toEqual([
        { deadlineMs: 45_000, file: 'example.spec.ts', line: 3, test: 'nested' },
      ]);
    } finally {
      rmSync(fixture, { recursive: true, force: true });
    }
  });

  it('resolves helper modules through each supported candidate', () => {
    const fixture = mkdtempSync(join(tmpdir(), 'luxar-timeout-helpers-'));
    try {
      const cases = [
        { importPath: './helpers/file', modulePath: 'helpers/file.ts' },
        { importPath: './helpers/directory', modulePath: 'helpers/directory/index.ts' },
        { importPath: './helpers/exact.ts', modulePath: 'helpers/exact.ts' },
      ];
      for (const { modulePath } of cases) {
        const path = join(fixture, modulePath);
        mkdirSync(join(path, '..'), { recursive: true });
        writeFileSync(path, 'export const wait = async (page, timeout = 45_000) => {};');
      }
      const specPath = join(fixture, 'example.spec.ts');
      const source = cases
        .map(({ importPath }) => `import { wait } from '${importPath}';`)
        .join('\n');
      const cache = new Map();
      const first = helperSourcesForSpec(source, specPath, cache);
      const second = helperSourcesForSpec(source, specPath, cache);

      expect([...first.keys()]).toEqual(cases.map(({ importPath }) => importPath));
      for (const { importPath } of cases)
        expect(second.get(importPath)).toBe(first.get(importPath));
    } finally {
      rmSync(fixture, { recursive: true, force: true });
    }
  });

  it('fails closed when a shared-helper module cannot be resolved', () => {
    const fixture = mkdtempSync(join(tmpdir(), 'luxar-timeout-helpers-'));
    try {
      expect(() =>
        helperSourcesForSpec(
          "import { wait } from './helpers/missing';",
          join(fixture, 'example.spec.ts')
        )
      ).toThrow(/Could not resolve shared helper module/);
    } finally {
      rmSync(fixture, { recursive: true, force: true });
    }
  });
});

describe('baseline updates', () => {
  it('parses only the supported update flag', () => {
    expect(parseArgs([])).toEqual({ updateBaseline: false });
    expect(parseArgs(['--update-baseline'])).toEqual({ updateBaseline: true });
    expect(() => parseArgs(['--unknown'])).toThrow(/Unknown argument/);
  });

  it('reports command errors without a stack trace', () => {
    const result = spawnSync(
      process.execPath,
      [join(import.meta.dirname, 'check-e2e-timeout-budgets.mjs'), '--unknown'],
      { encoding: 'utf8' }
    );

    expect(result.status).toBe(1);
    expect(result.stdout).toBe('');
    expect(result.stderr).toBe('Unknown argument: --unknown\n');
  });

  it('rewrites sorted counts while preserving baseline metadata', () => {
    const fixture = mkdtempSync(join(tmpdir(), 'luxar-timeout-baseline-'));
    const baselinePath = join(fixture, 'baseline.json');
    try {
      writeFileSync(baselinePath, '{"note":"keep me","files":{"old.spec.ts":2}}\n');
      saveBaseline(
        [{ file: 'z.spec.ts' }, { file: 'a.spec.ts' }, { file: 'z.spec.ts' }],
        baselinePath
      );

      expect(JSON.parse(readFileSync(baselinePath, 'utf8'))).toEqual({
        note: 'keep me',
        files: { 'a.spec.ts': 1, 'z.spec.ts': 2 },
      });
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
