/**
 * Global Setup for Playwright E2E Tests
 *
 * This file runs BEFORE any tests and verifies pre-conditions:
 * 1. Servers can be reached, and belong to THIS checkout
 * 2. Example dataset freshness and required dataset presence (warn)
 * 3. Generated zarr fixtures exist, are complete, are current, and are served (throw — 19 specs
 *    hard-depend on them)
 * 4. Basic environment checks
 *
 * It also stamps the run's parallelism. That belongs here rather than in `playwright.config.ts`
 * because Playwright applies `--workers=N` / `--debug` AFTER the config module is
 * evaluated, but hands this hook the resolved `FullConfig`. (Not reached by `--list`, which runs
 * no global setup — so listing tests stays silent.) `config.workers` is the run's CEILING, not
 * the concurrency it reaches: Playwright narrows it to `min(workers, maxConcurrentTestGroups)`
 * after this hook, so a one-file run can be stamped `max 3` and then report "using 1 worker".
 */

import type { FullConfig } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import {
  assertCheckoutServerIdentity,
  assertHTTPResource,
  requireE2EServerMetadata,
} from '../../../tools/e2e-server-identity';
import {
  FIXTURES_REPO_RELATIVE_PATH,
  isGeneratedFixtureComplete,
  parseGeneratedFixtureNames,
} from '../../../tools/fixture-manifest';
import {
  checkExampleFixtureFreshness,
  exposeExampleFixtureFreshnessToWorkers,
  reportExampleFixtureFreshness,
  type ExampleFixtureFreshness,
} from '../../../tools/example-fixture-freshness';
import { e2eWorkerPlan, formatE2EParallelismStamp } from '../../../tools/e2e-workers';
import { areFixturesStale } from '../../../tools/fixture-freshness';

// `package.json` declares `"type": "module"`, so the CommonJS `__dirname`
// global is undefined at module load. Reconstruct it from `import.meta.url`.
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Required datasets for E2E tests
const REQUIRED_DATASETS = [
  'simple_nd_example.luxar.zarr',
  'build_example_manual.luxar.zarr',
  'build_example_structured.luxar.zarr',
  'dimension_navigation_example.luxar.zarr',
  'dimension_sliders_5d_example.luxar.zarr',
  'dense_grid_5d_example.luxar.zarr',
  'layers_test_example.luxar.zarr',
  // slice-cache.spec.ts: 4D progressive points (additive LODs + hidden
  // discrete frame dim) — the S-cache's target configuration.
  'progressive_timelapse_example.luxar.zarr',
  // Read by frame-pacing.spec.ts (the #1724 regression) and by the un-parked
  // case in all-examples-smoke-test.spec.ts. Listed so a run without it
  // reports the name up front (and so its reachability is probed) — like
  // every example above, a missing one WARNS rather than throwing; the throw
  // path is check 3's generated fixtures.
  'performance_benchmark_example.luxar.zarr',
];

/**
 * Fail with an actionable message when generated zarr fixtures are absent, stale, or unserved.
 *
 * Three distinct failure modes, checked separately because their fixes differ:
 *
 *  1. **Never generated, or half generated** — a clean checkout on which vitest has never
 *     run, or a generator killed mid-write. Both are reported by name and both are fixed by
 *     rerunning the generator, so they are one check; `isGeneratedFixtureComplete` is what
 *     keeps an interrupted run's stump directory from passing as a fixture.
 *  2. **Generated but stale** — a generator or local Python source reachable from its imports
 *     changed after the fixtures were written. Regenerating refreshes the stores and stamp.
 *  3. **Generated but unreachable** — the data server is rooted somewhere other than the
 *     repository, so the specs' `/packages/luxar-viewer/tests/fixtures/...` URLs 404 even
 *     though the files exist. One HTTP probe settles this; probing all ~47 would add 47
 *     round-trips to every run to re-answer the same question about the serving root.
 *
 * **Skipped for the suites that read no fixtures**, which is the whole point of
 * `LUXAR_E2E_NO_FIXTURES`:
 *
 *  - smoke, an explicit five-file allowlist chosen so that none of them reads
 *    `tests/fixtures/`, whose CI job generates datasets at runtime via `make run-examples`
 *    and pulls no Git LFS;
 *  - the perf benchmarks, which share this global setup through
 *    `playwright.perf.config.ts` but match only `*perf-bench.spec.ts` — none of which
 *    reads `tests/fixtures/` either.
 *
 * Requiring fixtures in either would abort a suite deliberately built not to need them.
 *
 * A flag set by the invoking `package.json` script, rather than a spec-list inspection:
 * Playwright's `FullConfig` does not expose which files the CLI filter selected, and
 * re-deriving the allowlist here would put a second copy of it one edit away from
 * disagreeing with `package.json`. The flag lives on the same line as the file list (or
 * the `--config`), so they move together.
 */
async function assertGeneratedFixtures(projectRoot: string, dataBaseURL: string): Promise<void> {
  if (process.env.LUXAR_E2E_NO_FIXTURES === '1') {
    console.log('⏭️  Skipping the generated-fixture check (LUXAR_E2E_NO_FIXTURES=1)');
    return;
  }
  const fixturesDir = path.join(projectRoot, FIXTURES_REPO_RELATIVE_PATH);
  const expected = parseGeneratedFixtureNames(path.join(fixturesDir, 'generate_test_data.py'));
  const missing = expected.filter(
    (name) => !isGeneratedFixtureComplete(path.join(fixturesDir, name))
  );

  if (missing.length > 0) {
    const shown = missing.slice(0, 10).map((name) => `   - ${name}`);
    const elided =
      missing.length > shown.length ? `   … and ${missing.length - shown.length} more` : '';
    throw new Error(
      `${missing.length}/${expected.length} generated zarr fixtures are missing or ` +
        `incomplete in ${fixturesDir}:\n` +
        [...shown, elided].filter(Boolean).join('\n') +
        '\n\nGenerate them with:\n' +
        '  pnpm test:generate-fixtures\n' +
        '(`pnpm test` does this automatically; Playwright deliberately does not, because the ' +
        'generator takes 1-2 minutes.)'
    );
  }

  if (areFixturesStale(projectRoot, fixturesDir)) {
    throw new Error(
      `Generated zarr fixtures are stale in ${fixturesDir}.\n\nRegenerate them with:\n` +
        '  pnpm test:generate-fixtures\n' +
        '(Playwright checks freshness but deliberately does not run the 1-2 minute generator.)'
    );
  }

  // The serving-root probe. `assertHTTPResource` names the URL, which is the diagnostic:
  // seeing the full `/packages/luxar-viewer/tests/fixtures/...` path 404 is what tells you
  // the data server is rooted at the wrong directory.
  const probeURL = new URL(
    `/${FIXTURES_REPO_RELATIVE_PATH}/${encodeURIComponent(expected[0])}/`,
    dataBaseURL
  ).toString();
  await assertHTTPResource(`Generated fixture ${expected[0]}`, probeURL);
  console.log(`✅ ${expected.length} generated zarr fixtures current and served`);
}

export default async function globalSetup(config: FullConfig) {
  console.log('\n🔍 Running pre-flight checks...\n');

  // `console.error`, not `console.log`: stdout is Playwright's machine-readable channel
  // (`--reporter=json`, `--list`), so diagnostics belong on stderr. `config.workers` is the only
  // count available this early, and it is a ceiling — the stamp labels it as one.
  console.error(formatE2EParallelismStamp(config.workers, e2eWorkerPlan()));

  const serverMetadata = requireE2EServerMetadata(config.metadata);
  await assertCheckoutServerIdentity(
    'Viewer',
    serverMetadata.viewerIdentityURL,
    serverMetadata.checkout
  );
  await assertCheckoutServerIdentity(
    'Dataset',
    serverMetadata.dataIdentityURL,
    serverMetadata.checkout
  );
  console.log(`✅ Viewer and dataset servers match: ${serverMetadata.checkout.projectRoot}`);

  const projectRoot = serverMetadata.checkout.projectRoot;
  const examplesDir = path.join(projectRoot, 'datasets/examples');
  const datasetWarnings: string[] = [];

  // Check 1: Verify examples directory exists
  if (!fs.existsSync(examplesDir)) {
    datasetWarnings.push('examples directory missing');
    // `missing`, not `unavailable`: this is a certainty with an exact remedy,
    // and — unlike the old boolean — it now reaches the workers, so a spec that
    // dies on a readiness timeout says why instead of just how long it waited.
    const absent: ExampleFixtureFreshness = { status: 'missing', detail: examplesDir };
    exposeExampleFixtureFreshnessToWorkers(absent);
    reportExampleFixtureFreshness(absent);
    // Don't throw - allow tests that don't need examples to run
    // (e.g., basic-rendering, viewer-initialization, test-fixtures, geometry-types)
  } else {
    console.log(`✅ Examples directory found: ${examplesDir}`);

    const freshness = checkExampleFixtureFreshness(projectRoot);
    exposeExampleFixtureFreshnessToWorkers(freshness);
    const examplesWarned = reportExampleFixtureFreshness(freshness);
    if (examplesWarned) {
      datasetWarnings.push(
        freshness.status === 'stale' ? 'example datasets stale' : 'example freshness unavailable'
      );
    }

    // Check 2: Verify required datasets exist locally and through the HTTP server.
    const missingDatasets: string[] = [];
    const foundDatasets: string[] = [];

    for (const dataset of REQUIRED_DATASETS) {
      const datasetPath = path.join(examplesDir, dataset);
      if (fs.existsSync(datasetPath)) {
        foundDatasets.push(dataset);
      } else {
        missingDatasets.push(dataset);
      }
    }

    console.log(`✅ Found ${foundDatasets.length}/${REQUIRED_DATASETS.length} required datasets`);

    for (const dataset of foundDatasets) {
      const datasetURL = new URL(
        `/datasets/examples/${encodeURIComponent(dataset)}/`,
        serverMetadata.dataBaseURL
      ).toString();
      await assertHTTPResource(`Required dataset ${dataset}`, datasetURL);
    }
    console.log(`✅ ${foundDatasets.length} required datasets are reachable over HTTP`);

    if (missingDatasets.length > 0) {
      datasetWarnings.push(`${missingDatasets.length} required example datasets missing`);
      console.warn('\n⚠️  Warning: Some datasets are missing:');
      for (const dataset of missingDatasets) {
        console.warn(`   - ${dataset}`);
      }
      console.warn('\n   Tests requiring these datasets will fail.');
      console.warn('   Run "make run-examples" to generate all datasets.\n');
    }
  }

  // Check 3: Generated zarr fixtures.
  //
  // THROW rather than warn, unlike the example datasets above. 19 specs read
  // `tests/fixtures/`, and a missing fixture there is not a degraded run — the spec
  // navigates to a 404 and dies on its own 45 s content-wait with no indication that the
  // cause was a fixture that was never generated. Examples warn because many specs do not
  // need them; fixtures are a hard dependency of every spec that names one.
  //
  // Deliberately NOT auto-generated the way the vitest global setup does it: that
  // generator takes 1-2 minutes, and silently spending that inside a Playwright global
  // setup is a worse failure mode than an actionable error.
  await assertGeneratedFixtures(projectRoot, serverMetadata.dataBaseURL);

  // Check 4: Verify we can write to test output directory
  const testResultsDir = path.join(__dirname, '../../../test-results');

  try {
    if (!fs.existsSync(testResultsDir)) {
      fs.mkdirSync(testResultsDir, { recursive: true });
    }
    // Also create debug subdirectory for agent-driver output
    const debugDir = path.join(testResultsDir, 'debug');
    if (!fs.existsSync(debugDir)) {
      fs.mkdirSync(debugDir, { recursive: true });
    }
    console.log('✅ Test output directories are writable');
  } catch (error) {
    console.error('❌ Cannot create test output directories:', error);
    throw error;
  }

  if (datasetWarnings.length > 0) {
    console.warn(
      `\n⚠️  Pre-flight checks completed with dataset warnings: ${datasetWarnings.join('; ')}.\n`
    );
  } else {
    console.log('\n✅ Pre-flight checks passed!\n');
  }
}
