import { execFileSync } from 'node:child_process';

// Keep synchronized with scripts/run_examples.py; the Python test enforces this contract.
const STALE_EXIT_CODE = 3;
export const EXAMPLE_DATASETS_STATUS_ENV = 'LUXAR_E2E_EXAMPLES_STATUS';

/**
 * How much this run can vouch for `datasets/examples`.
 *
 * FOUR states, not two, and the distinction is load-bearing. The wire to the
 * Playwright workers used to be a single `LUXAR_E2E_EXAMPLES_STALE=1` boolean,
 * so everything that was not *stale* — including a checkout where
 * `datasets/examples` does not exist at all — reached the workers looking
 * exactly like a healthy run. The observed cost: nine `transform-hierarchy`
 * specs failed with bare 45 s `waitForLuxarReady` timeouts, the page showing
 * only "Unable to Load Dataset", and nothing anywhere said the real cause was
 * a fixture directory that had never been generated.
 *
 * "Cannot vouch for it" has to be its own value, or it reads as "fine".
 * `missing` and `unavailable` are kept apart because one is a certainty we can
 * give an exact remedy for and the other is an unanswered question.
 */
export type ExampleFixtureStatus = 'current' | 'stale' | 'missing' | 'unavailable';

export interface ExampleFixtureFreshness {
  status: ExampleFixtureStatus;
  detail?: string;
}

export interface ExampleFixtureFreshnessReporter {
  log(message: string): void;
  warn(message: string): void;
}

function runFreshnessChecker(projectRoot: string): void {
  execFileSync('hatch', ['run', 'python', 'scripts/run_examples.py', '--check'], {
    cwd: projectRoot,
    stdio: 'pipe',
    timeout: 120_000,
  });
}

function checkerErrorDetail(error: object): string | undefined {
  if (!('stderr' in error)) return undefined;
  const stderr = (error as { stderr?: unknown }).stderr;
  if (typeof stderr === 'string') return stderr.trim() || undefined;
  if (Buffer.isBuffer(stderr)) return stderr.toString().trim() || undefined;
  return undefined;
}

export function checkExampleFixtureFreshness(projectRoot: string): ExampleFixtureFreshness {
  try {
    runFreshnessChecker(projectRoot);
    return { status: 'current' };
  } catch (error: unknown) {
    if (error && typeof error === 'object' && 'status' in error) {
      const status = (error as { status?: unknown }).status;
      if (status === STALE_EXIT_CODE) return { status: 'stale' };
      return { status: 'unavailable', detail: checkerErrorDetail(error) };
    }
    return { status: 'unavailable' };
  }
}

export function exposeExampleFixtureFreshnessToWorkers(
  freshness: ExampleFixtureFreshness,
  environment: Record<string, string | undefined> = process.env
): void {
  // Only `current` clears the variable. Every other status — including the two
  // that mean "unverified" — is forwarded verbatim, so a worker can tell them
  // apart instead of inferring health from an absent flag.
  if (freshness.status === 'current') {
    delete environment[EXAMPLE_DATASETS_STATUS_ENV];
  } else {
    environment[EXAMPLE_DATASETS_STATUS_ENV] = freshness.status;
  }
}

export function reportExampleFixtureFreshness(
  freshness: ExampleFixtureFreshness,
  reporter: ExampleFixtureFreshnessReporter = console
): boolean {
  if (freshness.status === 'current') {
    reporter.log('✅ Example datasets match the current fixture producer');
    return false;
  }

  if (freshness.status === 'missing') {
    reporter.warn('⚠️  Example datasets have not been generated.');
    if (freshness.detail) reporter.warn(`   Expected them at: ${freshness.detail}`);
    reporter.warn('   Run "make run-examples" from the repository root to generate them.');
    reporter.warn('   Every spec that reads datasets/examples will fail until you do.');
    reporter.warn('   Continuing so specs that do not read example datasets can still run.\n');
    return true;
  }

  if (freshness.status === 'stale') {
    reporter.warn('⚠️  Example datasets are stale.');
    reporter.warn('   Run "make run-examples" from the repository root to refresh them.');
    reporter.warn('   Continuing so specs that do not read example datasets can still run.\n');
    return true;
  }

  reporter.warn('⚠️  Could not run the example fixture freshness checker.');
  if (freshness.detail) reporter.warn(`   ${freshness.detail}`);
  reporter.warn('   Continuing with presence checks only.\n');
  return true;
}
