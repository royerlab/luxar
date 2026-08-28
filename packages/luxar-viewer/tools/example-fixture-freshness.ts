import { execFileSync } from 'node:child_process';

// Keep synchronized with scripts/run_examples.py; the Python test enforces this contract.
const STALE_EXIT_CODE = 3;
export const EXAMPLE_DATASETS_STALE_ENV = 'LUXAR_E2E_EXAMPLES_STALE';

export interface ExampleFixtureFreshness {
  status: 'current' | 'stale' | 'unavailable';
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
  if (freshness.status === 'stale') {
    environment[EXAMPLE_DATASETS_STALE_ENV] = '1';
  } else {
    delete environment[EXAMPLE_DATASETS_STALE_ENV];
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
