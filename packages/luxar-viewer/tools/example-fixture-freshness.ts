import { execFileSync } from 'node:child_process';

// Keep synchronized with scripts/run_examples.py; the Python test enforces this contract.
const STALE_EXIT_CODE = 3;

export interface ExampleFixtureFreshness {
  status: 'current' | 'stale' | 'unavailable';
  detail?: string;
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
