import { execFileSync } from 'node:child_process';

export type ExampleFixtureFreshness = 'current' | 'stale' | 'unavailable';

type FreshnessChecker = (projectRoot: string) => void;

function runFreshnessChecker(projectRoot: string): void {
  execFileSync('hatch', ['run', 'python', 'scripts/run_examples.py', '--check'], {
    cwd: projectRoot,
    stdio: 'pipe',
  });
}

export function checkExampleFixtureFreshness(
  projectRoot: string,
  checker: FreshnessChecker = runFreshnessChecker
): ExampleFixtureFreshness {
  try {
    checker(projectRoot);
    return 'current';
  } catch (error: unknown) {
    if (error && typeof error === 'object' && 'status' in error) {
      const status = (error as { status?: unknown }).status;
      if (typeof status === 'number') return 'stale';
    }
    return 'unavailable';
  }
}
