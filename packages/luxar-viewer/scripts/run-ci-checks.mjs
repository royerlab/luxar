#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import process from 'node:process';
import { pathToFileURL } from 'node:url';

export const CI_CHECKS = [
  'check:overrides',
  'check:format',
  'typecheck',
  'lint',
  'check:layers',
  'check:knip:ci',
  'test:coverage',
  'check:coverage-slack',
];

/** Run every named check and return the ones that failed. */
export function runChecks(checks, runner) {
  const failures = [];
  for (const check of checks) {
    if (runner(check) !== 0) failures.push(check);
  }
  return failures;
}

function runPnpmCheck(check) {
  console.log(`\n=== pnpm run ${check} ===\n`);
  const command = process.platform === 'win32' ? 'pnpm.cmd' : 'pnpm';
  const result = spawnSync(command, ['run', check], { stdio: 'inherit' });
  if (result.error) console.error(`Failed to run ${check}: ${result.error.message}`);
  return result.status ?? 1;
}

function main() {
  const failures = runChecks(CI_CHECKS, runPnpmCheck);
  if (failures.length === 0) {
    console.log('\ncheck:ci: all checks passed.');
    return;
  }
  console.error(`\ncheck:ci: ${failures.length} check(s) failed: ${failures.join(', ')}`);
  process.exitCode = 1;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) main();
