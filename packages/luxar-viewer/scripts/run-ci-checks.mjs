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

/** Run named checks, aggregating failures unless bail is requested. */
export function runChecks(checks, runner, options = {}) {
  const failures = [];
  for (const check of checks) {
    const result = runner(check);
    if (result.signal) return { failures, killed: { check, signal: result.signal } };
    if (result.status !== 0) {
      failures.push(check);
      if (options.bail) break;
    }
  }
  return { failures, killed: null };
}

function runPnpmCheck(check) {
  console.log(`\n=== pnpm run ${check} ===\n`);
  const result = spawnSync('pnpm', ['run', check], { stdio: 'inherit' });
  if (result.error) console.error(`Failed to run ${check}: ${result.error.message}`);
  return { status: result.status ?? 1, signal: result.signal };
}

function main() {
  const { failures, killed } = runChecks(CI_CHECKS, runPnpmCheck, {
    bail: process.argv.slice(2).includes('--bail'),
  });
  if (killed) {
    console.error(
      `\ncheck:ci: ${killed.check} was killed by ${killed.signal}; remaining checks skipped.`
    );
    process.exitCode = 1;
    return;
  }
  if (failures.length === 0) {
    console.log('\ncheck:ci: all checks passed.');
    return;
  }
  console.error(`\ncheck:ci: ${failures.length} check(s) failed: ${failures.join(', ')}`);
  process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main();
