#!/usr/bin/env node
/**
 * Run the viewer merge gates without the old `&&` chain's fail-fast behavior.
 * Ordinary failures are collected so one CI run reports them together, while
 * killed checks stop immediately to avoid manufacturing dependent failures.
 */

import { spawnSync } from 'node:child_process';
import { constants } from 'node:os';
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
    const signal = result.signal ?? signalFromStatus(result.status);
    if (signal) return { failures, killed: { check, signal } };
    if (result.status !== 0) {
      failures.push(check);
      if (options.bail) break;
    }
  }
  return { failures, killed: null };
}

function signalFromStatus(status) {
  if (typeof status !== 'number' || status < 128) return null;
  const signalNumber = status - 128;
  return (
    Object.entries(constants.signals).find(([, number]) => number === signalNumber)?.[0] ?? null
  );
}

function runPnpmCheck(check) {
  console.log(`\n=== pnpm run ${check} ===\n`);
  const result = spawnSync('pnpm', ['run', check], { stdio: 'inherit' });
  if (result.error) console.error(`Failed to run ${check}: ${result.error.message}`);
  return { status: result.status ?? 1, signal: result.signal };
}

/** Report an aggregated result and return whether every check passed. */
export function reportResult({ failures, killed }, output = console) {
  if (failures.length > 0) {
    output.error(`\ncheck:ci: ${failures.length} check(s) failed: ${failures.join(', ')}`);
  }
  if (killed) {
    output.error(
      `\ncheck:ci: ${killed.check} was killed by ${killed.signal}; remaining checks skipped.`
    );
  }
  if (failures.length === 0 && !killed) output.log('\ncheck:ci: all checks passed.');
  return failures.length === 0 && !killed;
}

function main() {
  const result = runChecks(CI_CHECKS, runPnpmCheck, {
    bail: process.argv.slice(2).includes('--bail'),
  });
  if (!reportResult(result)) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main();
