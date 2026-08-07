import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  diagnosticLevel,
  diffWarningMultisets,
  ensureConversionError,
  loadWarningBaseline,
  normalizeDiagnostic,
  parseArgs,
  saveWarningBaseline,
} from './check-typedoc-warnings.mjs';

test('recognizes only tagged TypeDoc console diagnostics', () => {
  assert.equal(diagnosticLevel('\u001b[93m[warning]\u001b[0m unresolved link'), 'warning');
  assert.equal(diagnosticLevel('[error] conversion failed'), 'error');
  assert.equal(diagnosticLevel('dependency warning without a TypeDoc tag'), undefined);
});

test('normalizes checkout paths and whitespace', () => {
  const root = '/tmp/checkouts/luxar/packages/luxar-viewer';
  assert.equal(
    normalizeDiagnostic(
      '\u001b[93m[warning]\u001b[0m The relative path /tmp/checkouts/luxar/packages/luxar-viewer/examples/embed/  is not a file',
      root
    ),
    'The relative path <viewer>/examples/embed/ is not a file'
  );
});

test('parses checker arguments and rejects an omitted baseline path', () => {
  assert.deepEqual(parseArgs(['--', '--json', '--update-baseline', '--baseline', 'known.json']), {
    baseline: 'known.json',
    json: true,
    updateBaseline: true,
  });
  assert.throws(() => parseArgs(['--baseline']), /requires a path/);
  assert.throws(() => parseArgs(['--baseline', '--json']), /requires a path/);
  assert.throws(() => parseArgs(['--unknown']), /Unknown argument/);
});

test('reports an unexplained conversion failure explicitly', () => {
  assert.deepEqual(ensureConversionError(false, []), ['TypeDoc conversion produced no project']);
  assert.deepEqual(ensureConversionError(false, ['compiler error']), ['compiler error']);
  assert.deepEqual(ensureConversionError(true, []), []);
});

test('warning differences preserve duplicate counts', () => {
  assert.deepEqual(diffWarningMultisets(['a', 'a', 'c'], ['a', 'b']), {
    newWarnings: ['a', 'c'],
    fixedWarnings: ['b'],
  });
  assert.deepEqual(diffWarningMultisets([], []), {
    newWarnings: [],
    fixedWarnings: [],
  });
});

test('baseline round-trip is sorted and deterministic', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'typedoc-baseline-'));
  const baseline = path.join(directory, 'baseline.json');

  await saveWarningBaseline(baseline, ['z warning', 'a warning']);

  assert.deepEqual(await loadWarningBaseline(baseline), ['a warning', 'z warning']);
  const payload = JSON.parse(await readFile(baseline, 'utf8'));
  assert.deepEqual(payload.warnings, ['a warning', 'z warning']);
  assert.match(payload._comment, /newly introduced warnings fail/);
});

test('missing and malformed baselines fail closed', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'typedoc-baseline-'));
  const baseline = path.join(directory, 'baseline.json');

  await assert.rejects(loadWarningBaseline(baseline), /Could not read TypeDoc warning baseline/);
  await saveWarningBaseline(baseline, ['warning']);

  const payload = JSON.parse(await readFile(baseline, 'utf8'));
  payload.warnings = [42];
  await writeFile(baseline, JSON.stringify(payload), 'utf8');

  await assert.rejects(loadWarningBaseline(baseline), /Malformed TypeDoc warning baseline/);
});
