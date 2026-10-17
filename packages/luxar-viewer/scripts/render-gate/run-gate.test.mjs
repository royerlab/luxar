import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, expect, it } from 'vitest';

import { pacingBound } from './pacing.mjs';

const dir = mkdtempSync(join(tmpdir(), 'luxar-gate-report-'));
beforeEach(() => mkdirSync(dir, { recursive: true }));
afterEach(() => rmSync(dir, { recursive: true, force: true }));

it('reports the A/A buffers and scene state for an excluded view', () => {
  const score = (differing) => ({ differing, maxDriftUlp: 1, p9999Ulp: 1, flips: 0 });
  const report = {
    meta: {
      label: 'control',
      base: { ref: 'HEAD', sha: '1234567890' },
      cand: { ref: 'HEAD', sha: '1234567890' },
      host: 'test',
      gpu: 'test',
      verdict: 'INCOMPLETE',
      browserRelaunches: 0,
    },
    exact: [
      {
        case: 'glass',
        backend: 'webgl',
        dsf: 2,
        view: 'perspective#0',
        cls: 'IDENTICAL',
        status: 'excluded',
        failures: ['nondeterministic baseline'],
        hdr: score(0),
        ldr: score(0),
        control: {
          hdr: { ...score(6130), bbox: { count: 6130, minX: 10, minY: 20, maxX: 90, maxY: 110 } },
          ldr: { ...score(3363), bbox: null },
          pick: { lengthMismatch: true, first: 4096, second: 2048 },
          counts: { first: { lens: 'mesh:1:?:true' }, second: { lens: 'mesh:0:?:true' } },
          camera: { first: { world: [1] }, second: { world: [2] } },
          stable: { first: false, second: true },
          bufferStability: {
            first: { hdr: true, ldr: false },
            second: { hdr: true, ldr: true },
          },
          heatmap: 'heatmaps/glass-control-hdr.png',
          ldrHeatmap: 'heatmaps/glass-control-ldr.png',
        },
      },
    ],
    perf: null,
    suites: {},
  };
  const jsonPath = join(dir, 'report.json');
  mkdirSync(dir, { recursive: true });
  writeFileSync(jsonPath, JSON.stringify(report));
  execFileSync(process.execPath, ['scripts/render-gate/run-gate.mjs', '--from-json', jsonPath]);
  const markdown = readFileSync(join(dir, 'report.md'), 'utf8');
  expect(markdown).toContain('control: HDR 6130 px');
  expect(markdown).toContain('bbox [10,20]-[90,110]');
  expect(markdown).toContain('LDR 3363 px, drift 1.0, p99.99 1.0, flips 0, bbox -');
  expect(markdown).toContain('pick buffer sizes differ (4096 vs 2048)');
  expect(markdown).toContain('counts differ; camera differs');
  expect(markdown).toContain('within-page HDR/LDR stable true/false, true/true');
  expect(markdown).toContain('[A/A HDR heatmap](heatmaps/glass-control-hdr.png)');
  expect(markdown).toContain('[A/A LDR heatmap](heatmaps/glass-control-ldr.png)');
});

it('flags only WebGL throughput close to the measured 60 Hz floor', () => {
  expect(pacingBound('webgl', 17.2)).toBe(true);
  expect(pacingBound('webgl', 15.6)).toBe(false);
  expect(pacingBound('webgl', 18)).toBe(false);
  expect(pacingBound('webgpu', 17.2)).toBe(false);
  expect(pacingBound('webgl', NaN)).toBe(false);
});

it('shows pacing-bound arms in a saved throughput report', () => {
  const report = {
    meta: {
      label: 'pacing',
      base: { ref: 'HEAD', sha: '1234567890' },
      cand: { ref: 'HEAD', sha: '1234567890' },
      host: 'test',
      gpu: 'test',
      verdict: 'PASS',
      browserRelaunches: 0,
    },
    exact: null,
    perf: [
      {
        case: 'mixed',
        backend: 'webgl',
        status: 'pass',
        metrics: {
          throughputMs: {
            baseMedian: 17.2,
            candMedian: 2.6,
            ratio: 0.15,
            lo: 0.14,
            hi: 0.16,
            floor: 0.01,
            verdict: 'win',
            pacingBound: { base: true, cand: false },
          },
        },
      },
    ],
    suites: {},
  };
  const jsonPath = join(dir, 'report.json');
  writeFileSync(jsonPath, JSON.stringify(report));
  execFileSync(process.execPath, ['scripts/render-gate/run-gate.mjs', '--from-json', jsonPath]);
  expect(readFileSync(join(dir, 'report.md'), 'utf8')).toContain('pacing-bound: base');
});
