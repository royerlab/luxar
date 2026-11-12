import { execFileSync, spawnSync } from 'node:child_process';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, expect, it } from 'vitest';

import { pacingBound } from './pacing.mjs';

const dir = mkdtempSync(join(tmpdir(), 'luxar-gate-report-'));
beforeEach(() => mkdirSync(dir, { recursive: true }));
afterEach(() => rmSync(dir, { recursive: true, force: true }));

const gateScript = 'scripts/render-gate/run-gate.mjs';
const buildCache = '../../delme/gate-builds';
const cacheEntries = () => (existsSync(buildCache) ? readdirSync(buildCache).sort() : []);

it.each([
  { args: ['--help'], status: 0, output: /--backends <value>/ },
  { args: ['--bogus'], status: 2, output: /unknown option --bogus/ },
  { args: ['--suites', 'exact'], status: 2, output: /unknown option --suites/ },
  { args: ['--suite'], status: 2, output: /--suite requires a value/ },
  { args: ['--suite', '--heavy'], status: 2, output: /--suite requires a value/ },
  { args: ['--headed=x'], status: 2, output: /--headed does not take a value/ },
  { args: ['--chrome-args='], status: 2, output: /--chrome-args requires a value/ },
])('validates $args before any build', ({ args, status, output }) => {
  const cacheExisted = existsSync(buildCache);
  const before = cacheEntries();
  const result = spawnSync(process.execPath, [gateScript, ...args], {
    encoding: 'utf8',
    timeout: 20_000,
  });
  expect(result.error).toBeUndefined();
  expect(result.status).toBe(status);
  expect(result.stdout + result.stderr).toMatch(output);
  expect(result.stdout + result.stderr).not.toMatch(/ensureBuild|building|vite build/i);
  expect(existsSync(buildCache)).toBe(cacheExisted);
  expect(cacheEntries()).toEqual(before);
});

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

it.each([
  {
    high: true,
    medianPerCpu: 0.75,
    arms: { base: { median: 12, max: 14 }, cand: { median: 11.5, max: 13 } },
    line: 'base 12.0 (max 14.0), cand 11.5 (max 13.0)',
  },
  {
    high: false,
    medianPerCpu: 0.25,
    arms: { base: { median: 4, max: 6 }, cand: { median: 3.5, max: 5 } },
    line: 'base 4.0 (max 6.0), cand 3.5 (max 5.0)',
  },
])('reports host load with high=$high', ({ high, medianPerCpu, arms, line }) => {
  const report = {
    meta: {
      label: 'load',
      base: { ref: 'HEAD', sha: '1234567890' },
      cand: { ref: 'HEAD', sha: '1234567890' },
      host: 'test',
      gpu: 'test',
      verdict: 'PASS',
      browserRelaunches: 0,
      hostLoad: {
        cpus: 16,
        arms,
        medianPerCpu,
        high,
      },
    },
    exact: null,
    perf: null,
    suites: {},
  };
  const jsonPath = join(dir, 'report.json');
  writeFileSync(jsonPath, JSON.stringify(report));
  execFileSync(process.execPath, [gateScript, '--from-json', jsonPath]);
  const markdown = readFileSync(join(dir, 'report.md'), 'utf8');
  expect(markdown).toContain(`- host load (1-min average, 16 CPUs): ${line}`);
  expect(markdown.includes('**warning: the host was busy**')).toBe(high);
  if (high) expect(markdown).toContain('median load 0.75 per CPU');
});

it('flags only WebGL throughput close to the measured 60 Hz floor', () => {
  expect(pacingBound('webgl', 17.2)).toBe(true);
  expect(pacingBound('webgl', 15.6)).toBe(false);
  expect(pacingBound('webgl', 16)).toBe(false);
  expect(pacingBound('webgl', 17.7)).toBe(true);
  expect(pacingBound('webgl', 18.2)).toBe(false);
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
            verdict: 'pacing-bound',
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
  expect(readFileSync(join(dir, 'report.md'), 'utf8')).toContain('pacing-bound (base)');
});
