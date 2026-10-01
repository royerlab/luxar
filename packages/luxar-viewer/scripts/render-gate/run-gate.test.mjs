import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, expect, it } from 'vitest';

const dir = mkdtempSync(join(tmpdir(), 'luxar-gate-report-'));
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
          hdr: score(6130),
          ldr: score(3363),
          pick: null,
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
  expect(markdown).toContain('LDR 3363 px');
  expect(markdown).toContain('counts differ; camera differs');
  expect(markdown).toContain('within-page HDR/LDR stable true/false, true/true');
  expect(markdown).toContain('[A/A HDR heatmap](heatmaps/glass-control-hdr.png)');
  expect(markdown).toContain('[A/A LDR heatmap](heatmaps/glass-control-ldr.png)');
});
