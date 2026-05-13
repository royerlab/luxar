/**
 * Points migration perf gate — multi-sample FPS measurement.
 *
 * After the THREE.Points → THREE.Mesh + InstancedBufferAttribute
 * migration (commit a0566830), this spec captures N=5 FPS samples on a
 * representative points-heavy dataset so the regression check has real
 * statistical mass behind it rather than depending on a single-sample
 * comparison against a stale baseline.
 *
 * Output: the min / median / max FPS across N trials, written both to
 * the test console and to `points-migration-perf.json` at the package
 * root so it can be diffed against a previous run.
 *
 * Not a CI gate — strictly developer-facing diagnostic.
 *
 * @module tests/e2e/points-migration-perf.spec
 */

import { test } from '@playwright/test';
import { waitForLuxarReady } from './helpers';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUTPUT_PATH = path.join(__dirname, '../../../points-migration-perf.json');

// Densest points-only example dataset shipped with the repo. The
// other "dense" candidate (`dense_grid_5d_example.zarr`) is a 5-D
// scene where nD slicing reduces the visible point count to a small
// subset — that doesn't stress the per-instance pipeline we care
// about, so we benchmark the structured 3-D one.
const DATASET = 'http://localhost:9000/datasets/examples/dense_cubic_gradient_example.zarr';

const SAMPLE_COUNT = 5;
const SAMPLE_DURATION_MS = 2000;

function median(values: number[]): number {
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

test('points migration: capture N=5 FPS samples on dense_cubic_gradient', async ({ page }) => {
  test.setTimeout(120_000);

  await page.goto(`/?src=${DATASET}&debug`);
  await waitForLuxarReady(page);

  const samples: number[] = [];
  for (let i = 0; i < SAMPLE_COUNT; i++) {
    const fps = await page.evaluate(async (durationMs: number) => {
      const debug = (window as unknown as { __luxarDebug: { renderOnce: () => void } })
        .__luxarDebug;
      let frames = 0;
      const startTime = performance.now();
      return new Promise<number>((resolve) => {
        const measureFrame = () => {
          frames++;
          const elapsed = performance.now() - startTime;
          if (elapsed < durationMs) {
            debug.renderOnce();
            requestAnimationFrame(measureFrame);
          } else {
            resolve(frames / (elapsed / 1000));
          }
        };
        debug.renderOnce();
        requestAnimationFrame(measureFrame);
      });
    }, SAMPLE_DURATION_MS);

    samples.push(fps);
    console.log(`  sample ${i + 1}/${SAMPLE_COUNT}: ${fps.toFixed(1)} fps`);
  }

  const min = Math.min(...samples);
  const max = Math.max(...samples);
  const med = median(samples);

  const summary = {
    dataset: 'dense_cubic_gradient_example.zarr',
    sampleCount: SAMPLE_COUNT,
    sampleDurationMs: SAMPLE_DURATION_MS,
    fps: { min, median: med, max, samples },
    capturedAt: new Date().toISOString(),
  };

  console.log(
    `\n📊 N=${SAMPLE_COUNT} FPS samples on ${summary.dataset}:` +
      `\n   min: ${min.toFixed(1)}` +
      `\n   median: ${med.toFixed(1)}` +
      `\n   max: ${max.toFixed(1)}`
  );

  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(summary, null, 2));
});
