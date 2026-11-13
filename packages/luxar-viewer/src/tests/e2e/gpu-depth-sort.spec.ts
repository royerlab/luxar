/**
 * The GPU depth sort (#3162) on a real WebGPU device, against the CPU kernel.
 *
 * Runs in the native-WebGPU project only (`pnpm test:e2e:webgpu`); the harness
 * page (`/gpu-sort-harness.html`) builds synthetic element textures, sorts them
 * with `GpuDepthSorter`, and compares the result with `sort_splats_by_depth`
 * in the page.
 *
 * The contract it checks: the GPU order is a permutation, back to front by the
 * CPU's own keys except where a key landed one bucket away (WGSL may fuse a
 * multiply-add and its division is not correctly rounded), and stable. A
 * degenerate depth range or a node entirely behind the camera gives exactly
 * the CPU's identity order.
 *
 * @module tests/e2e/gpu-depth-sort.spec
 */

import { test, expect, type Page } from '@playwright/test';

import type { GpuSortCase, GpuSortReport } from './harnesses/gpu-sort-harness';
import { isWebGPUProject } from './webgpu-project';

async function runCase(page: Page, spec: GpuSortCase): Promise<GpuSortReport> {
  return page.evaluate((s) => window.__gpuSortHarness!.run(s), spec);
}

/** Near-boundary keys may differ by one bucket from the CPU's, rarely. */
function expectBackToFront(report: GpuSortReport, count: number): void {
  const allowed = Math.ceil(count * 1e-3) + 2;
  expect(report.permutation).toBe(true);
  expect(report.descents).toBeLessThanOrEqual(allowed);
  expect(report.maxDescent).toBeLessThanOrEqual(1);
  expect(report.tieInversions).toBeLessThanOrEqual(allowed);
}

test.describe('GPU depth sort', () => {
  test.beforeEach(async ({ page }, testInfo) => {
    test.skip(!isWebGPUProject(testInfo), 'needs a native WebGPU device');
    const allowFallback = process.env.LUXAR_E2E_WEBGPU_ALLOW_FALLBACK === '1';
    await page.goto(`/gpu-sort-harness.html${allowFallback ? '?allowFallbackAdapter' : ''}`);
    await page.waitForFunction(() => Boolean(window.__gpuSortHarness));
  });

  const sizes = [1, 7, 255, 1023, 1024, 1025, 4097, 100_000, 1_000_003];
  for (const count of sizes) {
    test(`orders ${count} splat centers back to front`, async ({ page }) => {
      const report = await runCase(page, {
        count,
        texelsPerElement: 4,
        centerMode: 'texel0',
        seed: count,
        layout: 'cloud',
      });
      expectBackToFront(report, count);
    });
  }

  test('orders point centers (3 texels per element)', async ({ page }) => {
    const report = await runCase(page, {
      count: 50_000,
      texelsPerElement: 3,
      centerMode: 'texel0',
      seed: 3,
      layout: 'cloud',
    });
    expectBackToFront(report, 50_000);
  });

  test('orders line segments by their midpoints (6 texels per element)', async ({ page }) => {
    const report = await runCase(page, {
      count: 50_000,
      texelsPerElement: 6,
      centerMode: 'midpoint01',
      seed: 6,
      layout: 'cloud',
    });
    expectBackToFront(report, 50_000);
  });

  test('puts elements behind the camera first, like the CPU', async ({ page }) => {
    const report = await runCase(page, {
      count: 200_000,
      texelsPerElement: 4,
      centerMode: 'texel0',
      seed: 11,
      layout: 'straddle',
    });
    expectBackToFront(report, 200_000);
  });

  for (const layout of ['behind', 'flat'] as const) {
    test(`keeps the identity order when the depth range is degenerate (${layout})`, async ({
      page,
    }) => {
      const report = await runCase(page, {
        count: 10_000,
        texelsPerElement: 4,
        centerMode: 'texel0',
        seed: 5,
        layout,
      });
      expect(report.permutation).toBe(true);
      expect(report.samePositions).toBe(10_000);
    });
  }

  test('sorts 10M splats', async ({ page }) => {
    test.setTimeout(180_000);
    const report = await runCase(page, {
      count: 10_000_000,
      texelsPerElement: 4,
      centerMode: 'texel0',
      seed: 10,
      layout: 'cloud',
      repeats: 40,
    });
    expectBackToFront(report, 10_000_000);
    test.info().annotations.push({
      type: 'gpu-sort-10m',
      description: `${report.gpuMs?.toFixed(2)} ms on ${report.adapter}; descents ${report.descents}`,
    });
  });
});
