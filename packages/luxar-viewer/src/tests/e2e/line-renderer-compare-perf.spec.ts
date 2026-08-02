/**
 * Line-rendering backend compare — WebGL vs WebGPU FPS benchmark.
 *
 * Captures N FPS samples on a line-heavy dataset under each rendering
 * backend (`?renderer=webgl` and `?renderer=webgpu`) so the WebGPU
 * regression for line-heavy scenes (reported on the Zebrahub demo)
 * can be quantified and tracked across changes.
 *
 * Two-pass structure:
 *   1. Load with `?renderer=webgl`, sample N FPS frames.
 *   2. Reload with `?renderer=webgpu`, sample N FPS frames.
 *   3. Write side-by-side stats + the ratio to a JSON artifact.
 *
 * The active backend is verified via `__luxarDebug.app.sceneManager`'s
 * capabilities (or the matching console log) so the comparison can't
 * silently pin both runs to the same backend after a fallback.
 *
 * Not a CI gate — strictly developer-facing diagnostic. Run with:
 *
 *   npx playwright test line-renderer-compare-perf.spec.ts --workers=1
 *
 * @module tests/e2e/line-renderer-compare-perf.spec
 */

import { test } from '@playwright/test';
import { waitForLuxarReady } from './helpers';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUTPUT_PATH = path.join(__dirname, '../../../line-renderer-compare-perf.json');

// Line-heavy demo. `lines_basic_example.luxar.zarr` is the universal
// fallback — small, always present, no nD navigation needed (lines
// render at the default slice). For the actual Zebrahub regression
// repro, swap to `zebrahub_velocity_streamlines_hifi.luxar.zarr` and
// navigate to a populated timepoint inside the spec (the hifi set is
// 4-D — lines aren't visible until you advance the time axis).
const DATASETS = [
  {
    name: 'lines_basic_example.luxar.zarr',
    url: 'http://localhost:9000/datasets/examples/lines_basic_example.luxar.zarr',
  },
  {
    name: 'zebrahub_velocity_streamlines_hifi.luxar.zarr',
    url: 'http://localhost:9000/datasets/demos/zebrahub_velocity_streamlines_hifi.luxar.zarr',
  },
];

const SAMPLE_COUNT = 5;
const SAMPLE_DURATION_MS = 2000;
const BACKENDS = ['webgl', 'webgpu'] as const;
type Backend = (typeof BACKENDS)[number];

function median(values: number[]): number {
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

interface BackendSample {
  backend: Backend;
  actualApi: string | null;
  fps: { min: number; median: number; max: number; samples: number[] };
}

interface DatasetResult {
  dataset: string;
  url: string;
  perBackend: BackendSample[];
  ratioWebgpuOverWebgl: number | null;
}

async function urlExists(url: string): Promise<boolean> {
  try {
    const r = await fetch(url, { method: 'HEAD' });
    return r.ok;
  } catch {
    return false;
  }
}

test('line rendering: compare WebGL vs WebGPU FPS on line-heavy scenes', async ({ page }) => {
  // Benchmark-only diagnostic — skipped by default so `pnpm test:e2e`
  // doesn't sit through a 10-minute FPS sample on every run. Opt in
  // with `LUXAR_RUN_BENCHMARKS=1 npx playwright test
  // line-renderer-compare-perf.spec.ts --workers=1`.
  test.skip(
    process.env.LUXAR_RUN_BENCHMARKS !== '1',
    'developer-facing benchmark; set LUXAR_RUN_BENCHMARKS=1 to run'
  );
  test.setTimeout(600_000);

  // Pick the first dataset whose URL is reachable (the dev static
  // server at :9000 must be up; `make demo` brings it online). The
  // hifi Zebrahub set is the priority; the bundled lines_basic
  // example is the universal fallback.
  let chosen: (typeof DATASETS)[number] | null = null;
  for (const d of DATASETS) {
    if (await urlExists(d.url)) {
      chosen = d;
      break;
    }
  }
  test.skip(chosen === null, 'No line-heavy dataset reachable at localhost:9000');
  if (!chosen) return;

  const results: DatasetResult = {
    dataset: chosen.name,
    url: chosen.url,
    perBackend: [],
    ratioWebgpuOverWebgl: null,
  };

  for (const backend of BACKENDS) {
    await page.goto(`/?src=${chosen.url}&renderer=${backend}&debug`);
    await waitForLuxarReady(page);

    // Read the actually-active backend AND the visible line count so a
    // silent fallback (e.g., WebGPU adapter unavailable) or a load
    // that produced an empty scene can't pollute the comparison.
    const probe = await page.evaluate(() => {
      const dbg = (
        window as unknown as {
          __luxarDebug?: {
            app?: { sceneManager?: { capabilities?: { apiSurface?: string }; scene?: unknown } };
          };
        }
      ).__luxarDebug;
      const api = dbg?.app?.sceneManager?.capabilities?.apiSurface ?? null;
      let visibleSegments = 0;
      const scene = dbg?.app?.sceneManager?.scene as
        { traverse?: (cb: (o: unknown) => void) => void } | undefined;
      scene?.traverse?.((obj: unknown) => {
        const o = obj as {
          userData?: { nodeType?: string };
          geometry?: { instanceCount?: number };
        };
        if (o.userData?.nodeType === 'lines' && typeof o.geometry?.instanceCount === 'number') {
          visibleSegments += o.geometry.instanceCount;
        }
      });
      return { api, visibleSegments };
    });
    const actualApi = probe.api;
    console.log(
      `  [${backend} → ${actualApi ?? '?'}] visible line segments: ${probe.visibleSegments}`
    );
    if (probe.visibleSegments === 0) {
      console.log(
        `  [${backend}] WARNING: 0 visible segments — dataset may need nD navigation to a populated slice. Skipping samples for this backend.`
      );
      results.perBackend.push({
        backend,
        actualApi,
        fps: { min: 0, median: 0, max: 0, samples: [] },
      });
      continue;
    }

    const samples: number[] = [];
    for (let i = 0; i < SAMPLE_COUNT; i++) {
      // Per-sample hard timeout: if rAF never fires (broken page,
      // backgrounded tab, etc.) the eval would otherwise hang until
      // the suite timeout. 3× sample duration is a generous cap.
      const fps = await page.evaluate(async (durationMs: number) => {
        const debug = (window as unknown as { __luxarDebug: { renderOnce: () => void } })
          .__luxarDebug;
        let frames = 0;
        const startTime = performance.now();
        return new Promise<number>((resolve) => {
          let timedOut = false;
          const watchdog = setTimeout(() => {
            timedOut = true;
            const elapsed = performance.now() - startTime;
            resolve(elapsed > 0 ? frames / (elapsed / 1000) : 0);
          }, durationMs * 3);
          const measureFrame = () => {
            if (timedOut) return;
            frames++;
            const elapsed = performance.now() - startTime;
            if (elapsed < durationMs) {
              debug.renderOnce();
              requestAnimationFrame(measureFrame);
            } else {
              clearTimeout(watchdog);
              resolve(frames / (elapsed / 1000));
            }
          };
          debug.renderOnce();
          requestAnimationFrame(measureFrame);
        });
      }, SAMPLE_DURATION_MS);

      samples.push(fps);
      console.log(
        `  [${backend} → ${actualApi ?? '?'}] sample ${i + 1}/${SAMPLE_COUNT}: ${fps.toFixed(1)} fps`
      );
    }
    if (samples.length === 0) continue;

    results.perBackend.push({
      backend,
      actualApi,
      fps: {
        min: Math.min(...samples),
        median: median(samples),
        max: Math.max(...samples),
        samples,
      },
    });
  }

  // Ratio is meaningful only if both backends actually dispatched to
  // distinct underlying APIs. If both fell back to the same API the
  // comparison is degenerate.
  const wgl = results.perBackend.find((b) => b.backend === 'webgl');
  const wgpu = results.perBackend.find((b) => b.backend === 'webgpu');
  if (wgl && wgpu && wgl.actualApi !== wgpu.actualApi && wgl.fps.median > 0) {
    results.ratioWebgpuOverWebgl = wgpu.fps.median / wgl.fps.median;
  }

  console.log(
    `\n📊 Line-rendering backend compare — ${results.dataset}` +
      results.perBackend
        .map(
          (b) =>
            `\n   [${b.backend} → ${b.actualApi ?? '?'}]` +
            ` min ${b.fps.min.toFixed(1)} / median ${b.fps.median.toFixed(1)} / max ${b.fps.max.toFixed(1)} fps`
        )
        .join('') +
      (results.ratioWebgpuOverWebgl !== null
        ? `\n   webgpu / webgl median ratio: ${results.ratioWebgpuOverWebgl.toFixed(2)}×`
        : '\n   (ratio omitted — backends collapsed to the same API)')
  );

  fs.writeFileSync(
    OUTPUT_PATH,
    JSON.stringify(
      {
        ...results,
        sampleCount: SAMPLE_COUNT,
        sampleDurationMs: SAMPLE_DURATION_MS,
        capturedAt: new Date().toISOString(),
      },
      null,
      2
    )
  );
});
