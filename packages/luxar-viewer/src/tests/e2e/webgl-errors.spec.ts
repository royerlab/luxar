/**
 * WebGL Error Detection Tests
 *
 * CRITICAL: These tests catch WebGL errors that indicate rendering bugs.
 *
 * WebGL errors like GL_INVALID_OPERATION are SILENT - they don't throw
 * JavaScript exceptions, so they can go unnoticed unless explicitly checked.
 *
 * This test suite was added after discovering hundreds of
 * "Vertex buffer is not big enough" errors in production demos.
 */

import { test, expect } from './fixtures';
import { evaluateWithDeadline, waitForLuxarReady } from './helpers';
import { flushRenderTicks, reportFlushVerdict } from './render-ticks';

// Test all example datasets for WebGL errors
const DATASETS = [
  'http://localhost:9000/datasets/examples/sharpness_showcase_example.luxar.zarr',
  'http://localhost:9000/datasets/examples/build_example_structured.luxar.zarr',
  'http://localhost:9000/datasets/examples/dense_grid_5d_example.luxar.zarr',
  'http://localhost:9000/datasets/examples/hierarchy_example.luxar.zarr',
  'http://localhost:9000/datasets/examples/transform_example.luxar.zarr',
];

test.describe('WebGL Error Detection - Critical', () => {
  test('should render without GL_INVALID_OPERATION errors', async ({ page }) => {
    const webglErrors: string[] = [];

    // Capture ONLY actual WebGL errors, not info messages
    page.on('console', (msg) => {
      const text = msg.text();
      // Only capture actual GL errors, not info messages about WebGL
      if (
        text.includes('GL_INVALID') ||
        text.includes('GL_OUT_OF_MEMORY') ||
        (text.includes('WebGL') && text.includes('error:')) ||
        (text.includes('glDrawArrays') && msg.type() === 'error') ||
        (text.includes('glDrawElements') && msg.type() === 'error')
      ) {
        // Filter out info messages (they start with [ℹ️] or contain "GPU stall")
        if (!text.includes('[ℹ️]') && !text.includes('GPU stall due to ReadPixels')) {
          webglErrors.push(text);
        }
      }
    });

    // Load a dataset known to have issues (sharpness showcase)
    await page.goto(`/?src=${DATASETS[0]}&debug`);
    await waitForLuxarReady(page);

    // Wait for initial render
    await page.waitForFunction(
      () => {
        const info = (window as any).__luxarDebug?.renderer?.info;
        return (info?.render?.frame ?? info?.frame ?? 0) > 2;
      },
      {
        timeout: 10000,
      }
    );

    // Drive several more render ticks to flush any delayed GL errors into the
    // console listener.
    const flush = await flushRenderTicks(page, 5);

    // Check for WebGL errors
    if (webglErrors.length > 0) {
      console.log('\n🚨 WebGL Errors Detected:');
      webglErrors.slice(0, 10).forEach((err) => console.log(`  - ${err}`));
      if (webglErrors.length > 10) {
        console.log(`  ... and ${webglErrors.length - 10} more`);
      }
      console.log('');
    }

    // CRITICAL: Fail test if ANY WebGL errors detected
    expect(webglErrors).toEqual([]);
    // Order matters: a real GL error must be the error you see, so the flush
    // verdict comes second — it fails only when the page is responsive and the
    // renderer still is not drawing, and otherwise just says what it saw.
    await reportFlushVerdict(page, flush);
  });

  test('should render all example datasets without WebGL errors', async ({ page }) => {
    for (const dataset of DATASETS) {
      const webglErrors: string[] = [];

      page.on('console', (msg) => {
        const text = msg.text();
        if (text.includes('GL_INVALID') || text.includes('WebGL error')) {
          webglErrors.push(text);
        }
      });

      await page.goto(`/?src=${dataset}&debug`);
      await waitForLuxarReady(page);

      await page.waitForFunction(
        () => {
          const info = (window as any).__luxarDebug?.renderer?.info;
          return (info?.render?.frame ?? info?.frame ?? 0) > 1;
        },
        { timeout: 10000 }
      );

      // Drive several more render ticks to flush delayed GL errors, on a
      // TIGHTER flush budget than the module default because this one test
      // loops over all five DATASETS, so every per-call bound is paid five
      // times. At the default the flush pair alone would be up to 5 x
      // (COUNTER_PROBE_TIMEOUT_MS 5 s + FLUSH_BUDGET_MS 10 s + a verdict probe
      // 5 s) = 100 s, i.e. the whole 60 s test budget again, merely relocated
      // out of a bare evaluate and into the flush. At 2 s it is 5 x (5 + 2 + 5)
      // = 60 s. Honestly: that is still the entire budget, so a fully starved
      // page exhausts this test no matter what the flush is bounded to — it
      // performs five `page.goto` + `waitForLuxarReady` scene loads in ONE
      // test, and a per-call bound cannot fix a per-test structure. Splitting
      // it per dataset would; the `WebGL Error Detection - All Datasets`
      // describe block below already does exactly that, and is out of scope
      // here.
      const flush = await flushRenderTicks(page, 3, 2000);

      if (webglErrors.length > 0) {
        console.log(`\n🚨 WebGL errors in ${dataset}:`);
        console.log(`  Found ${webglErrors.length} errors`);
        console.log(`  First error: ${webglErrors[0]}`);
      }

      expect(webglErrors).toEqual([]);
      // Order matters: a real GL error must be the error you see, so the flush
      // verdict comes second — it fails only when the page is responsive and
      // the renderer still is not drawing, and otherwise just says what it saw.
      await reportFlushVerdict(page, flush);
    }
  });

  test('should check WebGL state for errors programmatically', async ({ page }) => {
    await page.goto(`/?src=${DATASETS[0]}&debug`);
    await waitForLuxarReady(page);

    // Query WebGL error state directly. Bounded (#1651) not because of what
    // this probe does but because of how it gets there: a saturated rAF loop
    // starves the evaluate round trip itself for tens of seconds (a trivial
    // `() => 'ok'` evaluate was measured unanswered 5 s twelve times running),
    // and a bare evaluate can only be stopped by the whole test budget.
    const glProbe = page.evaluate(() => {
      const canvas = document.querySelector('canvas') as HTMLCanvasElement;
      if (!canvas) return { error: 'No canvas' };

      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      if (!gl) return { error: 'No WebGL context' };

      const errors: string[] = [];
      const errorCodes: { [key: number]: string } = {
        [gl.NO_ERROR]: 'NO_ERROR',
        [gl.INVALID_ENUM]: 'INVALID_ENUM',
        [gl.INVALID_VALUE]: 'INVALID_VALUE',
        [gl.INVALID_OPERATION]: 'INVALID_OPERATION',
        [gl.INVALID_FRAMEBUFFER_OPERATION]: 'INVALID_FRAMEBUFFER_OPERATION',
        [gl.OUT_OF_MEMORY]: 'OUT_OF_MEMORY',
        [gl.CONTEXT_LOST_WEBGL]: 'CONTEXT_LOST_WEBGL',
      };

      // Check for errors (calling getError clears the error)
      let errorCode = gl.getError();
      let safety = 0;
      while (errorCode !== gl.NO_ERROR && safety++ < 100) {
        errors.push(errorCodes[errorCode] || `UNKNOWN_ERROR(${errorCode})`);
        errorCode = gl.getError();
      }

      return {
        hasErrors: errors.length > 0,
        errors,
        contextLost: gl.isContextLost(),
      };
    });
    const glErrors = await evaluateWithDeadline<Awaited<typeof glProbe> | null>(
      glProbe,
      PROBE_TIMEOUT_MS,
      null
    );

    if (glErrors === null) {
      // Never a silent pass: an unanswered probe means this detector did
      // not run, which is not the same as "no GL errors".
      throw new Error(
        `The GL-state probe went unanswered for ${PROBE_TIMEOUT_MS} ms, so this detector could ` +
          'NOT run. The page keeps rendering while a saturated rAF loop starves the evaluate round ' +
          'trip, so the probe is never scheduled. See issue #1651.'
      );
    }

    console.log('WebGL State:', glErrors);

    expect(glErrors.hasErrors).toBe(false);
    expect(glErrors.contextLost).toBe(false);
  });

  test('should detect vertex buffer size mismatches', async ({ page }) => {
    const vertexBufferErrors: string[] = [];

    page.on('console', (msg) => {
      const text = msg.text();
      if (text.toLowerCase().includes('vertex buffer') && text.includes('big enough')) {
        vertexBufferErrors.push(text);
      }
    });

    await page.goto(`/?src=${DATASETS[0]}&debug`);
    await waitForLuxarReady(page);

    // Drive several render ticks to expose the issue, confirming each one
    // against the renderer's counter — a 10x renderOnce() loop inside one
    // evaluate only re-arms the idle timer, and the previous
    // `waitForNextRender(page, 10)` could fall silently through to its
    // best-effort fallback.
    const flush = await flushRenderTicks(page, 10);

    if (vertexBufferErrors.length > 0) {
      console.log('\n🚨 CRITICAL: Vertex buffer errors detected!');
      console.log(`  Count: ${vertexBufferErrors.length}`);
      console.log('  This indicates incorrect buffer sizing in the renderer');
      console.log(`  First error: ${vertexBufferErrors[0]}\n`);
    }

    // This should be ZERO
    expect(vertexBufferErrors).toEqual([]);
    // Order matters: a real buffer error must be the error you see, so the
    // flush verdict comes second — it fails only when the page is responsive
    // and the renderer still is not drawing, and otherwise just says what it
    // saw.
    await reportFlushVerdict(page, flush);
  });

  test('should verify all geometry buffers are correctly sized', async ({ page }) => {
    await page.goto(`/?src=${DATASETS[0]}&debug`);
    await waitForLuxarReady(page);

    // Bounded (#1651): a scene traversal is cheap, but a saturated rAF loop
    // starves the evaluate round trip regardless of what it asks for, and a
    // bare evaluate has no timeout of its own.
    const bufferProbe = page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const issues: Array<{ name: string; issue: string }> = [];

      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'points') {
          const geom = obj.geometry;
          // Per-point data is texture-backed: an RGBA32F element texture
          // holds 12 floats (3 texels) per point, and the only per-instance
          // data is the double-buffered ordering pair aSortedIndex /
          // aSortedIndexB. Malformed geometry means a missing element
          // texture, a texel buffer too small for the visible instance
          // count, or a missing/under-sized/mismatched ordering buffer.
          const texData = geom.userData?.elementTexture?.image?.data;
          const instanceCount = geom.instanceCount || 0;

          if (!texData) {
            issues.push({ name: obj.name, issue: 'Missing element texture' });
          } else {
            const texelCapacity = Math.floor(texData.length / 12);
            if (texelCapacity < instanceCount) {
              issues.push({
                name: obj.name,
                issue: `Texel capacity (${texelCapacity}) < instance count (${instanceCount})`,
              });
            }
            if (texData.length % 4 !== 0) {
              issues.push({
                name: obj.name,
                issue: `Element texture length (${texData.length}) is not a whole number of RGBA texels`,
              });
            }
          }

          const sortedIndex = geom.attributes.aSortedIndex;
          if (!sortedIndex) {
            issues.push({ name: obj.name, issue: 'Missing aSortedIndex attribute' });
          } else {
            if (sortedIndex.count < instanceCount) {
              issues.push({
                name: obj.name,
                issue: `aSortedIndex count (${sortedIndex.count}) < instance count (${instanceCount})`,
              });
            }
            if (sortedIndex.itemSize !== 1) {
              issues.push({
                name: obj.name,
                issue: `aSortedIndex itemSize is ${sortedIndex.itemSize}, expected 1`,
              });
            }
          }

          // The back buffer must exist and match: every shader references
          // both names (WebGPU throws on a referenced-but-absent attribute),
          // and three derives `_maxInstanceCount` from the SMALLEST
          // instanced attribute, so an under-sized back buffer would
          // silently clamp the draw.
          const sortedIndexB = geom.attributes.aSortedIndexB;
          if (!sortedIndexB) {
            issues.push({ name: obj.name, issue: 'Missing aSortedIndexB attribute' });
          } else if (sortedIndex && sortedIndexB.count !== sortedIndex.count) {
            issues.push({
              name: obj.name,
              issue: `aSortedIndexB count (${sortedIndexB.count}) != aSortedIndex count (${sortedIndex.count})`,
            });
          }
        }
      });

      return issues;
    });
    const bufferInfo = await evaluateWithDeadline<Awaited<typeof bufferProbe> | null>(
      bufferProbe,
      PROBE_TIMEOUT_MS,
      null
    );

    if (bufferInfo === null) {
      // An assumed-empty issue list would read as a pass; say plainly that
      // nothing was verified instead.
      throw new Error(
        `The geometry-buffer probe went unanswered for ${PROBE_TIMEOUT_MS} ms, so this test could ` +
          'NOT verify buffer sizing. The main thread is saturated and starving the evaluate round ' +
          'trip. See issue #1651.'
      );
    }

    if (bufferInfo.length > 0) {
      console.log('\n🚨 Buffer sizing issues found:');
      bufferInfo.forEach((issue) => {
        console.log(`  [${issue.name}] ${issue.issue}`);
      });
      console.log('');
    }

    expect(bufferInfo).toEqual([]);
  });
});

test.describe('WebGL Error Detection - All Datasets', () => {
  // Test that EVERY dataset renders without errors
  for (const dataset of DATASETS) {
    const datasetName = dataset.split('/').pop()?.replace('.zarr', '') || 'unknown';

    test(`should render ${datasetName} without WebGL errors`, async ({ page }) => {
      const webglErrors: string[] = [];

      page.on('console', (msg) => {
        if (msg.type() === 'error' || msg.text().includes('GL_')) {
          webglErrors.push(msg.text());
        }
      });

      await page.goto(`/?src=${dataset}&debug`);
      await waitForLuxarReady(page);

      await page.waitForFunction(
        () => {
          const info = (window as any).__luxarDebug?.renderer?.info;
          return (info?.render?.frame ?? info?.frame ?? 0) > 2;
        },
        { timeout: 10000 }
      );

      // Drive several more render ticks to flush delayed GL errors
      const flush = await flushRenderTicks(page, 3);

      const glErrors = webglErrors.filter((err) => err.includes('GL_INVALID'));

      if (glErrors.length > 0) {
        console.log(`\n❌ ${datasetName}: ${glErrors.length} WebGL errors`);
        console.log(`   First: ${glErrors[0]}`);
      }

      expect(glErrors).toEqual([]);
      // Order matters: a real GL error must be the error you see, so the flush
      // verdict comes second — it fails only when the page is responsive and
      // the renderer still is not drawing, and otherwise just says what it saw.
      await reportFlushVerdict(page, flush);
    });
  }
});

// ---------------------------------------------------------------------------
// File-local scaffolding.
//
// Kept at the FOOT of the file: the constant below is only read from inside
// test bodies (which run after the module has been evaluated), so nothing needs
// to move up. The tick-flushing scaffolding that used to live here now lives in
// `./render-ticks`, where it is unit-testable against a fake page.
// ---------------------------------------------------------------------------

/**
 * Deadline for the two DETECTOR probes in this file (the `gl.getError()` drain
 * and the geometry-buffer traversal). `page.evaluate` carries no timeout of its
 * own — neither `actionTimeout` nor `setDefaultTimeout` reaches it — so without
 * an explicit deadline an unanswered probe consumes the whole test budget
 * (#1651). These two are deliberately the most generous bound in the file
 * because their ANSWERS are what the tests assert on: cutting one short buys
 * nothing but a lost detector, whereas the counter probes in `./render-ticks`
 * only downgrade a verdict to a warning and so take a smaller bound. 15 s is a
 * generous allowance for merely being SCHEDULED late behind a deep backlog of
 * already-queued main-thread work — the measured producer of one here is the
 * OPFS write path (`OPFS timeout: set(...) exceeded 10000ms`, which #1647
 * addresses), and it starved this file on an IDLE box, so the worst case is
 * worse than anything measured here — while still being small enough that the
 * failure names the probe instead of arriving as an opaque test timeout inside
 * the 60 s per-test budget. It is deliberately NOT proof that the page is dead:
 * that starvation was measured to outlast any bound that fits in the test
 * budget (a trivial evaluate unanswered for 5 s, twelve times in a row, while
 * the page went on rendering), so each of the two call sites above says only
 * that its detector could not run.
 */
const PROBE_TIMEOUT_MS = 15000;
