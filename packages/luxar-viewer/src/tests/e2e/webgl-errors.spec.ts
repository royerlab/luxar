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
import { waitForLuxarReady, waitForNextRender } from './helpers';

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

    // Render several more frames to flush any delayed GL errors into the
    // console listener.
    await waitForNextRender(page, 5);

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

      // Render several more frames to flush delayed GL errors
      await waitForNextRender(page, 3);

      if (webglErrors.length > 0) {
        console.log(`\n🚨 WebGL errors in ${dataset}:`);
        console.log(`  Found ${webglErrors.length} errors`);
        console.log(`  First error: ${webglErrors[0]}`);
      }

      expect(webglErrors).toEqual([]);
    }
  });

  test('should check WebGL state for errors programmatically', async ({ page }) => {
    await page.goto(`/?src=${DATASETS[0]}&debug`);
    await waitForLuxarReady(page);

    // Query WebGL error state directly
    const glErrors = await page.evaluate(() => {
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

    // Trigger several renders to expose the issue
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      for (let i = 0; i < 10; i++) {
        debug.renderOnce();
      }
    });

    // Wait for the GL command buffer to drain through several real frames
    await waitForNextRender(page, 10);

    if (vertexBufferErrors.length > 0) {
      console.log('\n🚨 CRITICAL: Vertex buffer errors detected!');
      console.log(`  Count: ${vertexBufferErrors.length}`);
      console.log('  This indicates incorrect buffer sizing in the renderer');
      console.log(`  First error: ${vertexBufferErrors[0]}\n`);
    }

    // This should be ZERO
    expect(vertexBufferErrors).toEqual([]);
  });

  test('should verify all geometry buffers are correctly sized', async ({ page }) => {
    await page.goto(`/?src=${DATASETS[0]}&debug`);
    await waitForLuxarReady(page);

    const bufferInfo = await page.evaluate(() => {
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

      // Render several more frames to flush delayed GL errors
      await waitForNextRender(page, 3);

      const glErrors = webglErrors.filter((err) => err.includes('GL_INVALID'));

      if (glErrors.length > 0) {
        console.log(`\n❌ ${datasetName}: ${glErrors.length} WebGL errors`);
        console.log(`   First: ${glErrors[0]}`);
      }

      expect(glErrors).toEqual([]);
    });
  }
});
