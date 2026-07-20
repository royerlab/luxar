/**
 * Lines Blending Modes E2E Tests
 *
 * The lines twin of the points half of blending-modes.spec.ts
 * (three-geometry symmetry). Loads test_lines_blending_modes.luxar.zarr —
 * five crossing polyline layers named `lines_<mode>`, one per canonical
 * blending mode — and asserts the EXACT THREE.js material state each
 * mode must carry per `getCompleteBlendingState`
 * (src/rendering/blending-state.ts), plus a no-console-error smoke render.
 *
 * Fixture: packages/luxar-viewer/tests/fixtures/generate_test_data.py
 * ::generate_lines_blending_modes_test (run `pnpm test:generate-fixtures`).
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  waitForNextRender,
  getWebGLErrors,
  assertNoConsoleErrors,
} from './helpers';
import { EXPECTED_BLEND_STATE as EXPECTED_STATE } from './blending-expected-state';

const FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_lines_blending_modes.luxar.zarr';

test.describe('Lines blending modes (per-mode material state)', () => {
  test.slow();

  // Fail fast with an actionable message instead of an opaque 404:
  // this fixture is Python-generated and NOT covered by the Playwright
  // global-setup (which only checks datasets/examples).
  test.beforeAll(async () => {
    const { existsSync } = await import('node:fs');
    const path = await import('node:path');
    const { fileURLToPath } = await import('node:url');
    const specDir = path.dirname(fileURLToPath(import.meta.url));
    const fixtureDir = path.resolve(
      specDir,
      '../../../tests/fixtures/test_lines_blending_modes.luxar.zarr'
    );
    if (!existsSync(fixtureDir)) {
      throw new Error(
        `Missing fixture ${fixtureDir} — run \`pnpm test:generate-fixtures\` ` +
          'from packages/luxar-viewer/ first.'
      );
    }
  });

  /** Wait until every fixture layer's lines mesh has committed instances. */
  async function waitForLinesCommitted(page: import('@playwright/test').Page): Promise<void> {
    await page.waitForFunction(
      () => {
        const debug = (window as unknown as { __luxarDebug?: { scene?: unknown } }).__luxarDebug;
        if (!debug?.scene) return false;
        let committed = 0;
        (debug.scene as { traverse: (cb: (obj: unknown) => void) => void }).traverse((obj) => {
          const o = obj as {
            userData?: { nodeType?: string };
            geometry?: { instanceCount?: number };
          };
          if (o.userData?.nodeType === 'lines' && (o.geometry?.instanceCount ?? 0) > 0) {
            committed++;
          }
        });
        return committed >= 5;
      },
      undefined,
      { timeout: 30000 }
    );
  }

  /** Read {name → material state} for every lines mesh in the scene. */
  function readLinesMaterialStates(page: import('@playwright/test').Page) {
    return page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const states: Array<{
        name: string;
        blendingMode: string | undefined;
        blending: number;
        blendEquation: number;
        blendSrc: number;
        blendDst: number;
        depthTest: boolean;
        depthWrite: boolean;
        transparent: boolean;
      }> = [];
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'lines' && obj.material) {
          const m = obj.material;
          states.push({
            name: obj.name ?? '',
            blendingMode: m.userData?.blendingMode,
            blending: m.blending,
            blendEquation: m.blendEquation,
            blendSrc: m.blendSrc,
            blendDst: m.blendDst,
            depthTest: m.depthTest,
            depthWrite: m.depthWrite,
            transparent: m.transparent,
          });
        }
      });
      return states;
    });
  }

  test('each lines_<mode> layer carries the exact per-mode THREE blend state', async ({ page }) => {
    await page.goto(`/?src=${FIXTURE}&debug`);
    await waitForLuxarReady(page);
    await waitForLinesCommitted(page);

    const states = await readLinesMaterialStates(page);
    expect(states.length).toBe(6);

    // NOTE: `volumetric` asserts the phase-1 ADDITIVE fallback state
    // (EXPECTED_STATE.volumetric mirrors the additive row) while
    // userData.blendingMode keeps 'volumetric' — lines implement the
    // emission–absorption math in phase 4 (VOLUMETRIC_BLENDING_SPEC.md).
    for (const [mode, expected] of Object.entries(EXPECTED_STATE)) {
      const state = states.find((s) => s.name.includes(`lines_${mode}`));
      expect(state, `lines_${mode} mesh not found in scene`).toBeTruthy();
      expect(state!.blendingMode, `lines_${mode}: userData.blendingMode`).toBe(mode);
      expect(state!.blending, `lines_${mode}: blending`).toBe(expected.blending);
      expect(state!.blendEquation, `lines_${mode}: blendEquation`).toBe(expected.blendEquation);
      expect(state!.blendSrc, `lines_${mode}: blendSrc`).toBe(expected.blendSrc);
      expect(state!.blendDst, `lines_${mode}: blendDst`).toBe(expected.blendDst);
      expect(state!.depthTest, `lines_${mode}: depthTest`).toBe(expected.depthTest);
      expect(state!.depthWrite, `lines_${mode}: depthWrite`).toBe(expected.depthWrite);
      expect(state!.transparent, `lines_${mode}: transparent`).toBe(expected.transparent);
    }
  });

  test('renders all six line blending modes without WebGL/console errors', async ({ page }) => {
    await page.goto(`/?src=${FIXTURE}&debug`);
    await waitForLuxarReady(page);
    await waitForLinesCommitted(page);

    // Force multiple renders to flush any deferred errors.
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      for (let i = 0; i < 5; i++) debug.renderOnce();
    });
    await waitForNextRender(page);

    const webglErrors = await getWebGLErrors(page);
    expect(webglErrors.length).toBe(0);
    await assertNoConsoleErrors(page);
  });
});
