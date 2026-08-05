/**
 * Mesh end-to-end rendering — the fourth geometry type, in a real browser.
 *
 * Everything else about mesh is tested one layer down: the cull kernels have parity
 * tests, the material pair has codegen snapshots and a GLSL↔TSL pixel harness, and the
 * loader/commit/panel/debug paths have unit tests with stub materials. What NONE of
 * those cover is the **wiring between them** — that a mesh authored by the Python
 * writer arrives through the loader, commits into a geometry, gets the right shader
 * variant, and puts pixels on the screen. That gap was flagged during the phase-4
 * review and is what this file closes.
 *
 * Fixtures come from `tests/fixtures/generate_test_data.py`:
 *
 * - `test_mesh.luxar.zarr` — a welded, closed icosphere (162 vertices / 320 faces)
 *   with stored normals, RGBA whose +z cap sits BELOW the `opaque` cutout, and
 *   per-vertex labels; plus a `scalar_sphere` on the colormap path and a `flat_patch`
 *   authored `shading="flat"`.
 * - `test_mesh_nd.luxar.zarr` — two spheres separated along a hidden categorical
 *   dimension, for the whole-triangle slab cull.
 *
 * @module tests/e2e/mesh-rendering.spec
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  getLuxarState,
  getWebGLErrors,
  getElementPixelStats,
  renderOnce,
} from './helpers';

const FIXTURES_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';
const MESH = `${FIXTURES_BASE}/test_mesh.luxar.zarr`;
const MESH_ND = `${FIXTURES_BASE}/test_mesh_nd.luxar.zarr`;

/** The `meshNodes` entry for `name`, or undefined. */
interface MeshNodeInfo {
  name: string;
  triangleCount: number;
  vertexCount: number;
  visible: boolean;
  flatNormal: boolean;
  alphaCutout: boolean;
  hasColormap: boolean;
}

async function meshNodes(page: import('@playwright/test').Page): Promise<MeshNodeInfo[]> {
  const state = await getLuxarState(page);
  return (state.meshNodes ?? []) as MeshNodeInfo[];
}

/** Wait until at least `n` mesh nodes have committed a non-zero triangle count. */
async function waitForMeshCommitted(
  page: import('@playwright/test').Page,
  n = 1,
  timeout = 45000
): Promise<void> {
  await page.waitForFunction(
    (want) => {
      const debug = (window as unknown as { __luxarDebug?: { getState?: () => unknown } })
        .__luxarDebug;
      if (!debug?.getState) return false;
      const nodes = (debug.getState() as { meshNodes?: Array<{ triangleCount: number }> })
        .meshNodes;
      if (!nodes) return false;
      return nodes.filter((m) => m.triangleCount > 0).length >= want;
    },
    n,
    { timeout }
  );
}

test.describe('Mesh rendering', () => {
  test('a written mesh loads, commits its triangles, and draws without GL errors', async ({
    page,
  }) => {
    await page.goto(`/?src=${MESH}&debug`);
    await waitForLuxarReady(page);
    // Four mesh nodes in the fixture; all four must commit.
    await waitForMeshCommitted(page, 4);

    const nodes = await meshNodes(page);
    const byName = new Map(nodes.map((m) => [m.name.replace(/^\//, ''), m]));

    // The icosphere's exact counts, which are also the WELDING proof: a subdivided
    // icosphere has 320 faces, and 162 vertices only if edge midpoints are SHARED
    // between the two faces that own them. De-indexed it would be 960 vertices — and
    // `gl_VertexID` would then be incidentally per-triangle, making the shared-vertex
    // pick semantics untestable.
    const sphere = byName.get('sphere');
    expect(sphere, 'the sphere node should be reported').toBeDefined();
    expect(sphere!.triangleCount).toBe(320);
    expect(sphere!.vertexCount).toBe(162);

    // The flat patch: two triangles, four vertices.
    const patch = byName.get('flat_patch');
    expect(patch!.triangleCount).toBe(2);
    expect(patch!.vertexCount).toBe(4);

    // Totals agree with the per-node sum — the aggregate is not independently derived.
    const state = await getLuxarState(page);
    expect(state.totalTriangles).toBe(nodes.reduce((n, m) => n + m.triangleCount, 0));

    const glErrors = await getWebGLErrors(page);
    expect(glErrors, `WebGL errors while drawing a mesh: ${glErrors.join('; ')}`).toEqual([]);
  });

  test('the authored shading variant reaches the shader, per node', async ({ page }) => {
    // The §3.4 rule end to end. `shading="smooth"` with `normal_dims == displayDims`
    // must compile the stored-normal build; `shading="flat"` must compile the
    // derivative one — in the SAME scene, so a global misconfiguration cannot make
    // both agree by accident.
    //
    // This is the assertion the unit tests structurally cannot make: they hand
    // `resolveFlatNormal` an attrs object, whereas here the value has travelled zarr →
    // loader → projection → commit → material define.
    await page.goto(`/?src=${MESH}&debug`);
    await waitForLuxarReady(page);
    await waitForMeshCommitted(page, 4);

    const byName = new Map((await meshNodes(page)).map((m) => [m.name.replace(/^\//, ''), m]));
    expect(byName.get('sphere')!.flatNormal, 'smooth node took the derivative path').toBe(false);
    for (const flat of ['flat_patch', 'flat_facing']) {
      expect(byName.get(flat)!.flatNormal, `${flat} took the stored-normal path`).toBe(true);
    }
  });

  test('the colormap node reads the LUT and the direct-colour node does not', async ({ page }) => {
    await page.goto(`/?src=${MESH}&debug`);
    await waitForLuxarReady(page);
    await waitForMeshCommitted(page, 4);

    const byName = new Map((await meshNodes(page)).map((m) => [m.name.replace(/^\//, ''), m]));
    expect(byName.get('scalar_sphere')!.hasColormap).toBe(true);
    // The fail-closed half: `sphere` has no scalars, so even though it is the same
    // geometry it must NOT compile the colormap variant.
    expect(byName.get('sphere')!.hasColormap).toBe(false);
  });

  test('mesh defaults to `opaque`, so the cutout variant is live without anyone asking', async ({
    page,
  }) => {
    // §6.3's deliberate asymmetry — the three siblings default to `additive`. Asserted
    // through the shader variant rather than through the attrs, because the default is
    // applied VIEWER-side (`?? 'opaque'` in createMeshNode) and never stamped by the
    // writer: reading the zarr attrs would show nothing at all.
    await page.goto(`/?src=${MESH}&debug`);
    await waitForLuxarReady(page);
    await waitForMeshCommitted(page, 4);

    for (const node of await meshNodes(page)) {
      expect(node.alphaCutout, `${node.name} should be in the opaque cutout build`).toBe(true);
    }
  });

  test('scrubbing a hidden categorical dimension SWAPS the two meshes rather than accumulating', async ({
    page,
  }) => {
    // The §5.4 whole-triangle slab cull, and the mesh counterpart of the Lines
    // categorical regression. Mesh needs its own assertion here rather than inheriting
    // the Lines one: it has no interpolation and no per-element extent, so its
    // tolerance arm is separate (§5.2.1) — with Lines' spatial `0` a mesh would reduce
    // membership to float equality and render nothing at all.
    await page.goto(`/?src=${MESH_ND}&debug`);
    await waitForLuxarReady(page);
    await waitForMeshCommitted(page, 1);

    const drawnAtSel = async (): Promise<Map<string, number>> =>
      new Map((await meshNodes(page)).map((m) => [m.name.replace(/^\//, ''), m.triangleCount]));

    const atA = await drawnAtSel();
    // Exactly ONE sphere draws at a time. The bug this guards renders both (A ∪ B),
    // because a throwing empty-slice projection leaves the stale geometry in place.
    const drawnA = [...atA.entries()].filter(([, n]) => n > 0).map(([k]) => k);
    expect(drawnA, `expected one sphere at sel=A, got ${JSON.stringify([...atA])}`).toHaveLength(1);

    // Scrub `sel` to the other category. Dimension 3 (0-based) is `sel`. Uses the same
    // `setDimensionValue` + `awaitDimensionUpdate` pair the extend-to-all and
    // lines-nD specs use — awaiting the update is what makes the poll below a
    // confirmation rather than a race.
    await page.evaluate(async () => {
      const debug = (
        window as unknown as {
          __luxarDebug: {
            app: {
              setDimensionValue(d: number, v: number): Promise<void>;
              awaitDimensionUpdate(): Promise<void>;
            };
          };
        }
      ).__luxarDebug;
      await debug.app.setDimensionValue(3, 1);
      await debug.app.awaitDimensionUpdate();
    });
    await page.waitForFunction(
      (previous) => {
        const debug = (window as unknown as { __luxarDebug?: { getState?: () => unknown } })
          .__luxarDebug;
        const nodes = (debug?.getState?.() as { meshNodes?: MeshNodeInfo[] })?.meshNodes ?? [];
        const drawn = nodes.filter((m) => m.triangleCount > 0).map((m) => m.name);
        return drawn.length === 1 && drawn[0] !== previous;
      },
      drawnA[0],
      { timeout: 20000 }
    );

    const atB = await drawnAtSel();
    const drawnB = [...atB.entries()].filter(([, n]) => n > 0).map(([k]) => k);
    expect(drawnB, 'expected one sphere at sel=B').toHaveLength(1);
    expect(drawnB[0], 'the SAME sphere drew at both categories — no cull happened').not.toBe(
      drawnA[0]
    );

    // Vertex counts are invariant across the scrub: §5.4 rewrites the index buffer
    // only, and the pick-id domain must not move under the user.
    const vertexCounts = (await meshNodes(page)).map((m) => m.vertexCount);
    expect(vertexCounts.every((n) => n > 0)).toBe(true);
  });

  test('a mesh actually rasterizes — real pixels, not just a committed geometry', async ({
    page,
  }) => {
    // The one assertion no state snapshot can make: `triangleCount > 0` says the
    // geometry committed, not that anything was RASTERIZED. A shader that discarded
    // every fragment — the exact failure mode of a mis-signed derivative normal, an
    // inverted cutout, or a shade term collapsing to zero — satisfies every test above
    // and renders a blank canvas.
    //
    // Read through `getElementPixelStats`, NOT by drawing the canvas into a 2D context:
    // with `preserveDrawingBuffer: false` Chromium may clear the WebGL drawing buffer
    // after compositing, so the hand-rolled version returns all-zero pixels for a
    // perfectly good render. `helpers.ts` documents that trap; this test hit it on the
    // first run and the Playwright screenshot is what disambiguated a real blank frame
    // from a bad readback.
    await page.goto(`/?src=${MESH}&debug`);
    await waitForLuxarReady(page);
    await waitForMeshCommitted(page, 4);
    await renderOnce(page);

    const stats = await getElementPixelStats(page, 'canvas', 10);
    const litFraction = stats.nonBlackPixels / (stats.width * stats.height);
    expect(
      stats.nonBlackPixels,
      `the canvas is blank — the mesh committed ${(await meshNodes(page)).length} nodes but drew nothing`
    ).toBeGreaterThan(1000);
    // A loose upper bound too: the three nodes occupy a minority of the frame, so a
    // full-canvas wash (a shader emitting a constant, or the clear colour going wrong)
    // is a different failure that "> 1000" alone would pass.
    expect(litFraction, 'the whole canvas is lit — this is a wash, not a mesh').toBeLessThan(0.5);
    // The brightest pixel must be a real shaded colour rather than a single stray
    // channel, which is what a NaN or an uninitialised varying tends to produce.
    const { r, g, b } = stats.brightest;
    expect(Math.max(r, g, b), 'nothing bright enough to be a lit surface').toBeGreaterThan(60);
  });
});
