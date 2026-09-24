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
  captureCanvasRGBA,
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
    test.setTimeout(120000);

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
    test.setTimeout(120000);

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
    test.setTimeout(120000);

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
    test.setTimeout(120000);

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
    test.setTimeout(120000);

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
    test.setTimeout(120000);

    // The one assertion no state snapshot can make: `triangleCount > 0` says the
    // geometry committed, not that anything was RASTERIZED. A shader that discarded
    // every fragment — the exact failure mode of a mis-signed derivative normal, an
    // inverted cutout, or a shade term collapsing to zero — satisfies every test above
    // and renders a blank canvas.
    //
    // Read through `getElementPixelStats`, which explicitly captures the post-processed
    // renderer framebuffer. A hand-rolled 2D-context readback can see a cleared WebGL
    // drawing buffer when `preserveDrawingBuffer` is false.
    await page.goto(`/?src=${MESH}&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitForMeshCommitted(page, 4);
    await renderOnce(page);

    const stats = await getElementPixelStats(page, 'canvas#app', 10, 'framebuffer');
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

  test('flying into a mesh fades it out smoothly instead of clipping (#1431)', async ({ page }) => {
    test.slow();

    // Mesh was the one geometry type with no `perspectiveNearFade`: a triangle
    // clipped hard against the near plane while the other three faded. This pins the
    // fixed behaviour where it is observable — in framebuffer pixels, through the
    // production material, on a mesh that came out of the writer.
    //
    // The approach fly-in is driven by SCALING `uNearCull` rather than by moving the
    // camera, and that is a measurement decision rather than a shortcut. The fade is
    // a function of `-viewZ / nearCull` alone, so scaling `nearCull` by s is exactly
    // equivalent to dividing every view depth by s — i.e. to a dolly. What it avoids
    // is everything a positional dolly would confound the measurement WITH: the band
    // is only 0.1% of the scene diagonal wide (`nearCull = 1e-3 · diagonal`), so a
    // dolly would have to be placed to sub-thousandth precision, would fight the
    // near-plane floor (`far / MAX_NEAR_FAR_RATIO`, the same order of magnitude at
    // that range), and would change the mesh's screen COVERAGE at every step — so a
    // frame-mean would be tracking framing, not fade. Here the geometry, the framing
    // and the shade term are pixel-identical between steps and the fade is the only
    // variable.
    //
    // `?dpr=1` is load-bearing, not tidiness: adaptive DPR resizes the drawing
    // buffer whenever it re-times a frame, every resize runs
    // `updateMaterialsForCurrentCamera()`, and that rebroadcast overwrites
    // `uNearCull` with the scene's real value — one un-faded frame in the middle of
    // the sweep. Pinning the ratio is what the param is for (see `?dpr=` in
    // `config/url-params.ts`); without it this test fails roughly four runs in five.
    await page.goto(`/?src=${MESH}&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitForMeshCommitted(page, 4);
    await renderOnce(page);

    // Distance from the camera to the orbit target: the scale the sweep is expressed
    // in, so it needs no knowledge of the fixture's world units. `controls` is the
    // ControlsManager, which owns the concrete orbit/fly/ortho controls and exposes
    // the pivot through `getFocusTarget()` — there is no `.target` on it.
    const targetDistance = await page.evaluate(() => {
      const debug = (
        window as unknown as {
          __luxarDebug: {
            camera: { position: { distanceTo(v: unknown): number } };
            controls: { getFocusTarget(): unknown };
          };
        }
      ).__luxarDebug;
      return debug.camera.position.distanceTo(debug.controls.getFocusTarget());
    });
    expect(targetDistance).toBeGreaterThan(0);

    /**
     * Write `uNearCull` on every MESH material in the scene. Identified by
     * `uAmbient` — the shade floor no other geometry type has — so this cannot
     * silently start driving a point/line/gsplat material instead.
     *
     * Safe to write directly only because the DPR is pinned: the manager rebroadcasts
     * on resize, load, or an ortho zoom change, and with adaptive DPR live the first
     * of those fires on its own mid-sweep (see the `?dpr=1` note above).
     */
    const setNearCull = async (value: number): Promise<number> =>
      page.evaluate((v) => {
        const debug = (
          window as unknown as {
            __luxarDebug: { scene: { traverse(cb: (o: unknown) => void): void } };
          }
        ).__luxarDebug;
        let touched = 0;
        debug.scene.traverse((object) => {
          const material = (object as { material?: unknown }).material;
          for (const m of Array.isArray(material) ? material : [material]) {
            const uniforms = (m as { uniforms?: Record<string, { value: unknown }> } | undefined)
              ?.uniforms;
            if (!uniforms?.uNearCull || !uniforms.uAmbient) continue;
            uniforms.uNearCull.value = v;
            touched++;
          }
        });
        return touched;
      }, value);

    const meanChannel = async (): Promise<number> => {
      const frame = await captureCanvasRGBA(page, 'canvas#app', 'framebuffer');
      let sum = 0;
      for (let i = 0; i < frame.rgba.length; i += 4) {
        sum += frame.rgba[i] + frame.rgba[i + 1] + frame.rgba[i + 2];
      }
      return sum / ((frame.rgba.length / 4) * 3);
    };

    // 0.4 → 1.2 × the target distance, in steps of 0.05. At 0.4 the whole surface
    // still sits beyond the band's outer edge (`2 · nearCull`) and the fade is a flat
    // 1.0; by 1.2 every fragment is inside the reject region and the mesh is gone.
    //
    // The step is HALF what it was. At 0.1 the transition spanned few enough samples
    // that one of them carried 34.5% of the whole drop — real headroom under the 50%
    // bound below, but not much, and that bound is the assertion separating a
    // smoothstep from a hard clip. Halving the step roughly doubles the samples across
    // the ramp and brings the largest single one down to 18.7% (measured; the full
    // per-step table is in the NO-POP comment below).
    const factors = Array.from({ length: 17 }, (_, i) => Number((0.4 + i * 0.05).toFixed(2)));
    const means: number[] = [];
    for (const f of factors) {
      const touched = await setNearCull(f * targetDistance);
      expect(touched, 'no mesh material carries a uNearCull uniform').toBeGreaterThan(0);
      await renderOnce(page);
      means.push(await meanChannel());
    }

    const total = means[0] - means[means.length - 1];
    const trace = means.map((m) => m.toFixed(2)).join(' → ');
    // The two assertions that fail without the fade, and the reason the first is a
    // FRACTION of the starting brightness rather than an absolute delta. Neutralizing
    // `nearFade` to a constant 1.0 in the fragment stage — the whole of the pre-fix
    // behaviour, since the shader then never reads `uNearCull` — was re-measured on
    // this sweep and leaves 21.87 of the starting 23.13 on screen across every step.
    // So an absolute bound like "> 1.0" PASSES on a build with no fade at all: the
    // 1.26 that moves there is unrelated drift, not the fade. Both literals are one
    // machine's numbers; what is portable is the shape — a fade-free build's total is
    // a rounding error next to the starting brightness, and a real fade's is most
    // of it.
    expect(
      total,
      `the mesh did not dim as the near-cull band swept over it: ${trace}`
    ).toBeGreaterThan(0.5 * means[0]);
    expect(
      means[means.length - 1],
      `the fully-faded frame should be essentially black: ${trace}`
    ).toBeLessThan(0.1 * means[0]);

    for (let i = 1; i < means.length; i++) {
      // MONOTONE: a fade that brightened anywhere would mean the ramp is not a
      // function of depth. The 0.25/255 slack absorbs AA and compositing dither.
      expect(means[i], `step ${factors[i]} brightened: ${trace}`).toBeLessThanOrEqual(
        means[i - 1] + 0.25
      );
      // NO POP: a hard clip is one step that takes the whole drop, so no single
      // step of a real fade may take half of it.
      //
      // What the sweep actually measures (deterministic here — three repeats of
      // the trace above were byte-identical), as a percentage of the 22.46 total:
      //
      //   0.3 2.2 11.4 9.4 13.8 17.5 18.7 15.8 1.4 0.9 2.9 3.5 1.8 0.4 0.0 0.0
      //
      // So the largest single step is 18.7% and the ramp is spread over about
      // eight of them. Read the same sweep at the old 0.05→0.1 step (every other
      // sample) and the largest becomes 34.5% — real headroom under the bound, but
      // close enough that a modest change in fixture or framing could push a
      // legitimately smooth fade over it. That is why the step was halved, and it
      // is also the honest reading of this bound: it rejects a CLIP, not every
      // sharp-ish ramp. The drop is genuinely front-loaded — four adjacent steps
      // carry 66% of it — because a smoothstep's slope peaks at the band centre.
      expect(
        means[i - 1] - means[i],
        `step ${factors[i]} is a POP, not a fade — it took ${(((means[i - 1] - means[i]) / total) * 100).toFixed(0)}% of the whole drop: ${trace}`
      ).toBeLessThan(0.5 * total);
    }
  });
});
