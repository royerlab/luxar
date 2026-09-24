/**
 * Mesh reveal ladder end-to-end — the guard the series never had.
 *
 * Every defect the mesh reveal ladder shipped with was found either by adversarial
 * review or by loading a real store in a browser by hand. None was caught by a unit
 * test, and the unit suites were green through all of them:
 *
 * - the committed prefix sized every GPU buffer, so each level rebound `position` /
 *   `color` / `normal` / the index and orphaned the previous level's GL buffers,
 *   which `three` frees from nowhere (#1521);
 * - fixing that left `color` with no content-refresh path, so every vertex a later
 *   level revealed kept the zero fill it was born with — `alpha = 0`, which is the
 *   whole coverage term for a mesh, so the revealed triangles were INVISIBLE and
 *   stayed so once the ladder completed (#1522);
 * - the parent group's descriptive attrs were missing, so the node bound no
 *   `normal` attribute at all and drew faceted and forced double-sided.
 *
 * What those have in common is that they are invisible below this layer. A unit test
 * can assert an attribute object was not rebound — and one did, and it passed
 * *because of* #1522, since an attribute that is never updated is maximally
 * identical. Only a real load can say whether the surface that arrives is the
 * surface that was authored.
 *
 * Fixture: `test_mesh_reveal_ladder.luxar.zarr` — a 4-level reveal of a subdivided
 * icosphere (1,280 faces; 1,101 ladder vertices against 642 in the source, the rest
 * being the boundary duplication a face-partition costs) beside an UNLADDERED copy
 * of the same surface. The control is load-bearing: each of the bugs above was a
 * DIFFERENCE between the two, and none is legible without something correct in the
 * same scene under the same camera.
 *
 * Deliberately NOT here, and each for a stated reason:
 *
 * - **the reveal ORDER** — that levels interleave stacked timepoints (#1514) and
 *   that each prefix is one connected patch (#1507). Properties of the authored
 *   permutation, pinned deterministically by Python unit tests; asserting them here
 *   would mean catching a mid-reveal frame, which is a timing race.
 * - **the buffer ORPHANING of #1521 itself.** This is the one worth spelling out,
 *   because a test for it was written here first and proved VACUOUS: capacity sizing
 *   and prefix sizing converge to the same buffer at completion, so no end-state
 *   observable distinguishes them — the symptom is a REBIND, which leaves no trace
 *   once the ladder is whole. Verified by mutation: sourcing capacity from
 *   `data.vertexCount` in the commit left all six tests this file then held — the
 *   five below plus that candidate — green. Its real guard
 *   is the unit-level assertion that every attribute OBJECT survives three growing
 *   commits (`mesh-geometry.test.ts`), which is exactly the history this layer
 *   cannot see. What survives here is the consequence that IS observable — the
 *   colour tail below, which prefix sizing plus a missing refresh made invisible.
 *
 * Every test in this file has been mutation-verified against the defect it names.
 *
 * @module tests/e2e/mesh-reveal-ladder.spec
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady, getLuxarState, getWebGLErrors, renderOnce } from './helpers';

const FIXTURES_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';
const LADDER = `${FIXTURES_BASE}/test_mesh_reveal_ladder.luxar.zarr`;

/** The authored totals, from the fixture generator. */
const SOURCE_FACES = 1280;
const LADDER_VERTICES = 1101;
const PLAIN_VERTICES = 642;
const LEVELS = 4;

/** What the page reports per mesh node, read straight off the scene graph. */
interface LadderProbe {
  name: string;
  /** `loadedLODCount` / `totalLODCount`, or null on an unladdered node. */
  loadedLODs: number | null;
  totalLODs: number | null;
  triangles: number;
  /** `position.count` — the CAPACITY since #1521, not the committed prefix. */
  positionCount: number;
  colorCount: number;
  colorItemSize: number | null;
  hasNormalAttr: boolean;
  ladderComplete: boolean | null;
  /** Whether the commit stamped an energy fraction at all (see #1521/§9.1). */
  energyStamped: boolean;
  /** RGBA at the FIRST and LAST vertex of the colour buffer. */
  colorFirst: number[] | null;
  colorLast: number[] | null;
}

/**
 * Probe both mesh nodes through the live scene graph.
 *
 * Reads `app.sceneManager.scene` rather than `getLuxarState()` because the state
 * summary reports counts, and what these assertions need is the BUFFERS — the
 * attribute a node actually bound, and the bytes in it at the far end.
 */
async function probeLadder(page: import('@playwright/test').Page): Promise<LadderProbe[]> {
  return page.evaluate(() => {
    const debug = (
      window as unknown as {
        __luxarDebug?: { app?: { sceneManager?: { scene?: unknown } } };
      }
    ).__luxarDebug;
    const root = debug?.app?.sceneManager?.scene as
      { traverse(cb: (o: Record<string, unknown>) => void): void } | undefined;
    const out: unknown[] = [];
    if (!root) return out as never;
    root.traverse((object) => {
      const o = object as {
        name?: string;
        userData?: Record<string, unknown>;
        geometry?: {
          getAttribute(
            n: string
          ): { count: number; itemSize: number; array: ArrayLike<number> } | undefined;
        };
      };
      if (o.userData?.nodeType !== 'mesh') return;
      const geometry = o.geometry;
      const color = geometry?.getAttribute('color');
      const loader = (o.userData.loader ?? {}) as {
        loadedLODCount?: number;
        totalLODCount?: number;
      };
      const at = (v: number): number[] | null => {
        if (!color) return null;
        const k = color.itemSize;
        return Array.from({ length: k }, (_, c) => color.array[v * k + c]);
      };
      out.push({
        name: (o.name ?? '').replace(/^\//, ''),
        loadedLODs: loader.loadedLODCount ?? null,
        totalLODs: loader.totalLODCount ?? null,
        triangles: (o.userData.visibleTriangleCount as number) ?? 0,
        positionCount: geometry?.getAttribute('position')?.count ?? 0,
        colorCount: color?.count ?? 0,
        colorItemSize: color?.itemSize ?? null,
        hasNormalAttr: !!geometry?.getAttribute('normal'),
        ladderComplete: (o.userData.committedLadderComplete as boolean) ?? null,
        energyStamped: 'committedEnergyFraction' in (o.userData ?? {}),
        colorFirst: at(0),
        colorLast: color ? at(color.count - 1) : null,
      });
    });
    return out as never;
  }) as Promise<LadderProbe[]>;
}

/** Wait until the laddered node reports every level committed. */
async function waitForRevealComplete(
  page: import('@playwright/test').Page,
  timeout = 60000
): Promise<void> {
  await page.waitForFunction(
    (want) => {
      const debug = (
        window as unknown as {
          __luxarDebug?: { app?: { sceneManager?: { scene?: unknown } } };
        }
      ).__luxarDebug;
      const root = debug?.app?.sceneManager?.scene as
        { traverse(cb: (o: Record<string, unknown>) => void): void } | undefined;
      if (!root) return false;
      let done = false;
      root.traverse((object) => {
        const o = object as { name?: string; userData?: Record<string, unknown> };
        if (o.name !== '/laddered') return;
        const loader = (o.userData?.loader ?? {}) as { loadedLODCount?: number };
        done = o.userData?.committedLadderComplete === true && loader.loadedLODCount === want;
      });
      return done;
    },
    LEVELS,
    { timeout }
  );
}

test.describe('Mesh reveal ladder', () => {
  test.describe.configure({ timeout: 180000 });

  test('the revealed surface converges to the unladdered control', async ({ page }) => {
    await page.goto(`/?src=${LADDER}&debug`);
    await waitForLuxarReady(page);
    await waitForRevealComplete(page);

    const byName = new Map((await probeLadder(page)).map((m) => [m.name, m]));
    const laddered = byName.get('laddered');
    const plain = byName.get('plain');
    expect(laddered, 'the laddered node should be reported').toBeDefined();
    expect(plain, 'the unladdered control should be reported').toBeDefined();

    // Same surface, so the same drawn triangle count — the levels are a PARTITION
    // of the faces, and a ladder that dropped or duplicated one would show here.
    expect(laddered!.triangles).toBe(SOURCE_FACES);
    expect(plain!.triangles).toBe(SOURCE_FACES);
    expect(laddered!.loadedLODs).toBe(LEVELS);
    expect(laddered!.totalLODs).toBe(LEVELS);

    // The ladder holds MORE vertices than the source: a face-partition duplicates
    // every vertex on a cut. That the two differ is the point — equal counts would
    // mean the levels were not independently re-indexed.
    expect(laddered!.positionCount).toBe(LADDER_VERTICES);
    expect(plain!.positionCount).toBe(PLAIN_VERTICES);
    expect(laddered!.positionCount).toBeGreaterThan(plain!.positionCount);
  });

  test('the colour of the LAST revealed vertex is written and opaque', async ({ page }) => {
    // #1522, and the assertion the identity-only unit test could not make. The
    // fixture is opaque everywhere by construction, so any transparency here is a
    // slot the ladder never wrote — and for padded RGB / RGBA that slot is
    // `(0,0,0,0)`, i.e. an invisible triangle rather than a merely wrong one.
    await page.goto(`/?src=${LADDER}&debug`);
    await waitForLuxarReady(page);
    await waitForRevealComplete(page);

    const laddered = (await probeLadder(page)).find((m) => m.name === 'laddered')!;
    expect(laddered.colorItemSize).toBe(4);
    const last = laddered.colorLast!;
    const first = laddered.colorFirst!;

    // Alpha first: it alone decides visibility.
    expect(last[3], 'the last revealed vertex must be opaque').toBeGreaterThan(0.99);
    expect(first[3]).toBeGreaterThan(0.99);
    // And the RGB must be real colour, not the zero fill. The fixture ramps hue
    // with position, so a written tail cannot be all-zero.
    expect(last.slice(0, 3).some((c) => c > 0.01)).toBe(true);
  });

  test('the parent carries the normal frame, so the ladder shades like the control', async ({
    page,
  }) => {
    // The ladder's parent group is the only thing a reader sees — its
    // `additive_<i>` subgroups are pruned from the scene graph — and a mesh
    // geometry's attribute SET is fixed once, at creation, from the parent's
    // `has_normals`. A parent carrying only counts binds no `normal` attribute, and
    // the levels' normals can then never reach the GPU however many arrive: the
    // node draws faceted and forced double-sided beside a smooth control. Both
    // nodes are authored `shading="smooth"` with the same normals, so they must
    // agree.
    await page.goto(`/?src=${LADDER}&debug`);
    await waitForLuxarReady(page);
    await waitForRevealComplete(page);

    const byName = new Map((await probeLadder(page)).map((m) => [m.name, m]));
    expect(byName.get('laddered')!.hasNormalAttr).toBe(true);
    expect(byName.get('plain')!.hasNormalAttr).toBe(true);
  });

  test('a reveal carries NO energy stamp, while the control carries one', async ({ page }) => {
    // §9.1's hard rule, end to end. The LOD fade divides committed brightness by
    // e(k) to compensate an incomplete EMISSIVE ladder — right for a coarse prefix
    // of a splat cloud, and exactly wrong for a partial object at FULL brightness,
    // which is what a reveal prefix is. So the stamp must be ABSENT here.
    //
    // The control is what stops this passing vacuously: an unladdered mesh IS its
    // complete content, so it stamps 1. If both were unstamped the assertion would
    // prove only that nothing stamps anything.
    await page.goto(`/?src=${LADDER}&debug`);
    await waitForLuxarReady(page);
    await waitForRevealComplete(page);

    const byName = new Map((await probeLadder(page)).map((m) => [m.name, m]));
    expect(byName.get('laddered')!.energyStamped).toBe(false);
    expect(byName.get('plain')!.energyStamped).toBe(true);
  });

  test('the reveal draws without GL errors', async ({ page }) => {
    await page.goto(`/?src=${LADDER}&debug`);
    await waitForLuxarReady(page);
    await waitForRevealComplete(page);
    await renderOnce(page);

    const errors = await getWebGLErrors(page);
    expect(errors, `GL errors during the reveal: ${JSON.stringify(errors)}`).toHaveLength(0);

    // And the scene reports both surfaces' triangles, so neither node was dropped.
    const state = await getLuxarState(page);
    expect(state.totalTriangles).toBe(SOURCE_FACES * 2);
  });
});
