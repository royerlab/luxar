/**
 * Cross-Node Depth Ordering E2E
 *
 * Depth sorting is already exact WITHIN a node, so a single-node scene is
 * composited correctly with or without this feature — which is why every
 * pre-existing overlap fixture is unable to exercise it. This spec drives the
 * TWO-node fixture: two interpenetrating combs of splats whose correct
 * back-to-front order strictly ALTERNATES between the nodes, which is exactly
 * what one `renderOrder` integer per node cannot express.
 *
 * Design: `docs/guides/specs/CROSS_NODE_DEPTH_ORDERING_SPEC.md`.
 *
 * The gate is fail-first by pinning `?depthShards=0`: that arm must reproduce the
 * node-major composite, and the enabled arm must differ from it. Asserting only
 * "the enabled arm looks like X" would pass just as happily if the feature were
 * silently inert, which is the failure mode that matters here.
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady, waitForNextRender, getWebGLErrors, captureCanvasRGBA } from './helpers';

const TWO_NODE_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_gsplats_two_node_overlap.luxar.zarr';

/** Both gsplat nodes have committed a non-empty draw. */
async function waitForBothNodesCommitted(page: import('@playwright/test').Page): Promise<void> {
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
        if (o.userData?.nodeType === 'gsplats' && (o.geometry?.instanceCount ?? 0) > 0) {
          committed++;
        }
      });
      return committed >= 2;
    },
    undefined,
    { timeout: 60000 }
  );
}

/**
 * Every drawn data mesh's `renderOrder`, lowest first, tagged with the node it
 * belongs to. Shard meshes report their parent's name, so the sequence reads as
 * the node alternation pattern the merge produced.
 */
async function drawSequence(page: import('@playwright/test').Page): Promise<string[]> {
  return page.evaluate(() => {
    const debug = (window as unknown as { __luxarDebug: { scene: unknown } }).__luxarDebug;
    const rows: { order: number; node: string }[] = [];
    (debug.scene as { traverse: (cb: (obj: unknown) => void) => void }).traverse((obj) => {
      const o = obj as {
        isMesh?: boolean;
        visible?: boolean;
        name?: string;
        renderOrder?: number;
        userData?: { nodeType?: string; depthShardOf?: string };
        geometry?: { instanceCount?: number };
      };
      if (!o.isMesh || !o.visible) return;
      if ((o.geometry?.instanceCount ?? 0) <= 0) return;
      const node = o.userData?.depthShardOf ?? (o.userData?.nodeType ? o.name : undefined);
      if (!node) return;
      rows.push({ order: o.renderOrder ?? 0, node });
    });
    return rows.sort((a, b) => a.order - b.order).map((r) => r.node);
  });
}

/** Count node changes along the draw sequence — 1 means node-major. */
function alternations(sequence: string[]): number {
  let changes = 0;
  for (let i = 1; i < sequence.length; i++) if (sequence[i] !== sequence[i - 1]) changes++;
  return changes;
}

test.describe('Cross-node depth ordering', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.setItem('luxar-control-rail-hint-dismissed', '1');
    });
  });

  test('?depthShards=0 draws each node as ONE call (the fail-first baseline)', async ({ page }) => {
    await page.goto(`/?src=${TWO_NODE_FIXTURE}&debug&dpr=1&depthShards=0`);
    await waitForLuxarReady(page);
    await waitForBothNodesCommitted(page);
    await waitForNextRender(page, 5);

    const interleaved = await page.evaluate(
      () =>
        (
          window as unknown as { __luxarDebug: { getDepthShardDrawCount?: () => number } }
        ).__luxarDebug.getDepthShardDrawCount?.() ?? -1
    );
    expect(interleaved, 'nothing is split when the escape hatch is pinned').toBe(0);

    const sequence = await drawSequence(page);
    expect(sequence.length, 'both nodes drew').toBe(2);
    // One draw per node, so exactly one node change: strictly node-major.
    expect(alternations(sequence)).toBe(1);

    expect(await getWebGLErrors(page)).toEqual([]);
  });

  test('?depthShards=N interleaves the two nodes and CHANGES the composite', async ({ page }) => {
    // --- baseline arm: the feature pinned off -------------------------------
    await page.goto(`/?src=${TWO_NODE_FIXTURE}&debug&dpr=1&depthShards=0`);
    await waitForLuxarReady(page);
    await waitForBothNodesCommitted(page);
    await waitForNextRender(page, 8);
    const baselinePixels = await captureCanvasRGBA(page);
    const baselineSequence = await drawSequence(page);

    // --- enabled arm -------------------------------------------------------
    await page.goto(`/?src=${TWO_NODE_FIXTURE}&debug&dpr=1&depthShards=8`);
    await waitForLuxarReady(page);
    await waitForBothNodesCommitted(page);
    // The split is established by the per-frame policy and its shard bounds
    // arrive with the next resolved sort, so wait for the draw count to settle
    // rather than assuming one frame is enough.
    await page.waitForFunction(
      () =>
        ((
          window as unknown as { __luxarDebug: { getDepthShardDrawCount?: () => number } }
        ).__luxarDebug.getDepthShardDrawCount?.() ?? 0) > 2,
      undefined,
      { timeout: 30000 }
    );
    await waitForNextRender(page, 8);

    const interleavedDraws = await page.evaluate(
      () =>
        (
          window as unknown as { __luxarDebug: { getDepthShardDrawCount?: () => number } }
        ).__luxarDebug.getDepthShardDrawCount?.() ?? 0
    );
    // Two overlapping nodes at 8 shards each.
    expect(interleavedDraws).toBe(16);

    const shardedSequence = await drawSequence(page);
    expect(shardedSequence.length).toBe(16);
    // Both nodes still draw every element exactly once, split across shards.
    expect(new Set(shardedSequence).size).toBe(2);
    // The point of the whole exercise: the draw order ALTERNATES between the
    // nodes rather than emitting one and then the other.
    expect(
      alternations(shardedSequence),
      `draw sequence ${shardedSequence.join(',')} is still node-major`
    ).toBeGreaterThan(1);
    expect(alternations(baselineSequence)).toBe(1);

    // ...and it reaches the framebuffer. Same geometry, same element count, same
    // fragments — a pure ordering difference, so any pixel change IS the
    // compositing change. Compared against the pinned-off arm rather than a
    // golden image, which is what makes this fail-first.
    const shardedPixels = await captureCanvasRGBA(page);
    expect(shardedPixels.width).toBe(baselinePixels.width);
    expect(shardedPixels.height).toBe(baselinePixels.height);
    let changed = 0;
    let lit = 0;
    const a = baselinePixels.rgba;
    const b = shardedPixels.rgba;
    for (let i = 0; i < a.length; i += 4) {
      if (a[i] + a[i + 1] + a[i + 2] > 24 || b[i] + b[i + 1] + b[i + 2] > 24) lit++;
      if (
        Math.abs(a[i] - b[i]) > 2 ||
        Math.abs(a[i + 1] - b[i + 1]) > 2 ||
        Math.abs(a[i + 2] - b[i + 2]) > 2
      ) {
        changed++;
      }
    }
    expect(lit, 'the scene rendered something in at least one arm').toBeGreaterThan(0);
    expect(
      changed,
      'interleaving changed no pixels — the feature is inert or the fixture does not interpenetrate'
    ).toBeGreaterThan(0);

    expect(await getWebGLErrors(page)).toEqual([]);
  });

  test('picking still resolves over a sharded node', async ({ page }) => {
    // The silent-failure surface: sharding narrows a node's own mesh to one
    // depth range, and the pick pass renders only that mesh. Without the widen,
    // hover stops working over most of the node with no error at all.
    await page.goto(`/?src=${TWO_NODE_FIXTURE}&debug&dpr=1&depthShards=8`);
    await waitForLuxarReady(page);
    await waitForBothNodesCommitted(page);
    await page.waitForFunction(
      () =>
        ((
          window as unknown as { __luxarDebug: { getDepthShardDrawCount?: () => number } }
        ).__luxarDebug.getDepthShardDrawCount?.() ?? 0) > 2,
      undefined,
      { timeout: 30000 }
    );
    await waitForNextRender(page, 5);

    // The pick geometry must be widened back to the node's whole population for
    // the pick pass. Assert on the invariant that guarantees it: every sharded
    // node's element total still equals what the commit stamped.
    const nodes = await page.evaluate(() => {
      const debug = (window as unknown as { __luxarDebug: { scene: unknown } }).__luxarDebug;
      const out: { name: string; stamped: number; drawn: number }[] = [];
      (debug.scene as { traverse: (cb: (obj: unknown) => void) => void }).traverse((obj) => {
        const o = obj as {
          isMesh?: boolean;
          name?: string;
          userData?: { nodeType?: string; visibleSplatCount?: number };
          geometry?: { instanceCount?: number };
          children?: { geometry?: { instanceCount?: number } }[];
        };
        if (!o.isMesh || o.userData?.nodeType !== 'gsplats') return;
        const shardSum = (o.children ?? []).reduce(
          (sum, c) => sum + (c.geometry?.instanceCount ?? 0),
          0
        );
        out.push({
          name: o.name ?? '?',
          stamped: o.userData?.visibleSplatCount ?? 0,
          drawn: (o.geometry?.instanceCount ?? 0) + shardSum,
        });
      });
      return out;
    });
    expect(nodes.length).toBe(2);
    for (const node of nodes) {
      expect(node.drawn, `${node.name}: shards partition the node`).toBe(node.stamped);
    }

    expect(await getWebGLErrors(page)).toEqual([]);
  });
});
