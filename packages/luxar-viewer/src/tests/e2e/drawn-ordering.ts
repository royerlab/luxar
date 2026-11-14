/**
 * The depth ordering each instanced node is DRAWN with, for E2E assertions.
 *
 * Read through `__luxarDebug.readDrawnOrdering`, which reads the selected
 * ordering buffer back from the GPU on native WebGPU. A test must not read the
 * attribute's host `array` instead: when the GPU depth sort (#3162) writes the
 * order, the host array is left behind, so it would assert on an order the
 * screen does not show.
 *
 * @module tests/e2e/drawn-ordering
 */

import type { Page } from '@playwright/test';

/** An instanced node type that depth-sorts. */
export type SortedNodeType = 'gsplats' | 'points' | 'lines';

/** One node's drawn ordering under the current camera. */
export interface DrawnOrderingState {
  uuid: string;
  /** Elements drawn (`visibleSplatCount` / `visiblePointCount` / `visibleSegmentCount`). */
  count: number;
  ordering: number[];
  /** View-space z of each drawn element's sort center, in draw order. */
  viewZs: number[];
  /** The ordering is `[0, 1, …, count - 1]`. */
  identity: boolean;
  /** View z never decreases along the ordering (equal-depth slack 1e-4). */
  backToFront: boolean;
}

/**
 * Every node of `nodeType` with at least `minCount` drawn elements, with its
 * drawn ordering and the view depths along it. Sort centers follow the element
 * layouts: texel 0 for splats (16 floats each) and points (12), the midpoint of
 * texels 0 and 1 for line segments (24).
 */
export async function drawnOrderingStates(
  page: Page,
  nodeType: SortedNodeType,
  minCount = 2
): Promise<DrawnOrderingState[]> {
  return page.evaluate(
    async ({ nodeType, minCount }) => {
      const debug = (window as unknown as { __luxarDebug?: any }).__luxarDebug;
      if (!debug?.scene || !debug.readDrawnOrdering) return [];
      const nodes: any[] = [];
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType !== nodeType) return;
        const u = obj.userData;
        const count = u.visibleSplatCount ?? u.visiblePointCount ?? u.visibleSegmentCount ?? 0;
        if (count >= minCount && obj.geometry?.userData?.elementTexture) nodes.push(obj);
      });
      const floats = { gsplats: 16, points: 12, lines: 24 }[nodeType];
      const midpoint = nodeType === 'lines';
      const mwi = debug.camera.matrixWorldInverse.elements;
      const states = [];
      for (const obj of nodes) {
        const u = obj.userData;
        const count = u.visibleSplatCount ?? u.visiblePointCount ?? u.visibleSegmentCount;
        const ordering: number[] | null = await debug.readDrawnOrdering(obj.uuid, count);
        if (!ordering) continue;
        const t = obj.geometry.userData.elementTexture.image.data;
        const mw = obj.matrixWorld.elements;
        const viewZs: number[] = [];
        let identity = true;
        let backToFront = true;
        let previous = -Infinity;
        ordering.forEach((i, j) => {
          if (i !== j) identity = false;
          const b = i * floats;
          const x = midpoint ? (t[b] + t[b + 4]) / 2 : t[b];
          const y = midpoint ? (t[b + 1] + t[b + 5]) / 2 : t[b + 1];
          const z = midpoint ? (t[b + 2] + t[b + 6]) / 2 : t[b + 2];
          const wx = mw[0] * x + mw[4] * y + mw[8] * z + mw[12];
          const wy = mw[1] * x + mw[5] * y + mw[9] * z + mw[13];
          const wz = mw[2] * x + mw[6] * y + mw[10] * z + mw[14];
          const viewZ = mwi[2] * wx + mwi[6] * wy + mwi[10] * wz + mwi[14];
          viewZs.push(viewZ);
          if (viewZ < previous - 1e-4) backToFront = false;
          previous = Math.max(previous, viewZ);
        });
        states.push({ uuid: obj.uuid, count, ordering, viewZs, identity, backToFront });
      }
      return states;
    },
    { nodeType, minCount }
  );
}

/** Some node of `nodeType` is drawn in a non-identity (sorted) order. */
export async function anyNodeSorted(page: Page, nodeType: SortedNodeType): Promise<boolean> {
  return (await drawnOrderingStates(page, nodeType)).some((s) => !s.identity);
}

/** Every node of `nodeType` (at least one) is drawn back to front. */
export async function allNodesBackToFront(page: Page, nodeType: SortedNodeType): Promise<boolean> {
  const states = await drawnOrderingStates(page, nodeType);
  return states.length > 0 && states.every((s) => s.backToFront);
}
