/**
 * Geometry-behaviour row `monitorVisibleCount`: a commit's visible-element
 * count reaches the data monitor's headline card for its type.
 *
 * The chain has four links, each written once per type and each of which has
 * dropped a type before (mesh was missing from all three headline surfaces
 * while its counts were being aggregated): the commit stamps the count on the
 * node, `updateVisibleCountsInMonitor` sums it into the per-type total,
 * `aggregateGlobalStats` projects that onto `GlobalStats`' named field, and
 * `headlineCounts` reads the field back for the card. The probe drives all
 * four with real code and asserts the card shows exactly what was committed.
 */
import { expect } from 'vitest';

import { SceneLoader } from '../../../data/scene-loader';
import type { SceneLoaderMonitorPort } from '../../../data/scene-loader-monitor-port';
import { updateVisibleCountsInMonitor } from '../../../data/scene-loader/monitor/visible-counts';
import { GEOMETRY_TYPES, type GeometryTypeName } from '../../../types/format-contract';
import { aggregateGlobalStats } from '../../../ui/data-loading-monitor/metrics/global-stats';
import { headlineCounts } from '../../../ui/data-loading-monitor/headline-counts';
import { GEOMETRY_COMMITS, makeCommitScene } from '../../helpers/geometry-commits';
import { defineBehaviourConformance } from '../../_conformance/define-behaviour-conformance';

const zeros = (): Record<GeometryTypeName, number> =>
  Object.fromEntries(GEOMETRY_TYPES.map((t) => [t, 0])) as Record<GeometryTypeName, number>;

defineBehaviourConformance('monitorVisibleCount', {
  async holds(type) {
    const loader = new SceneLoader({ enableMonitor: false });
    const root = makeCommitScene(loader);
    await GEOMETRY_COMMITS[type].commit(loader, 3);

    const visibleByType = zeros();
    const monitor = {
      updateVisibleCount: (t: GeometryTypeName, count: number) => {
        visibleByType[t] = count;
      },
      updateDroppedElementCount: () => {},
      updateVisibleCountsByPath: () => {},
    } as unknown as SceneLoaderMonitorPort;
    updateVisibleCountsInMonitor(root, monitor);

    const stats = aggregateGlobalStats({
      metrics: new Map(),
      loaders: new Map(),
      lodStates: new Map(),
      rates: { queriesPerSec: 0 },
      sceneGraph: { totalByType: zeros(), visibleByType, droppedElements: 0 },
      recommendations: [],
    });
    const cards = headlineCounts(stats);
    expect(cards.find((c) => c.type === type)?.visible).toBe(3);
    // Only this type was committed, so every other card reads zero.
    for (const other of cards) {
      if (other.type !== type) expect(other.visible, other.type).toBe(0);
    }
  },
});
