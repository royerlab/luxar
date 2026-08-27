/**
 * The per-geometry headline counts, in one table.
 *
 * Three surfaces show "how much of this type is on screen": the Overview
 * tab's hero cards (`templates/overview.ts`), their incremental patcher
 * (`tabs/overview.ts`), and the collapsed monitor's compact badge
 * (`data-loading-monitor.ts`). Each used to hard-code its own list of three
 * types with its own presence test and its own label strings, which is how
 * mesh came to be missing from all three at once while the counts it needed
 * were already being aggregated: `visibleByType.mesh` was pushed every update
 * cycle by `monitor/visible-counts.ts` and then dropped on the floor.
 *
 * `Record<GeometryTypeName, …>` makes adding a geometry type a compile error
 * here rather than a silent omission in three places, and every consumer reads
 * labels, DOM field ids, unit nouns and presence from this one table — so a
 * card, its patch target and its badge entry cannot disagree.
 *
 * @module ui/data-loading-monitor/headline-counts
 */

import type { GlobalStats } from '../../types/data-monitor-types';
import { GEOMETRY_TYPES, type GeometryTypeName } from '../../types/format-contract';

/** One geometry type's headline count, resolved against a `GlobalStats` tick. */
export interface HeadlineCount {
  /** Geometry type this entry describes. */
  type: GeometryTypeName;
  /** Hero-card title, e.g. `VISIBLE POINTS`. */
  label: string;
  /**
   * DOM `data-field` id of the card's value cell (`-sub` for its subtitle).
   * STABLE: the incremental patcher queries these, and `tabs/overview.ts`
   * treats their absence as "structure not painted yet, rebuild".
   */
  field: string;
  /** Short unit noun for the compact badge, e.g. `pts`. */
  unit: string;
  /** Element noun for prose (tooltips), e.g. `line segments`. */
  noun: string;
  /** Elements of this type currently on screen. */
  visible: number;
  /** Elements of this type across the whole dataset. */
  total: number;
}

/**
 * Static half of the table: everything that does not depend on a tick.
 *
 * The count fields are read by name rather than keyed by type because
 * `GlobalStats` names each pair after that type's own element noun
 * (`visibleSegments`, not `visibleLines`) — see the field block there.
 */
const DESCRIPTORS: Record<
  GeometryTypeName,
  Omit<HeadlineCount, 'type' | 'visible' | 'total'> & {
    read: (stats: GlobalStats) => { visible: number; total: number };
  }
> = {
  points: {
    label: 'VISIBLE POINTS',
    field: 'visible-points',
    unit: 'pts',
    noun: 'points',
    read: (s) => ({ visible: s.visiblePoints, total: s.datasetSize }),
  },
  lines: {
    label: 'VISIBLE LINES',
    field: 'visible-lines',
    unit: 'lines',
    noun: 'line segments',
    read: (s) => ({ visible: s.visibleSegments, total: s.datasetSegments }),
  },
  gsplats: {
    label: 'VISIBLE SPLATS',
    field: 'visible-splats',
    unit: 'splats',
    noun: 'Gaussian splats',
    read: (s) => ({ visible: s.visibleSplats, total: s.datasetSplats }),
  },
  mesh: {
    // Triangles, matching the drawn-primitive convention the whole monitor
    // uses for mesh (the scene-graph badge, the visible-counts walk, the
    // timing tag) — and the same choice `lines` makes in counting segments
    // rather than vertices.
    label: 'VISIBLE TRIANGLES',
    field: 'visible-triangles',
    unit: 'tris',
    noun: 'mesh triangles',
    read: (s) => ({ visible: s.visibleTriangles, total: s.datasetTriangles }),
  },
};

/** Every geometry type's headline count for this tick, in `GEOMETRY_TYPES` order. */
export function headlineCounts(stats: GlobalStats): HeadlineCount[] {
  return GEOMETRY_TYPES.map((type) => {
    const d = DESCRIPTORS[type];
    const { visible, total } = d.read(stats);
    return { type, label: d.label, field: d.field, unit: d.unit, noun: d.noun, visible, total };
  });
}

/**
 * The headline counts for the types actually PRESENT in the scene — the one
 * presence test, so a card, its patch target and its badge entry always agree
 * on which types exist.
 *
 * A type counts as present when either number is non-zero: `total > 0` covers
 * a layer whose elements are all outside the current slice, and `visible > 0`
 * covers the window before dataset attrs have been read.
 */
export function presentHeadlineCounts(stats: GlobalStats): HeadlineCount[] {
  return headlineCounts(stats).filter((c) => c.total > 0 || c.visible > 0);
}

/**
 * Compact-badge tooltip for one type — same sentence, that type's element
 * noun, capitalized here (rather than at the call site) so the badge's wording
 * lives with the table it comes from.
 */
export function compactTooltip(noun: string): string {
  const sentenceNoun = noun.charAt(0).toUpperCase() + noun.slice(1);
  return (
    `${sentenceNoun} currently on screen (inside the active nD slice). ` +
    'Expand the monitor for totals and per-layer detail'
  );
}

/** Hero-card tooltip for one type — same sentence, that type's element noun. */
export function headlineTooltip(noun: string): string {
  return (
    `How many ${noun} are on screen right now versus how many the whole dataset holds. ` +
    'The two differ because only data inside the current nD slice is shown, and ' +
    'level-of-detail (LOD) streaming may not have loaded full resolution yet'
  );
}
