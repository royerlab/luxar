/**
 * Unit tests for ui/data-loading-monitor/timing-panel.ts.
 *
 * The renderer is a pure function from TimingEntry → HTML string.
 * The handlers + in-place updater touch the DOM (jsdom) but no
 * external dependencies, so we exercise them directly against
 * fixtures rather than mocking.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { TimingEntry } from '../../../profiling/update-profiler';
import {
  toggleExpanded,
  renderHierarchicalTimingPanel,
  attachTimingPanelHandlers,
  updateTimingPanelValues,
} from '../../../ui/data-loading-monitor/timing-panel';

function makeEntry(overrides: Partial<TimingEntry> = {}): TimingEntry {
  return {
    name: 'Total Update',
    lastMs: 5,
    avgMs: 4,
    count: 1,
    children: [],
    overBudget: false,
    ...overrides,
  };
}

function renderInto(html: string): HTMLElement {
  const container = document.createElement('div');
  container.innerHTML = html;
  document.body.appendChild(container);
  return container;
}

beforeEach(() => {
  document.body.innerHTML = '';
});

describe('renderHierarchicalTimingPanel', () => {
  it('renders an empty placeholder when count=0', () => {
    const html = renderHierarchicalTimingPanel(makeEntry({ count: 0 }));
    expect(html).toContain('luxar-timing-panel--empty');
    expect(html).toContain('No timing data yet');
  });

  it('renders header / body / footer when data is present', () => {
    const html = renderHierarchicalTimingPanel(makeEntry({ count: 5 }));
    expect(html).toContain('luxar-timing-panel__header');
    expect(html).toContain('luxar-timing-panel__body');
    expect(html).toContain('luxar-timing-panel__footer');
    expect(html).toContain('5 updates');
    expect(html).toContain('Operation');
    expect(html).toContain('Last');
    expect(html).toContain('Avg');
  });

  it('marks rows over the 16ms budget with the over-budget class', () => {
    const html = renderHierarchicalTimingPanel(
      makeEntry({ count: 1, lastMs: 25, overBudget: true })
    );
    expect(html).toContain('luxar-timing-panel__row--over');
  });

  it('renders em-dashes and skip class for skipped entries', () => {
    const html = renderHierarchicalTimingPanel(
      makeEntry({
        count: 1,
        children: [
          {
            name: 'Points',
            lastMs: 0,
            avgMs: 0,
            count: 0,
            children: [],
            metadata: { skipped: true, skipReason: 'no data' },
          },
        ],
      })
    );
    expect(html).toContain('luxar-timing-panel__row--skipped');
    expect(html).toContain('luxar-timing-panel__tag--skip');
    expect(html).toContain('—');
  });

  it('renders metadata tags for chunks / cache / counts / info', () => {
    // Children with a name that doesn't match Points/Lines/GSplats fall
    // through the aggregator's "Unknown" branch and are emitted as-is,
    // preserving their full metadata. (The aggregated Points/Lines/
    // GSplats nodes only carry points/segments/splats + skipped + info.)
    const root = makeEntry({
      count: 1,
      children: [
        {
          name: 'Custom Pass',
          lastMs: 5,
          avgMs: 4,
          count: 1,
          children: [],
          metadata: {
            chunks: 12,
            cacheHits: 90,
            cacheMisses: 10,
            points: 5_500_000,
            info: 'extra',
          },
        },
      ],
    });
    const html = renderHierarchicalTimingPanel(root);
    expect(html).toContain('12 chunks');
    expect(html).toContain('90% cache');
    expect(html).toContain('5.5M pts');
    expect(html).toContain('extra');
  });

  it('formats counts with K and raw thresholds', () => {
    const root = makeEntry({
      count: 1,
      children: [
        {
          name: 'Lines (/foo)',
          lastMs: 5,
          avgMs: 4,
          count: 1,
          children: [],
          metadata: { segments: 1500 },
        },
        {
          name: 'GSplats (/bar)',
          lastMs: 5,
          avgMs: 4,
          count: 1,
          children: [],
          metadata: { splats: 42 },
        },
      ],
    });
    const html = renderHierarchicalTimingPanel(root);
    expect(html).toContain('1.5K segs');
    expect(html).toContain('42 splats');
  });

  it('flags low cache hit rate with the warn class', () => {
    // Same Unknown-pass-through trick as above so the cache metadata
    // survives aggregation.
    const root = makeEntry({
      count: 1,
      children: [
        {
          name: 'Custom Pass',
          lastMs: 5,
          avgMs: 4,
          count: 1,
          children: [],
          metadata: { cacheHits: 10, cacheMisses: 90 },
        },
      ],
    });
    const html = renderHierarchicalTimingPanel(root);
    expect(html).toContain('luxar-timing-panel__tag--warn');
    expect(html).toContain('10% cache');
  });

  it('aggregates multiple children of the same node-type into one row', () => {
    const root = makeEntry({
      count: 1,
      children: [
        {
          name: 'Points (/a)',
          lastMs: 5,
          avgMs: 4,
          count: 1,
          children: [],
          metadata: { points: 100 },
        },
        {
          name: 'Points (/b)',
          lastMs: 6,
          avgMs: 5,
          count: 1,
          children: [],
          metadata: { points: 200 },
        },
        {
          name: 'Lines (/c)',
          lastMs: 3,
          avgMs: 2,
          count: 1,
          children: [],
          metadata: { segments: 50 },
        },
      ],
    });
    const html = renderHierarchicalTimingPanel(root);

    // Two Points entries → one aggregated 'Points' row with combined metadata.
    expect(html).toContain('300 pts'); // 100 + 200
    expect(html).toContain('2 nodes'); // info tag for aggregated count
    expect(html).toContain('50 segs');
  });

  it('renders nested children when parent is expanded by default (depth < 2)', () => {
    const root = makeEntry({
      count: 1,
      children: [
        {
          name: 'Points (/foo)',
          lastMs: 5,
          avgMs: 4,
          count: 1,
          children: [{ name: 'Spatial Query', lastMs: 1, avgMs: 1, count: 1, children: [] }],
        },
      ],
    });
    const html = renderHierarchicalTimingPanel(root);
    expect(html).toContain('Spatial Query');
  });

  it('renders the named tooltip on the operation label', () => {
    const root = makeEntry({
      count: 1,
      children: [
        {
          name: 'Points (/foo)',
          lastMs: 5,
          avgMs: 4,
          count: 1,
          children: [{ name: 'Spatial Query', lastMs: 1, avgMs: 1, count: 1, children: [] }],
        },
      ],
    });
    const html = renderHierarchicalTimingPanel(root);
    expect(html).toContain('Ask the spatial index which data chunks intersect');
  });
});

describe('toggleExpanded + collapsed rendering', () => {
  // The expandedState map is module-scoped; keep paths unique per test
  // so previous toggles don't leak.
  it('toggle from default-expanded → collapsed hides children', () => {
    // Default: depth < 2 expanded. Toggle once to collapse.
    toggleExpanded('Total Update/Points/Spatial Query');

    const root = makeEntry({
      count: 1,
      children: [
        {
          name: 'Points (/x)',
          lastMs: 5,
          avgMs: 4,
          count: 1,
          children: [
            {
              name: 'Spatial Query',
              lastMs: 1,
              avgMs: 1,
              count: 1,
              children: [{ name: 'Inner', lastMs: 0.5, avgMs: 0.5, count: 1, children: [] }],
            },
          ],
        },
      ],
    });
    const html = renderHierarchicalTimingPanel(root);
    // 'Spatial Query' itself still renders, but its child 'Inner' must not.
    expect(html).toContain('Spatial Query');
    expect(html).not.toContain('Inner');
    // Reset for other tests — toggle once more to flip to expanded.
    toggleExpanded('Total Update/Points/Spatial Query');
  });
});

describe('attachTimingPanelHandlers', () => {
  it('wires click handlers to expand spans, calling onUpdate on click', () => {
    const html = renderHierarchicalTimingPanel(
      makeEntry({
        count: 1,
        children: [
          {
            name: 'Points (/foo)',
            lastMs: 5,
            avgMs: 4,
            count: 1,
            children: [{ name: 'Spatial Query', lastMs: 1, avgMs: 1, count: 1, children: [] }],
          },
        ],
      })
    );
    const container = renderInto(html);
    const onUpdate = vi.fn();

    attachTimingPanelHandlers(container, onUpdate);

    const expand = container.querySelector('.luxar-timing-panel__expand') as HTMLElement;
    expand.click();

    expect(onUpdate).toHaveBeenCalledTimes(1);
  });

  it('skips re-attaching handlers (idempotent via dataset marker)', () => {
    const html = renderHierarchicalTimingPanel(
      makeEntry({
        count: 1,
        children: [
          {
            name: 'Points (/foo)',
            lastMs: 5,
            avgMs: 4,
            count: 1,
            children: [{ name: 'Spatial Query', lastMs: 1, avgMs: 1, count: 1, children: [] }],
          },
        ],
      })
    );
    const container = renderInto(html);
    const onUpdate = vi.fn();

    attachTimingPanelHandlers(container, onUpdate);
    attachTimingPanelHandlers(container, onUpdate);

    const expand = container.querySelector('.luxar-timing-panel__expand') as HTMLElement;
    expand.click();

    // If duplicate attach was permitted, onUpdate would fire twice.
    expect(onUpdate).toHaveBeenCalledTimes(1);
  });
});

describe('updateTimingPanelValues', () => {
  it('updates last/avg textContent in place', () => {
    const initial = makeEntry({ count: 1, lastMs: 5, avgMs: 4 });
    const container = renderInto(renderHierarchicalTimingPanel(initial));

    const updated = makeEntry({ count: 2, lastMs: 12, avgMs: 8 });
    const ok = updateTimingPanelValues(container, updated);
    expect(ok).toBe(true);

    const last = container.querySelector('.luxar-timing-panel__last');
    const avg = container.querySelector('.luxar-timing-panel__avg');
    expect(last?.textContent).toBe('12ms');
    expect(avg?.textContent).toBe('8.0ms');
  });

  it('updates the footer update count', () => {
    const initial = makeEntry({ count: 5 });
    const container = renderInto(renderHierarchicalTimingPanel(initial));

    updateTimingPanelValues(container, makeEntry({ count: 99 }));

    expect(container.querySelector('.luxar-timing-panel__update-count')?.textContent).toBe(
      '99 updates'
    );
  });

  it('returns false when the body element is missing', () => {
    const container = document.createElement('div');
    expect(updateTimingPanelValues(container, makeEntry({ count: 1 }))).toBe(false);
  });

  it('returns false when child set has changed (signals full re-render)', () => {
    const initial = makeEntry({
      count: 1,
      children: [
        {
          name: 'Points (/foo)',
          lastMs: 5,
          avgMs: 4,
          count: 1,
          children: [{ name: 'Spatial Query', lastMs: 1, avgMs: 1, count: 1, children: [] }],
        },
      ],
    });
    const container = renderInto(renderHierarchicalTimingPanel(initial));

    const reshaped = makeEntry({
      count: 2,
      children: [
        {
          name: 'Points (/foo)',
          lastMs: 6,
          avgMs: 5,
          count: 2,
          children: [
            { name: 'Spatial Query', lastMs: 1, avgMs: 1, count: 1, children: [] },
            { name: 'Load Arrays', lastMs: 2, avgMs: 2, count: 1, children: [] },
          ],
        },
      ],
    });
    expect(updateTimingPanelValues(container, reshaped)).toBe(false);
  });
});

describe('stale rows', () => {
  it('greys stale rows with the stale class and a tooltip note', () => {
    const root = makeEntry({
      count: 3,
      children: [
        makeEntry({
          name: 'GSplats (/g)',
          lastMs: 29,
          children: [
            makeEntry({ name: 'Load Arrays', lastMs: 94, stale: true }),
            makeEntry({ name: 'Project to 3D', lastMs: 2.7 }),
          ],
        }),
      ],
    });
    const html = renderHierarchicalTimingPanel(root);
    expect(html).toContain('luxar-timing-panel__row--stale');
    expect(html).toContain('did not run in the latest update');
  });

  it('excludes stale children lastMs from the type aggregation sum', () => {
    // Two gsplats nodes: one fresh 29ms, one stale 94ms. The aggregated
    // GSplats row must show 29ms, not 123ms.
    const root = makeEntry({
      count: 3,
      children: [
        makeEntry({ name: 'GSplats (/a)', lastMs: 29 }),
        makeEntry({ name: 'GSplats (/b)', lastMs: 94, stale: true }),
      ],
    });
    const html = renderHierarchicalTimingPanel(root);
    expect(html).toContain('29ms');
    expect(html).not.toContain('123ms');
  });

  it('shows the last-known (stale) sum on an all-stale aggregated row', () => {
    const root = makeEntry({
      count: 3,
      children: [makeEntry({ name: 'GSplats (/a)', lastMs: 94, stale: true })],
    });
    const html = renderHierarchicalTimingPanel(root);
    // Value preserved (greyed via the stale class), not zeroed.
    expect(html).toContain('94ms');
    expect(html).toContain('luxar-timing-panel__row--stale');
  });
});

describe('LOD Refinement tree', () => {
  function makeRefinementRoot(): TimingEntry {
    return makeEntry({
      name: 'LOD Refinement',
      lastMs: 82,
      avgMs: 91,
      count: 5,
      children: [
        makeEntry({
          name: 'GSplats (/g)',
          lastMs: 82,
          avgMs: 91,
          children: [makeEntry({ name: 'Load Arrays', lastMs: 78, avgMs: 85 })],
        }),
      ],
    });
  }

  it('renders the refinement tree as a second section with a pass count', () => {
    const html = renderHierarchicalTimingPanel(makeEntry({ count: 8 }), makeRefinementRoot());
    expect(html).toContain('LOD Refinement');
    expect(html).toContain('8 updates · 5 refinement passes');
  });

  it('omits the refinement section when no pass has recorded', () => {
    const html = renderHierarchicalTimingPanel(
      makeEntry({ count: 8 }),
      makeEntry({ name: 'LOD Refinement', count: 0 })
    );
    expect(html).not.toContain('LOD Refinement');
    expect(html).toContain('8 updates');
    expect(html).not.toMatch(/\d+ refinement pass/);
  });

  it('renders refinement data even when Total Update has no updates yet', () => {
    const html = renderHierarchicalTimingPanel(makeEntry({ count: 0 }), makeRefinementRoot());
    expect(html).not.toContain('luxar-timing-panel--empty');
    expect(html).toContain('LOD Refinement');
  });

  it('updateTimingPanelValues requests a full re-render when the refinement tree first appears', () => {
    const container = renderInto(renderHierarchicalTimingPanel(makeEntry({ count: 8 })));
    // No refinement section rendered yet → structural change → false.
    expect(updateTimingPanelValues(container, makeEntry({ count: 9 }), makeRefinementRoot())).toBe(
      false
    );
  });

  it('updateTimingPanelValues updates both trees in place once rendered', () => {
    const refinement = makeRefinementRoot();
    const container = renderInto(
      renderHierarchicalTimingPanel(makeEntry({ count: 8 }), refinement)
    );
    const ok = updateTimingPanelValues(container, makeEntry({ count: 9 }), {
      ...refinement,
      count: 6,
    });
    expect(ok).toBe(true);
    expect(container.querySelector('.luxar-timing-panel__update-count')?.textContent).toBe(
      '9 updates · 6 refinement passes'
    );
  });
});
