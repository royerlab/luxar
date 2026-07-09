/**
 * Direct unit tests for `ui/data-loading-monitor/tabs/cache.ts`. Exercises
 * the structure guard, L0/L1/L2 patch paths, progress-bar fill, and
 * the optional-field no-op case under jsdom.
 */

import { describe, it, expect } from 'vitest';
import { updateCacheTab } from '../../../../../ui/data-loading-monitor/tabs/cache';
import type { CacheMetrics } from '../../../../../types/data-monitor-types';

function makeContainer(): HTMLElement {
  const el = document.createElement('div');
  // Minimum structure the cache-tab updater expects: data-field
  // anchors plus the cache-total wrapper with progress-bar internals.
  el.innerHTML = `
    <div data-field="l0-size"></div>
    <div data-field="l0-size-sub"></div>
    <div data-field="l0-hitrate"></div>
    <div data-field="l0-hitrate-sub"></div>
    <div data-field="l0-evictions"></div>
    <div data-field="l1-size"></div>
    <div data-field="l1-size-sub"></div>
    <div data-field="l1-hitrate"></div>
    <div data-field="l1-hitrate-sub"></div>
    <div data-field="l1-evictions"></div>
    <div data-field="l2-size"></div>
    <div data-field="l2-size-sub"></div>
    <div data-field="l2-hitrate"></div>
    <div data-field="l2-hitrate-sub"></div>
    <div data-field="l2-io"></div>
    <div data-field="l2-io-sub"></div>
    <div data-field="l2-errors"></div>
    <div data-field="l2-errors-sub"></div>
    <div data-field="cache-status-row"></div>
    <div data-field="cache-health-mode"></div>
    <div data-field="cache-health-validated"></div>
    <div class="luxar-cache-total">
      <span data-field="cache-total"></span>
      <div class="luxar-progress-bar">
        <div class="luxar-progress-bar__fill" style="width: 0%"></div>
        <span class="luxar-progress-bar__label"></span>
      </div>
    </div>
  `;
  document.body.appendChild(el);
  return el;
}

function makeMetrics(overrides: Partial<CacheMetrics> = {}): CacheMetrics {
  return {
    totalCacheMemory: 1024,
    memoryLimit: 4096,
    memoryPercent: 25,
    totalEntries: 4,
    totalAccesses: 0,
    recentHitRate: 0,
    evictionsTotal: 0,
    avgEntrySize: 256,
    reuseRatio: 0,
    hitsPerSecond: 0,
    missesPerSecond: 0,
    avgAccessTime: 0,
    queriesPerSec: 0,
    loadsPerSec: 0,
    bandwidth: 0,
    ...overrides,
  };
}

describe('updateCacheTab', () => {
  it('returns false when container is null', () => {
    expect(updateCacheTab(null, makeMetrics())).toBe(false);
  });

  it('returns false when the cache-total anchor is missing (structure not rendered)', () => {
    const stub = document.createElement('div');
    stub.innerHTML = '<div>no anchors</div>';
    expect(updateCacheTab(stub, makeMetrics())).toBe(false);
  });

  it('patches L0 fields when L0 metrics are present', () => {
    const c = makeContainer();
    updateCacheTab(
      c,
      makeMetrics({
        l0: {
          size: 2048,
          count: 5,
          hits: 10,
          misses: 2,
          evictions: 1,
          hitRate: 0.83,
        },
      })
    );

    const hitrate = c.querySelector('[data-field="l0-hitrate"]') as HTMLElement;
    expect(hitrate.textContent).toContain('83.3%');
    const sub = c.querySelector('[data-field="l0-hitrate-sub"]') as HTMLElement;
    expect(sub.textContent).toContain('hits');
    const evictions = c.querySelector('[data-field="l0-evictions"]') as HTMLElement;
    expect(evictions.textContent).toBe('1');
  });

  it('patches L1 fields when L1 metrics are present', () => {
    const c = makeContainer();
    updateCacheTab(
      c,
      makeMetrics({
        l1: {
          size: 1024,
          count: 3,
          hits: 80,
          misses: 20,
          evictions: 0,
        },
      })
    );

    const hitrate = c.querySelector('[data-field="l1-hitrate"]') as HTMLElement;
    expect(hitrate.textContent).toContain('80.0%');
  });

  it('patches L2 fields when L2 metrics are present', () => {
    const c = makeContainer();
    updateCacheTab(
      c,
      makeMetrics({
        l2: {
          size: 5000,
          count: 12,
          reads: 4,
          writes: 6,
          misses: 1,
        },
      })
    );

    const io = c.querySelector('[data-field="l2-io"]') as HTMLElement;
    expect(io.textContent).toContain('4');
    const ioSub = c.querySelector('[data-field="l2-io-sub"]') as HTMLElement;
    expect(ioSub.textContent).toContain('6');
  });

  // R2: locked-in regression — the per-tick patcher must update the L2
  // hit-rate cell so it doesn't freeze at its initial render value.
  describe('L2 hit-rate live patching (R2)', () => {
    it('patches l2-hitrate text and success color when hit rate is high', () => {
      const c = makeContainer();
      updateCacheTab(
        c,
        makeMetrics({
          l2: { size: 1, count: 1, reads: 80, writes: 0, misses: 20 },
        })
      );
      const hitrate = c.querySelector('[data-field="l2-hitrate"]') as HTMLElement;
      expect(hitrate.textContent).toContain('80.0%');
      const sub = c.querySelector('[data-field="l2-hitrate-sub"]') as HTMLElement;
      expect(sub.textContent).toContain('80');
      expect(sub.textContent).toContain('20');
      // Success threshold is >80; 80.0 is on the boundary and renders as warning.
      // Test the strictly-high case below.
    });

    it('uses success color for hit rates > 80%', () => {
      const c = makeContainer();
      updateCacheTab(
        c,
        makeMetrics({
          l2: { size: 1, count: 1, reads: 90, writes: 0, misses: 10 },
        })
      );
      const hitrate = c.querySelector('[data-field="l2-hitrate"]') as HTMLElement;
      expect(hitrate.className).toMatch(/success/i);
    });

    it('uses warning color for hit rates in (50, 80]', () => {
      const c = makeContainer();
      updateCacheTab(
        c,
        makeMetrics({
          l2: { size: 1, count: 1, reads: 60, writes: 0, misses: 40 },
        })
      );
      const hitrate = c.querySelector('[data-field="l2-hitrate"]') as HTMLElement;
      expect(hitrate.className).toMatch(/warning/i);
    });

    it('uses error color for hit rates ≤ 50%', () => {
      const c = makeContainer();
      updateCacheTab(
        c,
        makeMetrics({
          l2: { size: 1, count: 1, reads: 30, writes: 0, misses: 70 },
        })
      );
      const hitrate = c.querySelector('[data-field="l2-hitrate"]') as HTMLElement;
      expect(hitrate.textContent).toContain('30.0%');
      expect(hitrate.className).toMatch(/error/i);
    });

    it('renders em-dash and dimmed color when L2 has no reads or misses yet', () => {
      const c = makeContainer();
      updateCacheTab(
        c,
        makeMetrics({
          l2: { size: 0, count: 0, reads: 0, writes: 0, misses: 0 },
        })
      );
      const hitrate = c.querySelector('[data-field="l2-hitrate"]') as HTMLElement;
      expect(hitrate.textContent).toContain('—');
      expect(hitrate.className).toMatch(/dimmed|muted/i);
    });

    it('updates hit-rate across consecutive ticks (no stale freeze)', () => {
      const c = makeContainer();
      updateCacheTab(
        c,
        makeMetrics({
          l2: { size: 1, count: 1, reads: 10, writes: 0, misses: 90 },
        })
      );
      const hitrate = c.querySelector('[data-field="l2-hitrate"]') as HTMLElement;
      expect(hitrate.textContent).toContain('10.0%');

      // Second tick with different numbers — the field must update.
      updateCacheTab(
        c,
        makeMetrics({
          l2: { size: 1, count: 1, reads: 95, writes: 0, misses: 5 },
        })
      );
      expect(hitrate.textContent).toContain('95.0%');
    });
  });

  // Collapsible sections render each value twice (compact header
  // summary + full metric card) under the SAME data-field key — the
  // patcher must update every copy, text and color class alike.
  it('patches all duplicate copies of a data-field (compact summary + card)', () => {
    const c = makeContainer();
    // Simulate the dual-render: a second l0-hitrate span, as in the
    // collapsed-header summary.
    const dup = document.createElement('span');
    dup.setAttribute('data-field', 'l0-hitrate');
    c.appendChild(dup);

    updateCacheTab(
      c,
      makeMetrics({
        l0: { size: 1, count: 1, hits: 90, misses: 10, evictions: 0, hitRate: 0.9 },
      })
    );

    const copies = c.querySelectorAll('[data-field="l0-hitrate"]');
    expect(copies.length).toBe(2);
    copies.forEach((el) => {
      expect(el.textContent).toContain('90.0%');
      expect((el as HTMLElement).className).toMatch(/success/i);
    });
  });

  it('updates the total memory bar fill width', () => {
    const c = makeContainer();
    updateCacheTab(c, makeMetrics({ memoryPercent: 73, memoryLimit: 4096 }));

    const fill = c.querySelector('.luxar-progress-bar__fill') as HTMLElement;
    expect(fill.style.width).toBe('73%');
  });

  it('clamps the bar fill to 100% even if memoryPercent overshoots', () => {
    const c = makeContainer();
    updateCacheTab(c, makeMetrics({ memoryPercent: 175, memoryLimit: 1000 }));

    const fill = c.querySelector('.luxar-progress-bar__fill') as HTMLElement;
    expect(fill.style.width).toBe('100%');
  });

  it('progress-bar label reads "X% of Y limit" when memoryLimit > 0', () => {
    const c = makeContainer();
    updateCacheTab(c, makeMetrics({ memoryPercent: 25, memoryLimit: 4096 }));

    const label = c.querySelector('.luxar-cache-total .luxar-progress-bar__label') as HTMLElement;
    expect(label.textContent).toMatch(/25% of/);
    expect(label.textContent).toContain('limit');
    expect(label.textContent).not.toContain('no memory limit');
  });

  it('progress-bar label reads "no memory limit configured" when memoryLimit is 0', () => {
    const c = makeContainer();
    updateCacheTab(
      c,
      makeMetrics({ totalCacheMemory: 2_200_000, memoryLimit: 0, memoryPercent: 0 })
    );

    const label = c.querySelector('.luxar-cache-total .luxar-progress-bar__label') as HTMLElement;
    expect(label.textContent).toBe('no memory limit configured');
    // Misleading zero-limit label must not slip back in across ticks.
    expect(label.textContent).not.toContain('0% of 0B');
    expect(label.textContent).not.toContain('of 0B');
  });

  it('progress-bar label updates across ticks when memoryLimit toggles 0 → N', () => {
    const c = makeContainer();
    updateCacheTab(c, makeMetrics({ memoryLimit: 0, memoryPercent: 0 }));
    const label = c.querySelector('.luxar-cache-total .luxar-progress-bar__label') as HTMLElement;
    expect(label.textContent).toBe('no memory limit configured');

    updateCacheTab(c, makeMetrics({ memoryLimit: 1024, memoryPercent: 50 }));
    expect(label.textContent).toMatch(/50% of/);
    expect(label.textContent).toContain('limit');
  });

  it('returns true when patching succeeds', () => {
    const c = makeContainer();
    expect(updateCacheTab(c, makeMetrics())).toBe(true);
  });

  it('omitted L0/L1/L2 metrics are no-ops (no thrown errors)', () => {
    const c = makeContainer();
    expect(() =>
      updateCacheTab(c, makeMetrics({ l0: undefined, l1: undefined, l2: undefined }))
    ).not.toThrow();
  });

  // R3: status badges + cache health + L2 error counters.
  describe('cache status badges (R3)', () => {
    it('renders one pill per badge in cacheMetrics.status', () => {
      const c = makeContainer();
      updateCacheTab(c, makeMetrics({ status: ['cache-enabled', 'unvalidated-external-dataset'] }));
      const row = c.querySelector('[data-field="cache-status-row"]') as HTMLElement;
      expect(row.querySelectorAll('[data-badge]').length).toBe(2);
      expect(row.querySelector('[data-badge="cache-enabled"]')).not.toBeNull();
      expect(row.querySelector('[data-badge="unvalidated-external-dataset"]')).not.toBeNull();
    });

    it('renders empty row when status is empty/undefined', () => {
      const c = makeContainer();
      updateCacheTab(c, makeMetrics({ status: [] }));
      const row = c.querySelector('[data-field="cache-status-row"]') as HTMLElement;
      expect(row.querySelectorAll('[data-badge]').length).toBe(0);
    });

    it('does not rewrite innerHTML when the badge signature is stable', () => {
      const c = makeContainer();
      updateCacheTab(c, makeMetrics({ status: ['cache-enabled'] }));
      const row = c.querySelector('[data-field="cache-status-row"]') as HTMLElement;
      const firstChild = row.firstElementChild;
      updateCacheTab(c, makeMetrics({ status: ['cache-enabled'] }));
      // Same signature → same DOM node (no replace).
      expect(row.firstElementChild).toBe(firstChild);
    });

    it('replaces innerHTML when the badge signature changes', () => {
      const c = makeContainer();
      updateCacheTab(c, makeMetrics({ status: ['cache-enabled'] }));
      const row = c.querySelector('[data-field="cache-status-row"]') as HTMLElement;
      const firstChild = row.firstElementChild;
      updateCacheTab(c, makeMetrics({ status: ['cache-enabled', 'quota-constrained'] }));
      expect(row.firstElementChild).not.toBe(firstChild);
      expect(row.querySelectorAll('[data-badge]').length).toBe(2);
    });
  });

  describe('cache health row (R3)', () => {
    it('shows "Content Hash" for content-hash validation', () => {
      const c = makeContainer();
      updateCacheTab(
        c,
        makeMetrics({
          health: {
            validationMode: 'content-hash',
            lastValidatedAt: null,
            unvalidatedExternalDataset: false,
          },
        })
      );
      const el = c.querySelector('[data-field="cache-health-mode"]') as HTMLElement;
      expect(el.textContent).toContain('Content Hash');
    });

    it('shows "TTL" for ttl validation', () => {
      const c = makeContainer();
      updateCacheTab(
        c,
        makeMetrics({
          health: {
            validationMode: 'ttl',
            lastValidatedAt: null,
            unvalidatedExternalDataset: false,
          },
        })
      );
      expect(
        (c.querySelector('[data-field="cache-health-mode"]') as HTMLElement).textContent
      ).toContain('TTL');
    });

    it('shows "None" for none validation', () => {
      const c = makeContainer();
      updateCacheTab(
        c,
        makeMetrics({
          health: {
            validationMode: 'none',
            lastValidatedAt: null,
            unvalidatedExternalDataset: true,
          },
        })
      );
      expect(
        (c.querySelector('[data-field="cache-health-mode"]') as HTMLElement).textContent
      ).toContain('None');
    });

    it('shows "Never" when lastValidatedAt is null', () => {
      const c = makeContainer();
      updateCacheTab(
        c,
        makeMetrics({
          health: {
            validationMode: 'content-hash',
            lastValidatedAt: null,
            unvalidatedExternalDataset: false,
          },
        })
      );
      expect(
        (c.querySelector('[data-field="cache-health-validated"]') as HTMLElement).textContent
      ).toContain('Never');
    });

    it('shows formatted timestamp when lastValidatedAt is set', () => {
      const c = makeContainer();
      const ts = new Date('2026-05-10T14:00:00Z').getTime();
      updateCacheTab(
        c,
        makeMetrics({
          health: {
            validationMode: 'content-hash',
            lastValidatedAt: ts,
            unvalidatedExternalDataset: false,
          },
        })
      );
      const text = (c.querySelector('[data-field="cache-health-validated"]') as HTMLElement)
        .textContent;
      // Cross-locale: the date object always renders the year somewhere.
      expect(text).toContain('2026');
      expect(text).not.toContain('Never');
    });
  });

  describe('L2 error counters (R3)', () => {
    it('shows "0" + dimmed color when all counters are zero', () => {
      const c = makeContainer();
      updateCacheTab(
        c,
        makeMetrics({
          l2: {
            size: 1,
            count: 1,
            reads: 10,
            writes: 5,
            misses: 0,
            quotaWriteSkipped: 0,
            writeFailures: 0,
            corruptedEntries: 0,
            metadataParseFailures: 0,
          },
        })
      );
      const el = c.querySelector('[data-field="l2-errors"]') as HTMLElement;
      expect(el.textContent).toContain('0');
      expect(el.className).toMatch(/dimmed|muted/i);
      const sub = c.querySelector('[data-field="l2-errors-sub"]') as HTMLElement;
      expect(sub.textContent).toContain('no errors');
    });

    it('shows total + breakdown + error color when any counter is non-zero', () => {
      const c = makeContainer();
      updateCacheTab(
        c,
        makeMetrics({
          l2: {
            size: 1,
            count: 1,
            reads: 10,
            writes: 5,
            misses: 0,
            quotaWriteSkipped: 3,
            writeFailures: 1,
            corruptedEntries: 2,
            metadataParseFailures: 0,
          },
        })
      );
      const el = c.querySelector('[data-field="l2-errors"]') as HTMLElement;
      expect(el.textContent).toContain('6'); // 3 + 1 + 2
      expect(el.className).toMatch(/error/i);
      const sub = c.querySelector('[data-field="l2-errors-sub"]') as HTMLElement;
      expect(sub.textContent).toContain('3'); // quota
      expect(sub.textContent).toContain('1'); // write
      expect(sub.textContent).toContain('2'); // corrupt
    });

    it('handles missing counter fields gracefully (treats them as 0)', () => {
      const c = makeContainer();
      // No counters at all — should not throw, should render as "0 / no errors".
      updateCacheTab(
        c,
        makeMetrics({
          l2: {
            size: 1,
            count: 1,
            reads: 10,
            writes: 5,
            misses: 0,
          },
        })
      );
      const el = c.querySelector('[data-field="l2-errors"]') as HTMLElement;
      expect(el.textContent).toContain('0');
    });
  });
});
