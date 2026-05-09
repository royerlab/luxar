/**
 * Direct unit tests for `ui/monitors/tabs/cache-tab.ts`. Exercises
 * the structure guard, L0/L1/L2 patch paths, progress-bar fill, and
 * the optional-field no-op case under jsdom.
 */

import { describe, it, expect } from 'vitest';
import { updateCacheTab } from '../../../../../ui/monitors/tabs/cache-tab';
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
    <div data-field="l2-io"></div>
    <div data-field="l2-io-sub"></div>
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
    evictionsPerMin: 0,
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
});
