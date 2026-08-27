/**
 * Unit tests for Overview-tab templates.
 */

import { describe, it, expect } from 'vitest';
import type { CacheMetrics, GlobalStats } from '../../../../../types/data-monitor-types';
import {
  renderOverviewContent,
  renderSecondaryMetrics,
} from '../../../../../ui/data-loading-monitor/templates/overview';

describe('renderOverviewContent — dropped elements', () => {
  it('renders a red card with the partition remedy when elements were dropped', () => {
    const html = renderOverviewContent(
      {
        datasetSize: 100,
        visiblePoints: 80,
        datasetSegments: 0,
        visibleSegments: 0,
        datasetSplats: 0,
        visibleSplats: 0,
        droppedElements: 20,
        avgQueryTime: 0,
        queriesPerSecond: 0,
      } as GlobalStats,
      { totalCacheMemory: 0, memoryLimit: 1 } as CacheMetrics
    );

    expect(html).toContain('DROPPED ELEMENTS');
    expect(html).toContain('luxar-color--error');
    expect(html).toContain('partition oversized nodes');
    expect(html).toContain('Split the dataset into multiple nodes or partition it');
  });
});

describe('renderSecondaryMetrics — requests served', () => {
  const memory = { used: 0, limit: 1000 };
  const querySpeed = { avgTime: 0, perSec: 0 };

  it('surfaces totalRequestsServed in the network detail when present', () => {
    const html = renderSecondaryMetrics(memory, querySpeed, {
      bytesTransferred: 100,
      requestCount: 7,
      bandwidth: 50,
      totalBytesServed: 200,
      totalRequestsServed: 42,
    });
    expect(html).toContain('42 reqs');
  });

  it('falls back to requestCount when totalRequestsServed is undefined', () => {
    const html = renderSecondaryMetrics(memory, querySpeed, {
      bytesTransferred: 100,
      requestCount: 7,
      bandwidth: 50,
    });
    expect(html).toContain('7 reqs');
  });
});
