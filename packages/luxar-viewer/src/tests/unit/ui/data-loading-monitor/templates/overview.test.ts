/**
 * Unit tests for Overview-tab templates.
 */

import { describe, it, expect } from 'vitest';
import { renderSecondaryMetrics } from '../../../../../ui/data-loading-monitor/templates/overview';

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
