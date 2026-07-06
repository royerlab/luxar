/**
 * The encoding-name -> worker-decode-kind mapping. A mis-route (e.g. geolog
 * -> 'log') decodes every HDR color with the wrong inverse compand and no
 * error, and no other unit test exercises this mapping (deep-check finding:
 * mutating it passed the whole suite).
 */

import { describe, it, expect } from 'vitest';
import { perChannelKindFor } from '../../../../data/loaders/spatial-query/range-loader/perchannel';

describe('perChannelKindFor', () => {
  it('routes every per-channel encoding name to its exact kind', () => {
    expect(perChannelKindFor('linear_perchannel_u8')).toBe('linear');
    expect(perChannelKindFor('linear_perchannel_u16')).toBe('linear');
    expect(perChannelKindFor('log_perchannel_u8')).toBe('log');
    expect(perChannelKindFor('log_perchannel_u16')).toBe('log');
    expect(perChannelKindFor('signed_log_perchannel_u8')).toBe('signed_log');
    expect(perChannelKindFor('signed_log_perchannel_u16')).toBe('signed_log');
    // The prefix trap: geolog must NOT be caught by the log branch.
    expect(perChannelKindFor('geolog_perchannel_u8')).toBe('geolog');
    expect(perChannelKindFor('geolog_perchannel_u16')).toBe('geolog');
  });
});
