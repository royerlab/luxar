/**
 * Tests for validateDepthShards (src/config/sections/depth-shards/validate.ts).
 * Each test exercises the per-section validator directly via invokeValidator().
 */

import { describe, it, expect } from 'vitest';
import { validateDepthShards } from '../../../../../config/sections/depth-shards/validate';
import { cloneConfig, invokeValidator } from '../../_fixtures';

describe('validateDepthShards', () => {
  it('passes on default config', () => {
    expect(invokeValidator(validateDepthShards).valid).toBe(true);
  });

  it('errors on a shard count below 2 or non-integer', () => {
    // 1 is not a split — it is an unsharded node paying the bookkeeping — so it
    // is an error rather than a silent no-op.
    for (const bad of [0, 1, -4, 2.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      const cfg = cloneConfig();
      cfg.depthShards.shardsPerNode = bad;
      const result = invokeValidator(validateDepthShards, cfg);
      expect(result.valid, `shardsPerNode=${bad}`).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('depthShards.shardsPerNode'));
    }
  });

  it('warns on a shard count past where the measured cost saturates', () => {
    const cfg = cloneConfig();
    cfg.depthShards.shardsPerNode = 512;
    cfg.depthShards.maxInterleavedDraws = 2048;
    const result = invokeValidator(validateDepthShards, cfg);
    expect(result.valid).toBe(true);
    expect(result.warnings).toContainEqual(expect.stringContaining('depthShards.shardsPerNode'));
  });

  it('errors on a draw budget below 2 or non-integer', () => {
    for (const bad of [0, 1, -1, 2.5, Number.NaN]) {
      const cfg = cloneConfig();
      cfg.depthShards.maxInterleavedDraws = bad;
      const result = invokeValidator(validateDepthShards, cfg);
      expect(result.valid, `maxInterleavedDraws=${bad}`).toBe(false);
      expect(result.errors).toContainEqual(
        expect.stringContaining('depthShards.maxInterleavedDraws')
      );
    }
  });

  it('warns when the budget is far past where the per-draw cost was measured', () => {
    const cfg = cloneConfig();
    cfg.depthShards.maxInterleavedDraws = 100_000;
    const result = invokeValidator(validateDepthShards, cfg);
    expect(result.valid).toBe(true);
    expect(result.warnings).toContainEqual(
      expect.stringContaining('depthShards.maxInterleavedDraws')
    );
  });

  it('warns when the per-node count alone exceeds the whole budget', () => {
    // Then even a single qualifying node is scaled down, so the per-node knob is
    // effectively dead and the budget is the real setting.
    const cfg = cloneConfig();
    cfg.depthShards.shardsPerNode = 64;
    cfg.depthShards.maxInterleavedDraws = 32;
    const result = invokeValidator(validateDepthShards, cfg);
    expect(result.valid).toBe(true);
    expect(result.warnings).toContainEqual(expect.stringContaining('exceeds maxInterleavedDraws'));
  });

  it('errors on a negative or non-integer minimum element count', () => {
    for (const bad of [-1, 2.5, Number.NaN]) {
      const cfg = cloneConfig();
      cfg.depthShards.minElements = bad;
      const result = invokeValidator(validateDepthShards, cfg);
      expect(result.valid, `minElements=${bad}`).toBe(false);
      expect(result.errors).toContainEqual(expect.stringContaining('depthShards.minElements'));
    }
  });

  it('accepts 0 minimum elements but warns', () => {
    const cfg = cloneConfig();
    cfg.depthShards.minElements = 0;
    const result = invokeValidator(validateDepthShards, cfg);
    expect(result.valid).toBe(true);
    expect(result.warnings).toContainEqual(expect.stringContaining('depthShards.minElements'));
  });
});
