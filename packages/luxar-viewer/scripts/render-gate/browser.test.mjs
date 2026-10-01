import { describe, expect, it } from 'vitest';

import { caseUrl, selectBackends } from './browser.mjs';

describe('render-gate backend URLs', () => {
  it('forces the WebGPURenderer WebGL2 fallback for the exact arm', () => {
    expect(caseUrl('http://localhost:4801', {}, 'webgpu-gl', 1, 'lodFinest')).toBe(
      'http://localhost:4801/?debug&dpr=1&renderer=webgpu&webgpuForceWebgl&lodFinest'
    );
  });
});

describe('render-gate backend selection', () => {
  it('keeps only perf backends when --suite all requests every exact arm', () => {
    expect(selectBackends(['webgl', 'webgpu', 'webgpu-gl'], ['webgl', 'webgpu'])).toEqual([
      'webgl',
      'webgpu',
    ]);
  });

  it('uses allowed defaults and respects a case-specific backend restriction', () => {
    expect(selectBackends(null, ['webgl', 'webgpu-gl'])).toEqual(['webgl', 'webgpu-gl']);
    expect(selectBackends(['webgl', 'webgpu-gl'], ['webgpu-gl'])).toEqual(['webgpu-gl']);
  });
});
