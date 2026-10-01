import { describe, expect, it } from 'vitest';

import { caseUrl } from './browser.mjs';

describe('render-gate backend URLs', () => {
  it('forces the WebGPURenderer WebGL2 fallback for the exact arm', () => {
    expect(caseUrl('http://localhost:4801', {}, 'webgpu-gl', 1, 'lodFinest')).toBe(
      'http://localhost:4801/?debug&dpr=1&renderer=webgpu&webgpuForceWebgl&lodFinest'
    );
  });
});
