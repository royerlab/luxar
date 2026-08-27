import { describe, expect, it } from 'vitest';

import { createKTX2TextureDecoder } from '../../../rendering/ktx2-texture-decoder';
import type { Renderer } from '../../../rendering/renderer-capabilities';

describe('createKTX2TextureDecoder', () => {
  it('rejects clearly when the renderer has no compressed-texture target', async () => {
    const renderer = {
      isWebGPURenderer: true,
      hasFeature: () => false,
    } as unknown as Renderer;

    const decode = createKTX2TextureDecoder(renderer);

    await expect(
      decode('mesh/texture', new Uint8Array(0), { width: 1, height: 1, channels: 4 })
    ).rejects.toThrow(/mesh\/texture.*ktx2.*raw.*jpeg/i);
  });
});
