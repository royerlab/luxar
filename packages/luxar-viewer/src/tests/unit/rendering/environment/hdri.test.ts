import { describe, expect, it } from 'vitest';

import { resolveEnvironmentUrl } from '../../../../rendering/environment/hdri';

describe('resolveEnvironmentUrl', () => {
  it('keeps store-relative URLs on the store origin', () => {
    expect(resolveEnvironmentUrl('env/studio.hdr', 'https://example.com/data/scene.zarr')).toBe(
      'https://example.com/data/scene.zarr/env/studio.hdr'
    );
  });

  it('rejects relative spellings that resolve to another origin', () => {
    expect(() =>
      resolveEnvironmentUrl('\\\\evil.com/env.hdr', 'https://example.com/data/scene.zarr')
    ).toThrow('outside the store origin');
  });

  it.each(['blob:https://example.com/id', 'data:image/png,abc'])(
    'rejects unsupported absolute URL %s',
    (url) => {
      expect(() => resolveEnvironmentUrl(url, undefined)).toThrow('scheme must be HTTP(S)');
    }
  );
});
