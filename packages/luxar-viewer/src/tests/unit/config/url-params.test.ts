import { describe, expect, it } from 'vitest';
import { readUrlParams } from '../../../config/url-params';

describe('readUrlParams', () => {
  it('returns null/false defaults for an empty query string', () => {
    const params = readUrlParams('');
    expect(params).toEqual({
      src: null,
      theme: null,
      debug: false,
      noCache: false,
      cacheDebug: false,
      clearCache: false,
      noPrefetch: false,
      prefetchDebug: false,
    });
  });

  it('parses src and theme as strings', () => {
    const params = readUrlParams('?src=https://example.com/data.zarr&theme=light');
    expect(params.src).toBe('https://example.com/data.zarr');
    expect(params.theme).toBe('light');
  });

  it('treats valueless flags as boolean true', () => {
    const params = readUrlParams('?debug&no-cache&cache-debug&clear-cache&no-prefetch&prefetch-debug');
    expect(params.debug).toBe(true);
    expect(params.noCache).toBe(true);
    expect(params.cacheDebug).toBe(true);
    expect(params.clearCache).toBe(true);
    expect(params.noPrefetch).toBe(true);
    expect(params.prefetchDebug).toBe(true);
  });

  it('accepts a leading question mark or omits it', () => {
    expect(readUrlParams('?debug').debug).toBe(true);
    expect(readUrlParams('debug').debug).toBe(true);
  });

  it('treats unknown parameters as inert (no fields added)', () => {
    const params = readUrlParams('?unknown=foo');
    expect(params.src).toBeNull();
    expect(params.debug).toBe(false);
  });
});
