import { describe, expect, it } from 'vitest';
import { normalizeDataSourceUrl, readUrlParams } from '../../../config/url-params';

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

  it('parses and trims valid src and theme strings', () => {
    const params = readUrlParams('?src=%20https://example.com/data.zarr%20&theme=light');
    expect(params.src).toBe('https://example.com/data.zarr');
    expect(params.theme).toBe('light');
  });

  it('accepts relative and root-relative data source URLs', () => {
    expect(readUrlParams('?src=datasets/test.zarr').src).toBe('datasets/test.zarr');
    expect(readUrlParams('?src=/datasets/test.zarr').src).toBe('/datasets/test.zarr');
  });

  it('rejects unsupported or unsafe data source URL values', () => {
    expect(readUrlParams('?src=file:///etc/passwd').src).toBeNull();
    expect(readUrlParams('?src=javascript:alert(1)').src).toBeNull();
    expect(readUrlParams('?src=data:text/html,boom').src).toBeNull();
    expect(readUrlParams('?src=//example.com/data.zarr').src).toBeNull();
    expect(readUrlParams('?src=https://example.com/%3Cscript%3E').src).toBeNull();
  });

  it('rejects empty, control-character, and overly long src values', () => {
    expect(normalizeDataSourceUrl('   ')).toBeNull();
    expect(normalizeDataSourceUrl('datasets/\u0000bad.zarr')).toBeNull();
    expect(normalizeDataSourceUrl(`https://example.com/${'a'.repeat(5000)}.zarr`)).toBeNull();
  });

  it('treats valueless flags as boolean true', () => {
    const params = readUrlParams(
      '?debug&no-cache&cache-debug&clear-cache&no-prefetch&prefetch-debug'
    );
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
