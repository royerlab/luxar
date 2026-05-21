import { describe, expect, it, vi } from 'vitest';
import {
  buildDataSourceBrowserUrl,
  normalizeDataSourceUrl,
  readUrlParams,
  replaceBrowserDataSourceUrl,
} from '../../../config/url-params';

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
      cacheStats: false,
      renderer: null,
      webgpuForceWebGL: false,
      perfTimestamp: false,
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
    expect(normalizeDataSourceUrl('datasets/\rbad.zarr')).toBeNull();
    expect(normalizeDataSourceUrl('datasets/\nbad.zarr')).toBeNull();
  });

  it('rejects mixed-case javascript: and other unsafe schemes', () => {
    // Mixed-case scheme must not bypass the deny list — `new URL()` lowercases
    // the protocol so the http/https check still rejects it.
    expect(normalizeDataSourceUrl('JaVaScRiPt:alert(1)')).toBeNull();
    expect(normalizeDataSourceUrl('vbscript:msgbox(1)')).toBeNull();
    expect(normalizeDataSourceUrl('blob:https://example.com/abc')).toBeNull();
    expect(normalizeDataSourceUrl('about:blank')).toBeNull();
    // Leading whitespace must not smuggle a dangerous scheme past trim().
    expect(normalizeDataSourceUrl('\t javascript:alert(1)')).toBeNull();
  });

  it('strips trailing slashes per CLAUDE.md zarr-loader gotcha', () => {
    expect(normalizeDataSourceUrl('https://example.com/data.zarr/')).toBe(
      'https://example.com/data.zarr'
    );
    expect(normalizeDataSourceUrl('https://example.com/data.zarr///')).toBe(
      'https://example.com/data.zarr'
    );
    expect(normalizeDataSourceUrl('datasets/test.zarr/')).toBe('datasets/test.zarr');
    // A bare "/" trims to empty and is rejected.
    expect(normalizeDataSourceUrl('/')).toBeNull();
  });

  // Mixed-case schemes are valid HTTP(S) per RFC 3986 §3.1.
  // normalize-data-source-url canonicalizes via url.href so every
  // downstream helper sees the same lowercase scheme form.
  it('canonicalizes mixed-case HTTP(S) schemes via URL.href', () => {
    expect(normalizeDataSourceUrl('HTTPS://Example.com/data.zarr')).toBe(
      'https://example.com/data.zarr'
    );
    expect(normalizeDataSourceUrl('Http://example.com/data.zarr')).toBe(
      'http://example.com/data.zarr'
    );
    // Trailing slashes still trimmed after canonicalization.
    expect(normalizeDataSourceUrl('HTTPS://example.com/foo.zarr/')).toBe(
      'https://example.com/foo.zarr'
    );
  });

  it('treats valueless flags as boolean true', () => {
    const params = readUrlParams(
      '?debug&no-cache&cache-debug&clear-cache&no-prefetch&prefetch-debug&cache-stats&webgpu-force-webgl'
    );
    expect(params.debug).toBe(true);
    expect(params.noCache).toBe(true);
    expect(params.cacheDebug).toBe(true);
    expect(params.clearCache).toBe(true);
    expect(params.noPrefetch).toBe(true);
    expect(params.prefetchDebug).toBe(true);
    expect(params.cacheStats).toBe(true);
    expect(params.webgpuForceWebGL).toBe(true);
  });

  it('cache-stats is independent of cache-debug (different concerns)', () => {
    expect(readUrlParams('?cache-stats').cacheStats).toBe(true);
    expect(readUrlParams('?cache-stats').cacheDebug).toBe(false);
    expect(readUrlParams('?cache-debug').cacheStats).toBe(false);
    expect(readUrlParams('?cache-debug').cacheDebug).toBe(true);
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

  describe('?webgpu-force-webgl', () => {
    it('parses the diagnostic WebGPURenderer WebGL-backend flag', () => {
      expect(readUrlParams('').webgpuForceWebGL).toBe(false);
      expect(readUrlParams('?webgpu-force-webgl').webgpuForceWebGL).toBe(true);
      expect(readUrlParams('?renderer=webgpu&webgpu-force-webgl').webgpuForceWebGL).toBe(true);
    });
  });

  describe('?perf-timestamp', () => {
    it('parses the GPU timestamp-query opt-in flag', () => {
      expect(readUrlParams('').perfTimestamp).toBe(false);
      expect(readUrlParams('?perf-timestamp').perfTimestamp).toBe(true);
      expect(readUrlParams('?renderer=webgpu&perf-timestamp').perfTimestamp).toBe(true);
    });

    it('does not affect other flags when present alone', () => {
      const params = readUrlParams('?perf-timestamp');
      expect(params.perfTimestamp).toBe(true);
      expect(params.debug).toBe(false);
      expect(params.webgpuForceWebGL).toBe(false);
    });
  });

  describe('?renderer=', () => {
    it('returns null when absent', () => {
      expect(readUrlParams('').renderer).toBeNull();
      expect(readUrlParams('?src=foo').renderer).toBeNull();
    });

    it('parses webgl / webgpu case-insensitively', () => {
      expect(readUrlParams('?renderer=webgl').renderer).toBe('webgl');
      expect(readUrlParams('?renderer=WebGL').renderer).toBe('webgl');
      expect(readUrlParams('?renderer=webgpu').renderer).toBe('webgpu');
      expect(readUrlParams('?renderer=WEBGPU').renderer).toBe('webgpu');
    });

    it('accepts webgl2 as an alias for webgl', () => {
      // Some users may type `?renderer=webgl2` since the underlying
      // backend is WebGL2 — accept it as a synonym.
      expect(readUrlParams('?renderer=webgl2').renderer).toBe('webgl');
    });

    it('treats unrecognized values as null (no override)', () => {
      // Defer-to-env behaviour for typos / future values we don't
      // know about yet. Avoids breaking the env-var precedence chain
      // when the URL is misspelled.
      expect(readUrlParams('?renderer=opengl').renderer).toBeNull();
      expect(readUrlParams('?renderer=').renderer).toBeNull();
      expect(readUrlParams('?renderer').renderer).toBeNull();
    });
  });
});

describe('buildDataSourceBrowserUrl', () => {
  it('sets src while preserving existing params and hash fragments', () => {
    const url = buildDataSourceBrowserUrl('https://example.com/picked.zarr', {
      pathname: '/viewer',
      search: '?debug&theme=light&src=old.zarr',
      hash: '#panel',
    });

    expect(url).toBe(
      '/viewer?debug=&theme=light&src=https%3A%2F%2Fexample.com%2Fpicked.zarr#panel'
    );
  });

  it('adds a query string when the current URL has no params', () => {
    const url = buildDataSourceBrowserUrl('datasets/picked.zarr', {
      pathname: '/viewer',
      search: '',
    });

    expect(url).toBe('/viewer?src=datasets%2Fpicked.zarr');
  });

  it('strips trailing slashes from src so downstream zarr fetches do not 404', () => {
    const url = buildDataSourceBrowserUrl('http://example.com/data.zarr///', {
      pathname: '/viewer',
      search: '',
    });

    expect(url).toBe('/viewer?src=http%3A%2F%2Fexample.com%2Fdata.zarr');
  });
});

describe('replaceBrowserDataSourceUrl', () => {
  it('uses history.replaceState with the centralized URL builder', () => {
    const replaceState = vi.fn();

    const ok = replaceBrowserDataSourceUrl('datasets/picked.zarr', {
      location: { pathname: '/viewer', search: '?debug', hash: '#dataset' },
      history: { replaceState },
    });

    expect(ok).toBe(true);
    expect(replaceState).toHaveBeenCalledWith(
      {},
      '',
      '/viewer?debug=&src=datasets%2Fpicked.zarr#dataset'
    );
  });

  it('returns false when replaceState is unavailable or blocked', () => {
    const ok = replaceBrowserDataSourceUrl('datasets/picked.zarr', {
      location: { pathname: '/viewer', search: '' },
      history: {
        replaceState: () => {
          throw new Error('blocked');
        },
      },
    });

    expect(ok).toBe(false);
  });

  it('normalizes trailing slashes in src so callers do not have to', () => {
    const replaceState = vi.fn();

    replaceBrowserDataSourceUrl('http://example.com/', {
      location: { pathname: '/viewer', search: '' },
      history: { replaceState },
    });

    expect(replaceState).toHaveBeenCalledWith({}, '', '/viewer?src=http%3A%2F%2Fexample.com');
  });
});
