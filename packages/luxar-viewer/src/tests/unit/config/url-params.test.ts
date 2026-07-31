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
      noSliceCache: false,
      cacheDebug: false,
      clearCache: false,
      lodFade: true, // LOD cross-fade is ON by default (opt-out via ?no-lod-fade)
      lodEnergyComp: true, // streaming brightness compensation is ON (opt-out via ?no-lod-energy)
      depthSort: true, // gsplat depth sorting is ON by default (opt-out via ?depthSort=0)
      lodFinest: false, // capture-quality force-finest is OFF by default (opt-in via ?lod-finest)
      noPrefetch: false,
      prefetchDebug: false,
      cacheStats: false,
      renderer: null,
      webgpuForceWebGL: false,
      perfTimestamp: false,
      gpuBudgetMB: null,
      cacheBudgetMB: null,
      dpr: null,
    });
  });

  it('lodFade defaults ON and is disabled only by ?no-lod-fade', () => {
    expect(readUrlParams('').lodFade).toBe(true);
    expect(readUrlParams('?debug').lodFade).toBe(true);
    expect(readUrlParams('?no-lod-fade').lodFade).toBe(false);
  });

  it('lodEnergyComp defaults ON and is disabled only by ?no-lod-energy', () => {
    expect(readUrlParams('').lodEnergyComp).toBe(true);
    expect(readUrlParams('?debug').lodEnergyComp).toBe(true);
    expect(readUrlParams('?no-lod-energy').lodEnergyComp).toBe(false);
  });

  it('depthSort defaults ON and is disabled only by an explicit 0/false/off value', () => {
    expect(readUrlParams('').depthSort).toBe(true);
    expect(readUrlParams('?debug').depthSort).toBe(true);
    // Flag-only and truthy forms keep it enabled.
    expect(readUrlParams('?depthSort').depthSort).toBe(true);
    expect(readUrlParams('?depthSort=1').depthSort).toBe(true);
    expect(readUrlParams('?depthSort=true').depthSort).toBe(true);
    // The escape hatch (deterministic E2E/visual runs).
    expect(readUrlParams('?depthSort=0').depthSort).toBe(false);
    expect(readUrlParams('?depthSort=false').depthSort).toBe(false);
    expect(readUrlParams('?depthSort=OFF').depthSort).toBe(false);
  });

  it('lodFinest defaults OFF and is enabled only by ?lod-finest', () => {
    expect(readUrlParams('').lodFinest).toBe(false);
    expect(readUrlParams('?debug').lodFinest).toBe(false);
    expect(readUrlParams('?lod-finest').lodFinest).toBe(true);
  });

  it('parses dpr as a positive float, rejecting zero/negative/non-numeric', () => {
    expect(readUrlParams('?dpr=1').dpr).toBe(1);
    expect(readUrlParams('?dpr=0.5').dpr).toBe(0.5);
    expect(readUrlParams('?dpr=2').dpr).toBe(2);
    // Zero and negative pixel ratios are meaningless → null (adaptive).
    expect(readUrlParams('?dpr=0').dpr).toBeNull();
    expect(readUrlParams('?dpr=-1').dpr).toBeNull();
    expect(readUrlParams('?dpr=abc').dpr).toBeNull();
    // Flag-only form parses as NaN → null.
    expect(readUrlParams('?dpr').dpr).toBeNull();
    expect(readUrlParams('').dpr).toBeNull();
  });

  it('parses gpuBudgetMB, accepting 0 (disable) and rejecting negatives', () => {
    expect(readUrlParams('?gpuBudgetMB=1536').gpuBudgetMB).toBe(1536);
    // 0 is a valid value — flows through as the explicit "disable" signal.
    expect(readUrlParams('?gpuBudgetMB=0').gpuBudgetMB).toBe(0);
    // Negative / non-numeric → null (auto-size).
    expect(readUrlParams('?gpuBudgetMB=-5').gpuBudgetMB).toBeNull();
    expect(readUrlParams('?gpuBudgetMB=abc').gpuBudgetMB).toBeNull();
    expect(readUrlParams('').gpuBudgetMB).toBeNull();
  });

  it('parses cacheBudgetMB (cache pool override), rejecting negatives/non-numeric', () => {
    expect(readUrlParams('?cacheBudgetMB=1536').cacheBudgetMB).toBe(1536);
    expect(readUrlParams('?cacheBudgetMB=512').cacheBudgetMB).toBe(512);
    expect(readUrlParams('?cacheBudgetMB=-5').cacheBudgetMB).toBeNull();
    expect(readUrlParams('?cacheBudgetMB=abc').cacheBudgetMB).toBeNull();
    expect(readUrlParams('').cacheBudgetMB).toBeNull();
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
      '?debug&no-cache&no-slice-cache&cache-debug&clear-cache&no-prefetch&prefetch-debug&cache-stats&webgpu-force-webgl'
    );
    expect(params.debug).toBe(true);
    expect(params.noCache).toBe(true);
    expect(params.noSliceCache).toBe(true);
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

    // [G9][P5] Audit: source's `normalizeRendererParam` lowercases AND
    // trims. Pre-audit only `webgl2` lowercase + mixed-case `WebGL` were
    // tested — never trailing whitespace, mixed-case `WEBGL2`, or
    // leading whitespace on the alias.
    it('accepts whitespace and mixed-case around the webgl2 alias', () => {
      expect(readUrlParams('?renderer=WEBGL2').renderer).toBe('webgl');
      expect(readUrlParams('?renderer=%20webgl2%20').renderer).toBe('webgl');
      expect(readUrlParams('?renderer=%20WebGL2').renderer).toBe('webgl');
    });

    it('accepts whitespace around webgpu', () => {
      expect(readUrlParams('?renderer=%20webgpu%20').renderer).toBe('webgpu');
    });
  });
});

// [G6][P5] Audit: pre-audit only the "5000" overflow case was tested for
// MAX_SRC_LENGTH (4096). The boundary itself (4096), one-below, and
// one-above must each be exercised to kill off-by-one mutants in the
// `src.length > MAX_SRC_LENGTH` predicate.
describe('normalizeDataSourceUrl — MAX_SRC_LENGTH boundary', () => {
  const MAX = 4096;
  // Build a URL that is EXACTLY `length` characters long. Use a short
  // scheme prefix so the URL parses; pad with valid path characters.
  function urlOfLength(length: number): string {
    const prefix = 'https://e.com/';
    const padLen = length - prefix.length;
    return prefix + 'a'.repeat(Math.max(0, padLen));
  }

  it('accepts a URL whose length is exactly MAX_SRC_LENGTH (4096)', () => {
    const src = urlOfLength(MAX);
    expect(src.length).toBe(MAX);
    expect(normalizeDataSourceUrl(src)).toBe(src);
  });

  it('accepts a URL whose length is MAX_SRC_LENGTH - 1 (4095)', () => {
    const src = urlOfLength(MAX - 1);
    expect(src.length).toBe(MAX - 1);
    expect(normalizeDataSourceUrl(src)).toBe(src);
  });

  it('rejects a URL whose length is MAX_SRC_LENGTH + 1 (4097)', () => {
    const src = urlOfLength(MAX + 1);
    expect(src.length).toBe(MAX + 1);
    expect(normalizeDataSourceUrl(src)).toBeNull();
  });

  // Empty string is its own boundary at the OTHER end of the range.
  it('rejects an empty string (length 0)', () => {
    expect(normalizeDataSourceUrl('')).toBeNull();
  });
});

// [G7][P5] / [M3][P11] Audit: pre-audit `hasUnsafeSrcCharacter` was only
// covered for `\0`, `\r`, `\n`. The source rejects the full
// 0x00..0x1f range plus 0x7f (DEL) plus `<` and `>` — fixed-array
// iteration means every character class needs at least one direct test
// or a mutation could narrow the predicate to a single char.
describe('normalizeDataSourceUrl — hasUnsafeSrcCharacter coverage', () => {
  it('rejects every control character in 0x01..0x1f', () => {
    for (let code = 0x01; code <= 0x1f; code++) {
      const src = `https://example.com/${String.fromCharCode(code)}bad.zarr`;
      expect(normalizeDataSourceUrl(src)).toBeNull();
    }
  });

  it('rejects 0x7f (DEL)', () => {
    const src = `https://example.com/${String.fromCharCode(0x7f)}bad.zarr`;
    expect(normalizeDataSourceUrl(src)).toBeNull();
  });

  it('rejects literal `<` in the path', () => {
    // The decoded form must be rejected even though the URLSearchParams
    // decoder also handles percent-encoded `%3C` (which is the existing
    // test). Both routes must converge on null.
    expect(normalizeDataSourceUrl('https://example.com/<script')).toBeNull();
  });

  it('rejects literal `>` in the path', () => {
    expect(normalizeDataSourceUrl('https://example.com/end>here')).toBeNull();
  });

  it('accepts the boundary character 0x20 (space) once trimmed out', () => {
    // Space is 0x20 — exactly above the control-char range. The source
    // trims surrounding whitespace; internal spaces would `new URL()` to
    // a percent-encoded form which is fine for a valid URL.
    expect(normalizeDataSourceUrl('  https://example.com/ok  ')).toBe('https://example.com/ok');
  });

  it('accepts the boundary character 0x21 (!) directly above DEL', () => {
    // 0x21 = `!` — not in the control range and not in {<,>}.
    expect(normalizeDataSourceUrl('https://example.com/!ok')).toBe('https://example.com/!ok');
  });
});

// [G17][P5] Audit: buildDataSourceBrowserUrl had no test for: URL with
// hash but no search; src containing query-reserved characters; empty
// src. Each exercises a separate code path in URLSearchParams.set + the
// optional hash concatenation.
describe('buildDataSourceBrowserUrl — edge cases', () => {
  it('preserves a hash fragment even when there is no existing search', () => {
    const url = buildDataSourceBrowserUrl('datasets/picked.zarr', {
      pathname: '/viewer',
      search: '',
      hash: '#tab=cache',
    });
    expect(url).toBe('/viewer?src=datasets%2Fpicked.zarr#tab=cache');
  });

  it('percent-encodes query-reserved characters in src (& = ?)', () => {
    // URLSearchParams.toString must encode these — a mutation that
    // bypassed it (e.g. by hand-concatenating) would let `&foo=bar`
    // smuggle a second query param.
    const url = buildDataSourceBrowserUrl('datasets/file?evil=true&x=1', {
      pathname: '/viewer',
      search: '',
    });
    // Just assert the encoded forms appear and no raw `&` or `?` leaked
    // INTO the value portion of src.
    expect(url).toContain('src=datasets%2Ffile%3Fevil%3Dtrue%26x%3D1');
  });

  it('accepts an empty src and emits src= with empty value', () => {
    // [P5] boundary: empty string is a valid input to URLSearchParams.set,
    // even if the upstream normalizer would reject it. Document the
    // builder's behaviour rather than make assumptions.
    const url = buildDataSourceBrowserUrl('', {
      pathname: '/viewer',
      search: '',
    });
    expect(url).toBe('/viewer?src=');
  });
});

// [G19][P5] Audit: `config` from src/config/index.ts is not directly
// tested for shape. Pin the contract — defaultZarrPath === '' means
// "show dataset browser". A mutation that swapped it for a sentinel
// would silently change first-time UX behaviour.
describe('config.defaultZarrPath', () => {
  it('is the empty string (browser-mode default)', async () => {
    const { config } = await import('../../../config');
    expect(config.defaultZarrPath).toBe('');
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

  it('strips trailing slashes from src for canonical URL storage', () => {
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
