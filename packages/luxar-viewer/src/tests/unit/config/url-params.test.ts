import { afterEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import {
  buildDataSourceBrowserUrl,
  getParam,
  hasParam,
  normalizeDataSourceUrl,
  readUrlParams,
  replaceBrowserDataSourceUrl,
  URL_PARAM_KEYS,
} from '../../../config/url-params';

const PACKAGE_ROOT = new URL('../../../../', import.meta.url);

/** The old kebab spellings retired by the 2026-09 camelCase hard cut. */
const RETIRED_KEBAB_SPELLINGS = [
  'no-cache',
  'no-slice-cache',
  'no-opfs',
  'cache-debug',
  'clear-cache',
  'no-lod-fade',
  'no-links',
  'no-lod-energy',
  'lod-finest',
  'no-blend-warmup',
  'no-prefetch',
  'prefetch-debug',
  'cache-stats',
  'webgpu-force-webgl',
  'perf-timestamp',
  'lod-bias',
  'no-density-guard',
  'density-cap',
  'bake-env',
  'env-resolution',
] as const;

describe('URL_PARAM_KEYS', () => {
  it('has unique wire spellings', () => {
    const values = Object.values(URL_PARAM_KEYS);
    expect(new Set(values).size).toBe(values.length);
  });

  it('is an identity table: every key IS its wire spelling', () => {
    // The identity is what makes `URL_PARAM_KEYS.noCache` read as the flag it
    // names and keeps `UrlParamKey` equal to the set of wire spellings.
    const nonIdentity = Object.entries(URL_PARAM_KEYS).filter(([key, wire]) => key !== wire);
    expect(nonIdentity).toEqual([]);
  });

  it('spells every parameter camelCase (no kebab-case, no exceptions)', () => {
    // The camelCase convention decided for the release. `bakeEnv` and
    // `envResolution` are emitted by the Python `luxar env bake` CLI
    // (cli/env_ops/bake.py), so a wire change here must land on that side too.
    const kebab = Object.values(URL_PARAM_KEYS).filter((wire) => !/^[a-z][A-Za-z0-9]*$/.test(wire));
    expect(kebab).toEqual([]);
  });

  it('no longer recognizes any retired kebab spelling (hard cut, no alias)', () => {
    const params = readUrlParams(`?${RETIRED_KEBAB_SPELLINGS.join('&')}`);
    expect(params).toEqual(readUrlParams(''));
    for (const retired of RETIRED_KEBAB_SPELLINGS) {
      expect(Object.values(URL_PARAM_KEYS) as string[]).not.toContain(retired);
    }
  });

  it.each([
    ['packages/luxar-viewer/README.md', 'README.md', '### URL Parameters', /^##/m],
    [
      'docs/guides/user/VIEWER_GUIDE.md',
      '../../docs/guides/user/VIEWER_GUIDE.md',
      '## URL Parameters',
      /^## /m,
    ],
  ])('documents every wire spelling in %s', (_label, relPath, heading, nextHeading) => {
    // Doc parity: the docs are prose, so nothing else would notice a renamed
    // or added parameter that never reached them. Slice the section under
    // the heading and require each spelling to appear as a parameter
    // (`?noCache`, `&noCache`, or a table cell `` `noCache` ``), not merely
    // as a substring of some other word.
    const text = readFileSync(fileURLToPath(new URL(relPath, PACKAGE_ROOT)), 'utf8');
    const start = text.indexOf(heading);
    expect(start, `${relPath} has a "${heading}" section`).toBeGreaterThan(-1);
    const body = text.slice(start + heading.length);
    const end = body.search(nextHeading);
    const section = end === -1 ? body : body.slice(0, end);
    const undocumented = Object.values(URL_PARAM_KEYS).filter((wire) => {
      const escaped = wire.replace(/[.*+?^${}()|[\]\\-]/g, '\\$&');
      return !new RegExp(`(?:[?&]|\\| \`)${escaped}(?![A-Za-z0-9-])`).test(section);
    });
    expect(undocumented).toEqual([]);
  });
});

describe('deprecated URL parameter aliases', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('resolves a flag through an injected alias and warns exactly once per alias', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const aliases = { 'legacy-no-cache': URL_PARAM_KEYS.noCache };
    const params = new URLSearchParams('?legacy-no-cache');
    expect(hasParam(params, URL_PARAM_KEYS.noCache, aliases)).toBe(true);
    expect(hasParam(params, URL_PARAM_KEYS.noCache, aliases)).toBe(true);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0]?.[0]).toContain('"legacy-no-cache" is deprecated');
    expect(warn.mock.calls[0]?.[0]).toContain('"noCache"');
  });

  it('returns the value supplied under an alias', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const aliases = { 'legacy-dpr': URL_PARAM_KEYS.dpr };
    expect(getParam(new URLSearchParams('?legacy-dpr=2'), URL_PARAM_KEYS.dpr, aliases)).toBe('2');
    expect(getParam(new URLSearchParams('?other=1'), URL_PARAM_KEYS.dpr, aliases)).toBeNull();
  });

  it('prefers the canonical spelling and does not warn when it is present', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const aliases = { 'legacy-theme': URL_PARAM_KEYS.theme };
    const params = new URLSearchParams('?theme=dark&legacy-theme=light');
    expect(getParam(params, URL_PARAM_KEYS.theme, aliases)).toBe('dark');
    expect(warn).not.toHaveBeenCalled();
  });

  it('is inert for an alias that maps to a different key', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const aliases = { 'legacy-kiosk': URL_PARAM_KEYS.kiosk };
    expect(hasParam(new URLSearchParams('?legacy-kiosk'), URL_PARAM_KEYS.debug, aliases)).toBe(
      false
    );
    expect(warn).not.toHaveBeenCalled();
  });
});

describe('readUrlParams', () => {
  it('returns null/false defaults for an empty query string', () => {
    const params = readUrlParams('');
    expect(params).toEqual({
      src: null,
      theme: null,
      title: null,
      control: null,
      controlToken: null,
      controlAllowCrossOrigin: false,
      panel: null,
      debug: false,
      kiosk: false,
      noCache: false,
      noSliceCache: false,
      noOpfs: false,
      opfsReadConcurrency: null,
      cacheDebug: false,
      clearCache: false,
      lodFade: true,
      allowLinks: true, // element links are ON by default (opt-out via ?noLinks)
      lodEnergyComp: true, // streaming brightness compensation is ON (opt-out via ?noLodEnergy)
      blendWarmup: true, // WebGL blend-variant warm-up is ON by default (opt-out via ?noBlendWarmup)
      depthSort: true, // gsplat depth sorting is ON by default (opt-out via ?depthSort=0)
      densityGuard: true, // projected-density guard is ON by default (opt-out via ?noDensityGuard)
      densityCap: null, // configured cap unless ?densityCap=N
      lodFinest: false, // capture-quality force-finest is OFF by default (opt-in via ?lodFinest)
      lodBias: null, // normal LOD thresholds unless ?lodBias=N
      noPrefetch: false,
      prefetchDebug: false,
      cacheStats: false,
      renderer: null,
      webgpuForceWebGL: false,
      perfTimestamp: false,
      gpuBudgetMB: null,
      cacheBudgetMB: null,
      dpr: null,
      input: null,
      lineJoin: null,
      linePrimitive: null,
      bakeEnv: false, // the `luxar env bake` driver's one-shot (opt-in via ?bakeEnv)
      probe: null,
      envResolution: null,
    });
  });

  it('parses the environment bake parameters (?bakeEnv&probe=&envResolution=)', () => {
    expect(readUrlParams('?bakeEnv').bakeEnv).toBe(true);
    expect(readUrlParams('?bakeEnv&probe=node:clusters/shell&envResolution=256')).toMatchObject({
      bakeEnv: true,
      probe: 'node:clusters/shell',
      envResolution: 256,
    });
    expect(readUrlParams('?probe=%201,2,3%20').probe).toBe('1,2,3');
    expect(readUrlParams('?probe=').probe).toBeNull();
    expect(readUrlParams('?envResolution=abc').envResolution).toBeNull();
    expect(readUrlParams('?envResolution=0').envResolution).toBe(16);
    expect(readUrlParams('?envResolution=8192').envResolution).toBe(1024);
  });

  it('parses ?densityCap= as a positive float, anything else → null', () => {
    expect(readUrlParams('?densityCap=8').densityCap).toBe(8);
    expect(readUrlParams('?densityCap=2.5').densityCap).toBe(2.5);
    expect(readUrlParams('?densityCap=0').densityCap).toBeNull();
    expect(readUrlParams('?densityCap=-4').densityCap).toBeNull();
    expect(readUrlParams('?densityCap=lots').densityCap).toBeNull();
    expect(readUrlParams('?densityCap=').densityCap).toBeNull();
  });

  it('parses ?title=, decoding and trimming; blank collapses to null', () => {
    expect(readUrlParams('?title=global_rivers_earth').title).toBe('global_rivers_earth');
    expect(readUrlParams('?title=Rivers%20of%20Earth').title).toBe('Rivers of Earth');
    expect(readUrlParams('?title=%20%20').title).toBeNull();
    expect(readUrlParams('?title=').title).toBeNull();
  });

  it('lodFade defaults ON and is disabled only by ?noLodFade', () => {
    expect(readUrlParams('').lodFade).toBe(true);
    expect(readUrlParams('?debug').lodFade).toBe(true);
    expect(readUrlParams('?noLodFade').lodFade).toBe(false);
  });

  it('allowLinks defaults ON and is disabled only by ?noLinks', () => {
    expect(readUrlParams('').allowLinks).toBe(true);
    expect(readUrlParams('?debug').allowLinks).toBe(true);
    expect(readUrlParams('?noLinks').allowLinks).toBe(false);
    // Not confusable with the other no-* flags that share a prefix.
    expect(readUrlParams('?noLodFade').allowLinks).toBe(true);
  });

  it('lodEnergyComp defaults ON and is disabled only by ?noLodEnergy', () => {
    expect(readUrlParams('').lodEnergyComp).toBe(true);
    expect(readUrlParams('?debug').lodEnergyComp).toBe(true);
    expect(readUrlParams('?noLodEnergy').lodEnergyComp).toBe(false);
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

  it('lodFinest defaults OFF and is enabled only by ?lodFinest', () => {
    expect(readUrlParams('').lodFinest).toBe(false);
    expect(readUrlParams('?debug').lodFinest).toBe(false);
    expect(readUrlParams('?lodFinest').lodFinest).toBe(true);
  });

  it('parses ?lodBias= as a positive float, anything else → null', () => {
    expect(readUrlParams('').lodBias).toBeNull();
    expect(readUrlParams('?lodBias=4').lodBias).toBe(4);
    expect(readUrlParams('?lodBias=0.25').lodBias).toBe(0.25);
    expect(readUrlParams('?lodBias=0').lodBias).toBeNull();
    expect(readUrlParams('?lodBias=-2').lodBias).toBeNull();
    expect(readUrlParams('?lodBias=lots').lodBias).toBeNull();
    expect(readUrlParams('?lodBias=').lodBias).toBeNull();
  });

  it('blendWarmup defaults ON and is disabled only by ?noBlendWarmup', () => {
    expect(readUrlParams('').blendWarmup).toBe(true);
    expect(readUrlParams('?debug').blendWarmup).toBe(true);
    expect(readUrlParams('?noBlendWarmup').blendWarmup).toBe(false);
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

  it('parses lineJoin, distinguishing "no override" from an explicit none', () => {
    expect(readUrlParams('?lineJoin=none').lineJoin).toBe('none');
    expect(readUrlParams('?lineJoin=miter').lineJoin).toBe('miter');
    // Case- and whitespace-insensitive, like the other enum params.
    expect(readUrlParams('?lineJoin=MITER').lineJoin).toBe('miter');
    // Unrecognised values must NOT silently mean "no join geometry" — they mean
    // "no override", so each node's authored style (or the default) still wins.
    expect(readUrlParams('?lineJoin=bevel').lineJoin).toBeNull();
    expect(readUrlParams('?lineJoin=round').lineJoin).toBeNull();
    expect(readUrlParams('?lineJoin=').lineJoin).toBeNull();
    expect(readUrlParams('?lineJoin').lineJoin).toBeNull();
    expect(readUrlParams('').lineJoin).toBeNull();
  });

  it('parses linePrimitive, rejecting unknown values as "no override"', () => {
    expect(readUrlParams('?linePrimitive=screen-space').linePrimitive).toBe('screen-space');
    expect(readUrlParams('?linePrimitive=capsule').linePrimitive).toBe('capsule');
    // Case- and whitespace-insensitive, like the other enum params.
    expect(readUrlParams('?linePrimitive=CAPSULE').linePrimitive).toBe('capsule');
    // Unrecognised values must never silently select a primitive — the
    // deleted 'volumetric' primitive included (#1352 deletion).
    expect(readUrlParams('?linePrimitive=volumetric').linePrimitive).toBeNull();
    expect(readUrlParams('?linePrimitive=quads').linePrimitive).toBeNull();
    expect(readUrlParams('?linePrimitive=').linePrimitive).toBeNull();
    expect(readUrlParams('?linePrimitive').linePrimitive).toBeNull();
    expect(readUrlParams('').linePrimitive).toBeNull();
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

  it('parses opfsReadConcurrency as a positive integer', () => {
    expect(readUrlParams('?opfsReadConcurrency=16').opfsReadConcurrency).toBe(16);
    expect(readUrlParams('?opfsReadConcurrency=0').opfsReadConcurrency).toBeNull();
    expect(readUrlParams('?opfsReadConcurrency=1.5').opfsReadConcurrency).toBeNull();
    expect(readUrlParams('?opfsReadConcurrency=abc').opfsReadConcurrency).toBeNull();
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
      '?debug&noCache&noSliceCache&noOpfs&cacheDebug&clearCache&noPrefetch&prefetchDebug&cacheStats&webgpuForceWebgl'
    );
    expect(params.debug).toBe(true);
    expect(params.noCache).toBe(true);
    expect(params.noSliceCache).toBe(true);
    expect(params.noOpfs).toBe(true);
    expect(params.cacheDebug).toBe(true);
    expect(params.clearCache).toBe(true);
    expect(params.noPrefetch).toBe(true);
    expect(params.prefetchDebug).toBe(true);
    expect(params.cacheStats).toBe(true);
    expect(params.webgpuForceWebGL).toBe(true);
  });

  it('?cacheStats is independent of ?cacheDebug (different concerns)', () => {
    expect(readUrlParams('?cacheStats').cacheStats).toBe(true);
    expect(readUrlParams('?cacheStats').cacheDebug).toBe(false);
    expect(readUrlParams('?cacheDebug').cacheStats).toBe(false);
    expect(readUrlParams('?cacheDebug').cacheDebug).toBe(true);
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

  describe('?webgpuForceWebgl', () => {
    it('parses the diagnostic WebGPURenderer WebGL-backend flag', () => {
      expect(readUrlParams('').webgpuForceWebGL).toBe(false);
      expect(readUrlParams('?webgpuForceWebgl').webgpuForceWebGL).toBe(true);
      expect(readUrlParams('?renderer=webgpu&webgpuForceWebgl').webgpuForceWebGL).toBe(true);
    });
  });

  describe('?perfTimestamp', () => {
    it('parses the GPU timestamp-query opt-in flag', () => {
      expect(readUrlParams('').perfTimestamp).toBe(false);
      expect(readUrlParams('?perfTimestamp').perfTimestamp).toBe(true);
      expect(readUrlParams('?renderer=webgpu&perfTimestamp').perfTimestamp).toBe(true);
    });

    it('does not affect other flags when present alone', () => {
      const params = readUrlParams('?perfTimestamp');
      expect(params.perfTimestamp).toBe(true);
      expect(params.debug).toBe(false);
      expect(params.webgpuForceWebGL).toBe(false);
    });
  });

  describe('?input=', () => {
    it('is null when absent or flag-only', () => {
      expect(readUrlParams('').input).toBeNull();
      expect(readUrlParams('?input').input).toBeNull();
    });

    it('accepts touch and mouse case-insensitively', () => {
      expect(readUrlParams('?input=touch').input).toBe('touch');
      expect(readUrlParams('?input=Mouse').input).toBe('mouse');
      expect(readUrlParams('?input=%20TOUCH%20').input).toBe('touch');
    });

    it('treats unrecognised values as no override', () => {
      expect(readUrlParams('?input=pen').input).toBeNull();
      expect(readUrlParams('?input=1').input).toBeNull();
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

  it('drops a stale ?title= but keeps the other params when src changes', () => {
    // ?title= names the dataset the server was started with. Switching
    // datasets in the browser modal must not carry it over, or a reload (or
    // a shared link) titles the tab after a scene it no longer shows.
    const url = buildDataSourceBrowserUrl('datasets/next.zarr', {
      pathname: '/viewer',
      search: '?src=datasets%2Fprev.zarr&title=Prev%20Scene&theme=dark',
    });
    expect(url).toBe('/viewer?src=datasets%2Fnext.zarr&theme=dark');
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
