/**
 * Unit tests for loadOverlayConfigs — reads overlay configs from the
 * `overlays/` group in a zarr scene, validates the `overlay_` type prefix,
 * and returns them sorted by z_index.
 *
 * Only the zarr store/location boundary is mocked. The facade `zarr.open`
 * delegates to `zarrita.open`, so we mock `zarrita` (matching the sibling
 * chunk-bounds-loader / color-loader tests). The store's `contents()` method
 * drives child enumeration via `hasContentsMethod`.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('zarrita', async () => {
  const actual = await vi.importActual<typeof import('zarrita')>('zarrita');
  return {
    ...actual,
    open: vi.fn(),
  };
});

import { open as zarrOpen } from 'zarrita';
import {
  loadOverlayConfigs,
  MAX_OVERLAY_HTML_CHARS,
} from '../../../../../data/loaders/overlays/overlay-loader';
import { log, Modules } from '../../../../../utils/log';

const mockOpen = vi.mocked(zarrOpen);

/** A minimal Location whose resolve() echoes the resolved path. */
function makeRootLocation(): import('zarrita').Location<import('zarrita').Readable> {
  return {
    resolve: vi.fn((name: string) => ({ path: name })),
  } as unknown as import('zarrita').Location<import('zarrita').Readable>;
}

/** A store exposing a consolidated-metadata style contents() listing. */
function makeStoreWithContents(paths: string[]): import('zarrita').Readable {
  return {
    get: vi.fn(),
    contents: vi.fn(async () => paths.map((path) => ({ path, kind: 'group' as const }))),
  } as unknown as import('zarrita').Readable;
}

/** A bare store with neither contents() nor list(). */
function makeStoreNoListing(): import('zarrita').Readable {
  return { get: vi.fn() } as unknown as import('zarrita').Readable;
}

/**
 * Wire up zarr.open to resolve groups by the resolved location's `path`.
 * `groups` maps a path → the attrs object (or a thrown error if absent and
 * `failOn` matches).
 */
function wireOpen(
  attrsByPath: Record<string, Record<string, unknown>>,
  opts: { rejectPaths?: string[] } = {}
) {
  mockOpen.mockImplementation(async (loc: unknown) => {
    const path = (loc as { path?: string } | undefined)?.path ?? '';
    if (opts.rejectPaths?.includes(path)) {
      throw new Error(`open failed for ${path}`);
    }
    if (path === 'overlays') {
      // The overlays group itself — attrs irrelevant, just must resolve.
      return { attrs: {} } as never;
    }
    const attrs = attrsByPath[path];
    if (attrs === undefined) {
      throw new Error(`Node not found: ${path}`);
    }
    return { attrs } as never;
  });
}

describe('loadOverlayConfigs', () => {
  let infoSpy: ReturnType<typeof vi.spyOn>;
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.restoreAllMocks();
    mockOpen.mockReset();
    infoSpy = vi.spyOn(log, 'info').mockImplementation(() => {});
    warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
  });

  it('returns child overlay configs sorted by z_index (order and content)', async () => {
    // Listed out of z_index order on purpose to prove sorting.
    const store = makeStoreWithContents([
      'overlays/title/.zattrs',
      'overlays/footer/.zattrs',
      'overlays/badge/.zattrs',
    ]);
    wireOpen({
      'overlays/title': {
        type: 'overlay_text',
        position: [0.5, 0.1],
        opacity: 0.8,
        anchor: 'top-center',
        z_index: 2,
        text: 'Hello',
        font_size: 24,
      },
      'overlays/footer': {
        type: 'overlay_html',
        position: [0, 1],
        z_index: 0,
        html: '<b>foot</b>',
      },
      'overlays/badge': {
        type: 'overlay_image',
        z_index: 1,
        image_file: 'badge.png',
        size: [0.1, 0.1],
      },
    });

    const result = await loadOverlayConfigs(store, makeRootLocation());

    // Sorted ascending by z_index: footer(0), badge(1), title(2).
    expect(result.map((c) => c.name)).toEqual(['footer', 'badge', 'title']);
    expect(result.map((c) => c.z_index)).toEqual([0, 1, 2]);

    // Content + defaults applied correctly.
    const footer = result[0];
    expect(footer.type).toBe('overlay_html');
    expect(footer.html).toBe('<b>foot</b>');
    expect(footer.position).toEqual([0, 1]);
    expect(footer.opacity).toBe(1.0); // default
    expect(footer.anchor).toBe('top-left'); // default
    expect(footer.transition).toBe('none'); // default
    expect(footer.transition_duration).toBe(0.3); // default
    expect(footer.interactive).toBe(false); // default
    expect(footer.hover).toBe(false); // default

    const title = result[2];
    expect(title.type).toBe('overlay_text');
    expect(title.text).toBe('Hello');
    expect(title.font_size).toBe(24);
    expect(title.position).toEqual([0.5, 0.1]);
    expect(title.opacity).toBe(0.8);
    expect(title.anchor).toBe('top-center');

    // Logged the count.
    expect(infoSpy).toHaveBeenCalled();
  });

  it('carries every video attribute through (the manager renders nothing without video_file)', async () => {
    const store = makeStoreWithContents(['overlays/turntable/.zattrs']);
    wireOpen({
      'overlays/turntable': {
        type: 'overlay_video',
        position: [0.06, 0.5],
        anchor: 'center-left',
        z_index: 0,
        video_file: 'video.webm',
        poster_file: 'poster.png',
        size: [0.26, null],
        loop: true,
        autoplay: true,
        muted: true,
        playback_rate: 1.5,
        visible_range: { story: 3 },
      },
    });

    const [video] = await loadOverlayConfigs(store, makeRootLocation());

    expect(video.type).toBe('overlay_video');
    expect(video.video_file).toBe('video.webm');
    expect(video.poster_file).toBe('poster.png');
    expect(video.size).toEqual([0.26, null]); // null height = keep the clip's aspect
    expect(video.loop).toBe(true);
    expect(video.autoplay).toBe(true);
    expect(video.muted).toBe(true);
    expect(video.playback_rate).toBe(1.5);
    expect(video.visible_range).toEqual({ story: 3 });
  });

  it('returns [] when there is no overlays group', async () => {
    const store = makeStoreWithContents([]);
    // Opening the overlays group itself rejects → "no overlays" path.
    mockOpen.mockRejectedValue(new Error('Node not found: overlays'));

    const result = await loadOverlayConfigs(store, makeRootLocation());

    expect(result).toEqual([]);
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('skips a child group whose type does not start with "overlay_"', async () => {
    const store = makeStoreWithContents([
      'overlays/real/.zattrs',
      'overlays/bogus/.zattrs',
      'overlays/notype/.zattrs',
    ]);
    wireOpen({
      'overlays/real': { type: 'overlay_text', z_index: 0, text: 'keep' },
      // Wrong prefix — must be skipped.
      'overlays/bogus': { type: 'points', z_index: 5 },
      // Missing type entirely — must be skipped.
      'overlays/notype': { z_index: 3 },
    });

    const result = await loadOverlayConfigs(store, makeRootLocation());

    expect(result.map((c) => c.name)).toEqual(['real']);
    expect(result[0].text).toBe('keep');
  });

  it('returns [] and warns when the store cannot be enumerated', async () => {
    const store = makeStoreNoListing();
    // The overlays group opens fine; enumeration is what fails (no contents/list).
    mockOpen.mockImplementation(async (loc: unknown) => {
      const path = (loc as { path?: string } | undefined)?.path ?? '';
      if (path === 'overlays') return { attrs: {} } as never;
      throw new Error(`unexpected open: ${path}`);
    });

    const result = await loadOverlayConfigs(store, makeRootLocation());

    expect(result).toEqual([]);
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][1]).toContain('Cannot enumerate overlay children');
  });

  it('warns and continues when zarr.open rejects for one child', async () => {
    const store = makeStoreWithContents([
      'overlays/good/.zattrs',
      'overlays/broken/.zattrs',
      'overlays/alsogood/.zattrs',
    ]);
    wireOpen(
      {
        'overlays/good': { type: 'overlay_text', z_index: 0, text: 'g1' },
        'overlays/alsogood': { type: 'overlay_text', z_index: 1, text: 'g2' },
      },
      { rejectPaths: ['overlays/broken'] }
    );

    const result = await loadOverlayConfigs(store, makeRootLocation());

    // The two good overlays survive; the broken one is dropped.
    expect(result.map((c) => c.name)).toEqual(['good', 'alsogood']);
    // The per-child failure was warned about, naming the broken overlay.
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][1]).toContain('broken');
  });

  it('drops an oversized html value and warns, naming the overlay and sizes (issue #768)', async () => {
    const oversized = 'a'.repeat(MAX_OVERLAY_HTML_CHARS + 1);
    const store = makeStoreWithContents(['overlays/huge/.zattrs']);
    wireOpen({
      'overlays/huge': { type: 'overlay_html', z_index: 0, html: oversized },
    });

    const result = await loadOverlayConfigs(store, makeRootLocation());

    // The overlay is still loaded, but its oversized html is stripped so
    // nothing that could hang the parser reaches the DOM.
    expect(result.map((c) => c.name)).toEqual(['huge']);
    expect(result[0].html).toBeUndefined();
    // Exactly one warning, on the SCENE_LOADER module, naming the overlay and both sizes.
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][0]).toBe(Modules.SCENE_LOADER);
    const msg = String(warnSpy.mock.calls[0][1]);
    expect(msg).toContain('huge');
    expect(msg).toContain(String(oversized.length));
    expect(msg).toContain(String(MAX_OVERLAY_HTML_CHARS));
  });

  it('drops an oversized text value and warns, naming the overlay and sizes (issue #768)', async () => {
    // `text` is consumed as an overlay_html hover template and reaches the DOM
    // via innerHTML, so it carries the same DoS threat as `html`.
    const oversized = 'a'.repeat(MAX_OVERLAY_HTML_CHARS + 1);
    const store = makeStoreWithContents(['overlays/wordy/.zattrs']);
    wireOpen({
      'overlays/wordy': { type: 'overlay_html', z_index: 0, text: oversized },
    });

    const result = await loadOverlayConfigs(store, makeRootLocation());

    expect(result.map((c) => c.name)).toEqual(['wordy']);
    expect(result[0].text).toBeUndefined();
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][0]).toBe(Modules.SCENE_LOADER);
    const msg = String(warnSpy.mock.calls[0][1]);
    expect(msg).toContain('wordy');
    expect(msg).toContain(String(oversized.length));
    expect(msg).toContain(String(MAX_OVERLAY_HTML_CHARS));
  });

  it('drops a non-string html value that would smuggle an oversized payload (issue #768)', async () => {
    // An array wrapping a huge string has .length 1, so a size check alone
    // passes it — but `innerHTML = value` coerces it to the full payload.
    // The loader must reject non-string html outright.
    const smuggled = ['<b>'.repeat(MAX_OVERLAY_HTML_CHARS)];
    const store = makeStoreWithContents(['overlays/sneaky/.zattrs']);
    wireOpen({
      'overlays/sneaky': { type: 'overlay_html', z_index: 0, html: smuggled },
    });

    const result = await loadOverlayConfigs(store, makeRootLocation());

    expect(result.map((c) => c.name)).toEqual(['sneaky']);
    expect(result[0].html).toBeUndefined();
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][0]).toBe(Modules.SCENE_LOADER);
    expect(String(warnSpy.mock.calls[0][1])).toContain('sneaky');
    expect(String(warnSpy.mock.calls[0][1])).toContain('not a string');
  });

  it('drops a non-string text value (issue #768)', async () => {
    const smuggled = ['x'.repeat(MAX_OVERLAY_HTML_CHARS * 2)];
    const store = makeStoreWithContents(['overlays/sneaky/.zattrs']);
    wireOpen({
      'overlays/sneaky': { type: 'overlay_html', z_index: 0, text: smuggled },
    });

    const result = await loadOverlayConfigs(store, makeRootLocation());

    expect(result.map((c) => c.name)).toEqual(['sneaky']);
    expect(result[0].text).toBeUndefined();
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(String(warnSpy.mock.calls[0][1])).toContain('not a string');
  });

  it('preserves oversized text on a plain overlay_text overlay (issue #768)', async () => {
    // A plain text overlay renders via textContent — linear cost, never
    // parsed as HTML — so its content must NOT be dropped by the cap.
    const big = 'x'.repeat(MAX_OVERLAY_HTML_CHARS + 1);
    const store = makeStoreWithContents(['overlays/prose/.zattrs']);
    wireOpen({
      'overlays/prose': { type: 'overlay_text', z_index: 0, text: big },
    });

    const result = await loadOverlayConfigs(store, makeRootLocation());

    expect(result[0].text).toBe(big);
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('preserves an html value at the cap length (issue #768)', async () => {
    const atCap = 'a'.repeat(MAX_OVERLAY_HTML_CHARS);
    const store = makeStoreWithContents(['overlays/ok/.zattrs']);
    wireOpen({
      'overlays/ok': { type: 'overlay_html', z_index: 0, html: atCap },
    });

    const result = await loadOverlayConfigs(store, makeRootLocation());

    expect(result[0].html).toBe(atCap);
    expect(warnSpy).not.toHaveBeenCalled();
  });
});
