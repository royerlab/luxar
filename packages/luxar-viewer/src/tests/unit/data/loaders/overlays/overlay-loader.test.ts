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
import { loadOverlayConfigs } from '../../../../../data/loaders/overlays/overlay-loader';
import { log } from '../../../../../utils/log';

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
});
