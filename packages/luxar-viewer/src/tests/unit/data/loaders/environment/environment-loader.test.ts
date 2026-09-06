/**
 * `loadBakedEnvironment`: the store contract a baked map rests on, from the viewer's
 * side. Absence is silent; a matching map loads as six half-bit faces; a stale or
 * malformed map is IGNORED with a warning rather than failing the scene — the
 * environment is lighting, not data.
 *
 * Only the zarr boundary is mocked (`zarrita.open` / `zarrita.get`), as the sibling
 * overlay-loader test does.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('zarrita', async () => {
  const actual = await vi.importActual<typeof import('zarrita')>('zarrita');
  return { ...actual, open: vi.fn(), get: vi.fn() };
});

import { open as zarrOpen, get as zarrGet } from 'zarrita';
import { loadBakedEnvironment } from '../../../../../data/loaders/environment/environment-loader';
import { log } from '../../../../../utils/log';

const mockOpen = vi.mocked(zarrOpen);
const mockGet = vi.mocked(zarrGet);

const RES = 4;
const HASH = 'feedface00';

function makeRootLocation(): import('zarrita').Location<import('zarrita').Readable> {
  return {
    resolve: vi.fn((name: string) => ({ path: name })),
  } as unknown as import('zarrita').Location<import('zarrita').Readable>;
}

function validAttrs(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    format: 'cube-faces-half',
    sample_format: 'half-float-bits',
    face_order: ['px', 'nx', 'py', 'ny', 'pz', 'nz'],
    coordinate_system: 'webgl',
    probe: { spec: 'auto', position: [0, 0, 0] },
    resolution: RES,
    scene_content_hash: HASH,
    faces: 'faces-0a1b2c3d',
    ...overrides,
  };
}

function wire(attrs: Record<string, unknown> | null, data?: ArrayLike<number>) {
  mockOpen.mockImplementation(async (loc: unknown) => {
    const path = (loc as { path: string }).path;
    if (path === 'environment') {
      if (!attrs) throw new Error('Node not found: environment');
      return { attrs } as never;
    }
    if (attrs && path === `environment/${String(attrs.faces)}`) return { path } as never;
    throw new Error(`Node not found: ${path}`);
  });
  mockGet.mockImplementation(
    async () =>
      ({
        data: data ?? new Uint16Array(6 * RES * RES * 4).fill(0x3c00),
        shape: [6, RES, RES, 4],
        stride: [],
      }) as never
  );
}

describe('loadBakedEnvironment', () => {
  let warn: ReturnType<typeof vi.spyOn>;
  beforeEach(() => {
    vi.restoreAllMocks();
    mockOpen.mockReset();
    mockGet.mockReset();
    warn = vi.spyOn(log, 'warning').mockImplementation(() => {});
    vi.spyOn(log, 'info').mockImplementation(() => {});
  });

  it('returns null, silently, when the store has no environment group', async () => {
    wire(null);
    expect(await loadBakedEnvironment(makeRootLocation(), HASH)).toBeNull();
    expect(warn).not.toHaveBeenCalled();
  });

  it('loads a matching map as six half-bit faces in three order', async () => {
    const data = new Uint16Array(6 * RES * RES * 4);
    for (let f = 0; f < 6; f++) data.fill(0x3c00 + f, f * RES * RES * 4, (f + 1) * RES * RES * 4);
    wire(validAttrs(), data);
    const map = await loadBakedEnvironment(makeRootLocation(), HASH);
    expect(map).not.toBeNull();
    expect(map!.resolution).toBe(RES);
    expect(map!.faces).toHaveLength(6);
    expect(map!.faces[5][0]).toBe(0x3c05);
    expect(map!.faces[0].length).toBe(RES * RES * 4);
    expect(map!.header.scene_content_hash).toBe(HASH);
    // The digest-named array was what got opened.
    expect(mockOpen).toHaveBeenCalledWith(
      { path: 'environment/faces-0a1b2c3d' },
      { kind: 'array' }
    );
    expect(warn).not.toHaveBeenCalled();
  });

  it('ignores a STALE map (scene changed since the bake) with a warning that says so', async () => {
    wire(validAttrs({ scene_content_hash: 'older' }));
    expect(await loadBakedEnvironment(makeRootLocation(), HASH)).toBeNull();
    expect(String(warn.mock.calls[0][1])).toContain('stale');
    expect(mockGet).not.toHaveBeenCalled();
  });

  it.each([
    ['format', { format: 'png' }, "is not 'cube-faces-half'"],
    ['face order', { face_order: ['nx', 'px', 'py', 'ny', 'pz', 'nz'] }, 'face_order'],
    ['resolution', { resolution: 0 }, 'resolution'],
    ['no faces attr', { faces: undefined }, 'names no faces array'],
    ['no scene hash', { scene_content_hash: '' }, 'no scene_content_hash'],
    ['sample format', { sample_format: 'float32' }, 'sample_format'],
  ])('ignores a malformed map (%s) with a warning', async (_label, overrides, needle) => {
    wire(validAttrs(overrides));
    expect(await loadBakedEnvironment(makeRootLocation(), HASH)).toBeNull();
    expect(String(warn.mock.calls[0][1])).toContain(needle);
  });

  it('ignores a map whose array has the wrong length, and a scene without a digest', async () => {
    wire(validAttrs(), new Uint16Array(10));
    expect(await loadBakedEnvironment(makeRootLocation(), HASH)).toBeNull();
    expect(String(warn.mock.calls[0][1])).toContain('samples');
    warn.mockClear();
    wire(validAttrs());
    expect(await loadBakedEnvironment(makeRootLocation(), undefined)).toBeNull();
    expect(String(warn.mock.calls[0][1])).toContain('no content_hash');
  });

  it('a failing array read degrades to no map rather than failing the load', async () => {
    wire(validAttrs());
    mockGet.mockRejectedValueOnce(new Error('network down'));
    expect(await loadBakedEnvironment(makeRootLocation(), HASH)).toBeNull();
    expect(String(warn.mock.calls[0][1])).toContain('could not read');
  });
});
