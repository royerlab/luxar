/**
 * The on-disk format-version policy, pinned as ONE case table.
 *
 * The same table — same ids, same inputs, same outcomes — lives in the Python
 * suite at `luxar/typing_utils/tests/test_format_version.py`. Keep the two in
 * lockstep: the point of a shared policy is that a store the Python reader
 * warns about is a store the viewer warns about, and a store one refuses the
 * other refuses.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const notifierMocks = vi.hoisted(() => ({ toast: vi.fn() }));
vi.mock('../../../utils/cross-layer/notifier', () => ({
  notifier: { toast: notifierMocks.toast },
}));

import {
  UnsupportedFormatVersionError,
  checkFormatVersion,
  enforceFormatVersion,
  parseFormatVersion,
  readSceneFormatVersion,
  unsupportedFormatVersionFrom,
  type FormatKind,
  type FormatVersionOutcome,
} from '../../../data/format-version';
import {
  FORMAT_TYPE_GSPLATS,
  FORMAT_TYPE_SCENE,
  GSPLATS_FORMAT_VERSION,
  LEGACY_SCENE_VERSION_ATTR,
  SCENE_FORMAT_VERSION,
  SUPPORTED_GSPLATS_FORMAT_VERSIONS,
  SUPPORTED_SCENE_VERSIONS,
} from '../../../types/format-contract';

const SCENE = ['scene', SCENE_FORMAT_VERSION, SUPPORTED_SCENE_VERSIONS] as const;
const GSPLATS = ['gsplats', GSPLATS_FORMAT_VERSION, SUPPORTED_GSPLATS_FORMAT_VERSIONS] as const;

type Kind = readonly [FormatKind, string, readonly string[]];

/** (id, kind, on-disk version, expected outcome) — mirrored by the Python test. */
const CASES: Array<[string, Kind, unknown, FormatVersionOutcome]> = [
  ['supported-scene-0.1', SCENE, '0.1', 'supported'],
  ['supported-scene-0.2', SCENE, '0.2', 'supported'],
  ['supported-gsplats-3.0', GSPLATS, '3.0', 'supported'],
  ['supported-gsplats-3.4', GSPLATS, '3.4', 'supported'],
  ['newer-minor-scene', SCENE, '0.3', 'newer-minor'],
  ['newer-minor-gsplats', GSPLATS, '3.5', 'newer-minor'],
  ['older-unsupported-scene', SCENE, '0.0', 'refuse'],
  ['older-unsupported-gsplats', GSPLATS, '2.0', 'refuse'],
  ['newer-major-scene', SCENE, '9.9', 'refuse'],
  ['newer-major-gsplats', GSPLATS, '9.9', 'refuse'],
  ['unparsable', SCENE, 'abc', 'refuse'],
  ['unparsable-none', SCENE, null, 'refuse'],
  ['unparsable-patch', SCENE, '0.2.1', 'refuse'],
];

describe('checkFormatVersion — shared case table', () => {
  it.each(CASES)('%s', (_id, kind, version, expected) => {
    const [name, current, supported] = kind;
    const { outcome, message } = checkFormatVersion(name, version, current, supported);
    expect(outcome).toBe(expected);
    if (expected === 'supported') {
      expect(message).toBe('');
    } else {
      expect(message).toContain(String(version));
      expect(message).not.toBe('');
    }
  });

  it('gsplats refuse message keeps the migrate hint', () => {
    const { message } = checkFormatVersion('gsplats', '2.0', GSPLATS[1], GSPLATS[2]);
    expect(message).toContain('luxar gsplat migrate-format');
    expect(message).toContain("Unsupported format_version: '2.0'");
  });

  it('scene refuse messages name the remedy', () => {
    expect(checkFormatVersion('scene', '0.0', SCENE[1], SCENE[2]).message).toContain(
      'Rebuild the scene'
    );
    expect(checkFormatVersion('scene', '9.9', SCENE[1], SCENE[2]).message).toContain(
      'Upgrade the viewer'
    );
  });

  it('newer-minor message says loading anyway and names the build version', () => {
    const { message } = checkFormatVersion('scene', '0.3', SCENE[1], SCENE[2]);
    expect(message).toContain('Loading anyway');
    expect(message).toContain(SCENE_FORMAT_VERSION);
  });
});

describe('parseFormatVersion', () => {
  it.each<[unknown, [number, number] | null]>([
    ['0.2', [0, 2]],
    ['3.4', [3, 4]],
    ['10.12', [10, 12]],
    ['abc', null],
    ['0.2.1', null],
    ['v0.2', null],
    [' 0.2', null],
    [0.2, null],
    [null, null],
    [undefined, null],
  ])('%j → %j', (raw, expected) => {
    expect(parseFormatVersion(raw)).toEqual(expected);
  });
});

describe('readSceneFormatVersion', () => {
  it('prefers format_version over the legacy key', () => {
    expect(readSceneFormatVersion({ format_version: '0.2' })).toBe('0.2');
    expect(readSceneFormatVersion({ format_version: '0.2', luxar_version: '0.1' })).toBe('0.2');
  });

  it('legacy-key-fallback: a 0.1 root with only luxar_version reads 0.1', () => {
    expect(readSceneFormatVersion({ [LEGACY_SCENE_VERSION_ATTR]: '0.1' })).toBe('0.1');
  });

  it('returns null when neither key is present', () => {
    expect(readSceneFormatVersion({ type: 'scene' })).toBeNull();
  });
});

describe('enforceFormatVersion — scene root attrs', () => {
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    notifierMocks.toast.mockClear();
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('supported: loads silently', () => {
    expect(
      enforceFormatVersion({ type: 'scene', format_version: '0.2', format_type: FORMAT_TYPE_SCENE })
    ).toBe('supported');
    expect(notifierMocks.toast).not.toHaveBeenCalled();
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('legacy-key-fallback: a 0.1 root loads silently', () => {
    expect(enforceFormatVersion({ type: 'scene', [LEGACY_SCENE_VERSION_ATTR]: '0.1' })).toBe(
      'supported'
    );
    expect(notifierMocks.toast).not.toHaveBeenCalled();
  });

  it('missing-without-format-type: tolerated, returns null', () => {
    expect(enforceFormatVersion({ type: 'scene' })).toBeNull();
    expect(enforceFormatVersion(undefined)).toBeNull();
    expect(notifierMocks.toast).not.toHaveBeenCalled();
  });

  it('missing-with-format-type: a stripped 0.2 header is refused', () => {
    expect(() => enforceFormatVersion({ type: 'scene', format_type: FORMAT_TYPE_SCENE })).toThrow(
      /format_type/
    );
  });

  it('newer-minor: warns (console + toast) and returns the outcome', () => {
    expect(enforceFormatVersion({ type: 'scene', format_version: '0.3' })).toBe('newer-minor');
    expect(notifierMocks.toast).toHaveBeenCalledTimes(1);
    expect(String(notifierMocks.toast.mock.calls[0][0])).toContain('0.3');
    expect(warnSpy).toHaveBeenCalledTimes(1);
  });

  it.each(['0.0', '9.9', 'abc'])('refuse: %s throws naming the version', (version) => {
    expect(() => enforceFormatVersion({ type: 'scene', format_version: version })).toThrow(version);
    expect(() => enforceFormatVersion({ type: 'scene', format_version: version })).toThrow(
      UnsupportedFormatVersionError
    );
    expect(notifierMocks.toast).not.toHaveBeenCalled();
  });

  it('unsupportedFormatVersionFrom finds the refusal through a cause chain', () => {
    let thrown: unknown;
    try {
      enforceFormatVersion({ type: 'scene', format_version: '9.9' });
    } catch (e) {
      thrown = e;
    }
    const wrapped = new Error('init failed', {
      cause: new Error('load failed', { cause: thrown }),
    });
    expect(unsupportedFormatVersionFrom(wrapped)?.message).toContain('9.9');
    expect(unsupportedFormatVersionFrom(new Error('plain'))).toBeUndefined();
    expect(unsupportedFormatVersionFrom('not an error')).toBeUndefined();
  });

  it('coerces a numeric format_version before matching', () => {
    expect(enforceFormatVersion({ type: 'scene', format_version: 0.2 })).toBe('supported');
  });
});

describe('enforceFormatVersion — detached gsplats root attrs', () => {
  beforeEach(() => {
    notifierMocks.toast.mockClear();
    vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('dispatches on format_type to the gsplats allowlist', () => {
    expect(enforceFormatVersion({ format_type: FORMAT_TYPE_GSPLATS, format_version: '3.4' })).toBe(
      'supported'
    );
    // 3.4 is NOT a scene version; without the dispatch this would refuse.
    expect(SUPPORTED_SCENE_VERSIONS).not.toContain('3.4');
  });

  it('newer-minor gsplats (3.5) warns and loads', () => {
    expect(enforceFormatVersion({ format_type: FORMAT_TYPE_GSPLATS, format_version: '3.5' })).toBe(
      'newer-minor'
    );
    expect(notifierMocks.toast).toHaveBeenCalledTimes(1);
  });

  it('older gsplats (2.0) is refused with the migrate hint', () => {
    expect(() =>
      enforceFormatVersion({ format_type: FORMAT_TYPE_GSPLATS, format_version: '2.0' })
    ).toThrow(/migrate-format/);
  });

  it('a gsplats root without format_version is refused', () => {
    expect(() => enforceFormatVersion({ format_type: FORMAT_TYPE_GSPLATS })).toThrow(
      /format_version/
    );
  });
});
