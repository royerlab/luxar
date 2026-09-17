/**
 * On-disk format-version policy — the viewer half of one shared rule.
 *
 * Mirrors `luxar/typing_utils/format_version.py` line for line and is pinned
 * by the SAME case table (same ids) in `src/tests/unit/data/format-version.test.ts`,
 * so a store the Python reader warns about is a store the viewer warns about,
 * and a store one refuses the other refuses:
 *
 * - **supported** — the version is in this build's `supported` allowlist:
 *   load silently.
 * - **newer-minor** — same MAJOR as the current writer version, higher MINOR
 *   (a `0.3` scene read by a `0.2` build; a `3.5` gsplats store read by a `3.4`
 *   build): load, but WARN (console + toast). A minor bump is additive by
 *   policy, so the viewer can still draw the store; it just cannot see what the
 *   newer writer added.
 * - **refuse** — anything else: an OLDER version that has fallen out of
 *   `supported` (`0.0`, `2.0`), a NEWER MAJOR (`9.9`), or a value that is not
 *   `MAJOR.MINOR` at all (`abc`). `enforceFormatVersion` THROWS an
 *   `UnsupportedFormatVersionError`; it propagates `SceneLoader.loadScene` →
 *   `core/app.ts` (`dataset-error`) → the error overlay, which shows the
 *   message verbatim (`core/bootstrap.ts` recognises the class the way it does
 *   `ArchiveFaultError`; the dataset-browser path already shows every message).
 *
 * A scene root may also carry NO version. Scene 0.1 wrote only the legacy
 * `luxar_version` key (`LEGACY_SCENE_VERSION_ATTR`), which readers fall back
 * to; a root that declares a `format_type` but no version is malformed and is
 * refused, while a root with neither is tolerated (external / hand-written
 * stores that predate any header).
 *
 * Shared case table (ids match the Python test):
 *
 * | id                              | input                    | outcome              |
 * |---------------------------------|--------------------------|----------------------|
 * | `supported`                     | scene 0.1 / 0.2          | supported            |
 * |                                 | gsplats 3.0 / 3.4        | supported            |
 * | `newer-minor`                   | scene 0.3, gsplats 3.5   | newer-minor          |
 * | `older-unsupported`             | scene 0.0, gsplats 2.0   | refuse               |
 * | `newer-major`                   | 9.9                      | refuse               |
 * | `unparsable`                    | `abc`                    | refuse               |
 * | `missing-with-format-type`      | `{format_type: …}`       | refuse               |
 * | `missing-without-format-type`   | `{type: 'scene'}`        | tolerated (null)     |
 * | `legacy-key-fallback`           | `{luxar_version: '0.1'}` | supported, reads 0.1 |
 */

import { log, Modules } from '../utils/log';
import { notifier } from '../utils/cross-layer/notifier';
import {
  FORMAT_TYPE_GSPLATS,
  GSPLATS_FORMAT_VERSION,
  LEGACY_SCENE_VERSION_ATTR,
  SCENE_FORMAT_VERSION,
  SUPPORTED_GSPLATS_FORMAT_VERSIONS,
  SUPPORTED_SCENE_VERSIONS,
} from '../types/format-contract';

/** Which on-disk format a version string belongs to. Decides the remedy text. */
export type FormatKind = 'scene' | 'gsplats';

/**
 * Thrown when a store's format version must be refused.
 *
 * A distinct class (not a bare `Error`) so the startup path in
 * `core/bootstrap.ts` can recognise it through the `cause` chain and put the
 * message itself — which names the version, the supported set and the remedy —
 * on the error overlay, the way it already does for `ArchiveFaultError`. Every
 * other startup failure keeps the generic "check the console" text.
 */
export class UnsupportedFormatVersionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'UnsupportedFormatVersionError';
  }
}

/**
 * Find an `UnsupportedFormatVersionError` in `error` or its `cause` chain
 * (bounded walk), or `undefined`. Mirrors `archiveFaultFrom`.
 */
export function unsupportedFormatVersionFrom(
  error: unknown
): UnsupportedFormatVersionError | undefined {
  let current = error;
  for (let depth = 0; depth < 8; depth++) {
    if (current instanceof UnsupportedFormatVersionError) return current;
    if (!(current instanceof Error)) return undefined;
    current = current.cause;
  }
  return undefined;
}

/** What a reader should do with a store of a given format version. */
export type FormatVersionOutcome = 'supported' | 'newer-minor' | 'refuse';

/** Result of `checkFormatVersion`: the outcome plus the user-facing text (empty when supported). */
export interface FormatVersionCheck {
  /** The policy arm the version landed in. */
  outcome: FormatVersionOutcome;
  /** Warning text for `newer-minor`, error text for `refuse`, `''` for `supported`. */
  message: string;
}

/**
 * Parse a `MAJOR.MINOR` string into `[major, minor]`; `null` otherwise.
 *
 * Deliberately strict: no patch component, no leading `v`, no whitespace.
 * Version strings are written by Luxar itself from the contract, so anything
 * else is a foreign or corrupt header, not a spelling to be lenient about.
 */
export function parseFormatVersion(version: unknown): [number, number] | null {
  if (typeof version !== 'string') return null;
  const m = /^(\d+)\.(\d+)$/.exec(version);
  if (!m) return null;
  return [Number(m[1]), Number(m[2])];
}

/**
 * Coerce an on-disk attr value to a version string.
 *
 * Strings pass through; numbers (a YAML-ish `0.2`) are stringified; anything
 * else — objects, booleans, `null` — is not a version and yields `null`, so
 * it lands in the unparsable arm rather than as `'[object Object]'`.
 */
function versionToString(value: unknown): string | null {
  if (typeof value === 'string') return value;
  if (typeof value === 'number') return String(value);
  return null;
}

type RefuseArm = 'older' | 'newer-major' | 'unparsable';

function remedy(kind: FormatKind, arm: RefuseArm): string {
  if (kind === 'gsplats' && arm === 'older') {
    return 'Convert it with `luxar gsplat migrate-format <input> <output.gsplats.zarr>`.';
  }
  if (kind === 'scene' && arm === 'older') {
    return 'Rebuild the scene with the current Luxar release.';
  }
  return 'Upgrade the viewer to a build that supports this format.';
}

function refuse(
  kind: FormatKind,
  label: string,
  shown: string,
  reason: string,
  arm: RefuseArm
): FormatVersionCheck {
  return {
    outcome: 'refuse',
    message:
      `Unsupported format_version: ${shown} for a ${label} store (${reason}). ` + remedy(kind, arm),
  };
}

/** Order `parsed` against `cur`: which refuse arm, or `'newer-minor'`. */
function classify(
  parsed: [number, number],
  cur: [number, number]
): 'newer-minor' | 'older' | 'newer-major' {
  const [major, minor] = parsed;
  if (major === cur[0] && minor > cur[1]) return 'newer-minor';
  if (major < cur[0] || (major === cur[0] && minor < cur[1])) return 'older';
  return 'newer-major';
}

/** Everything a message needs, resolved once so the arm builders stay flat. */
interface VersionContext {
  kind: FormatKind;
  label: string;
  /** How the on-disk value is quoted in messages (`'0.0'`, `null`). */
  shown: string;
  /** The value as a plain string, or `''` when it is not one. */
  asString: string;
  current: string;
  supportedText: string;
}

function parsedArmResult(
  ctx: VersionContext,
  arm: 'newer-minor' | 'older' | 'newer-major'
): FormatVersionCheck {
  if (arm === 'newer-minor') {
    return {
      outcome: 'newer-minor',
      message:
        `This ${ctx.label} store is format ${ctx.asString}, newer than the ${ctx.current} ` +
        'this viewer was built for. Loading anyway; content added by the newer ' +
        'format is not visible. Upgrade the viewer to read it fully.',
    };
  }
  if (arm === 'older') {
    const reason = `too old; supported: ${ctx.supportedText}`;
    return refuse(ctx.kind, ctx.label, ctx.shown, reason, 'older');
  }
  const reason = `newer major version; this viewer reads ${ctx.supportedText}`;
  return refuse(ctx.kind, ctx.label, ctx.shown, reason, 'newer-major');
}

/**
 * Classify `version` against this build's `current` / `supported` versions.
 *
 * @param kind `'scene'` or `'gsplats'` — picks the noun and the remedy.
 * @param version The on-disk value (any type; non-strings are unparsable).
 * @param current The version the paired Python writer emits.
 * @param supported The versions this build has read end-to-end.
 */
export function checkFormatVersion(
  kind: FormatKind,
  version: unknown,
  current: string,
  supported: readonly string[]
): FormatVersionCheck {
  const label = kind === 'scene' ? 'scene' : '.gsplats.zarr';
  if (typeof version === 'string' && supported.includes(version)) {
    return { outcome: 'supported', message: '' };
  }

  const asString = versionToString(version);
  const ctx: VersionContext = {
    kind,
    label,
    shown: typeof version === 'string' ? `'${version}'` : (asString ?? String(version)),
    asString: asString ?? '',
    current,
    supportedText: supported.join(', '),
  };
  const parsed = parseFormatVersion(version);
  if (parsed === null) {
    const reason = `not a MAJOR.MINOR version; supported: ${ctx.supportedText}`;
    return refuse(kind, label, ctx.shown, reason, 'unparsable');
  }

  const cur = parseFormatVersion(current);
  if (cur === null) {
    throw new Error(`current version ${current} is not MAJOR.MINOR`);
  }
  return parsedArmResult(ctx, classify(parsed, cur));
}

/**
 * Return a scene root's version string, or `null` when it carries none.
 *
 * `format_version` (scene 0.2+) wins; the 0.1 legacy key
 * `LEGACY_SCENE_VERSION_ATTR` is the fallback. Numbers are coerced to string so
 * a numeric `0.2` still compares against the allowlist.
 */
export function readSceneFormatVersion(attrs: Readonly<Record<string, unknown>>): string | null {
  const value = attrs.format_version ?? attrs[LEGACY_SCENE_VERSION_ATTR];
  return versionToString(value);
}

/** The version to check for a root of the given kind — gsplats has no legacy key. */
function readVersionFor(attrs: Readonly<Record<string, unknown>>, gsplats: boolean): string | null {
  return gsplats ? versionToString(attrs.format_version) : readSceneFormatVersion(attrs);
}

/**
 * Apply the policy to a store's root attrs; throw, warn or return.
 *
 * Dispatches on `format_type`: `FORMAT_TYPE_GSPLATS` selects the gsplats
 * current/supported set (a detached `.gsplats.zarr` loads directly in the
 * viewer), anything else is treated as a scene root.
 *
 * - `refuse` → `throw new UnsupportedFormatVersionError(message)` (surfaces,
 *   verbatim, on the error overlay).
 * - `newer-minor` → `log.warning` + `notifier.toast`, returns the outcome.
 * - `supported` → returns the outcome.
 * - No version but a `format_type` → refuse (a stripped 0.2 header is corrupt,
 *   not legacy).
 * - No version and no `format_type` → `null` (tolerated pre-header store).
 */
export function enforceFormatVersion(
  attrs: Readonly<Record<string, unknown>> | null | undefined
): FormatVersionOutcome | null {
  const a = attrs ?? {};
  const isGsplats = a.format_type === FORMAT_TYPE_GSPLATS;
  const version = readVersionFor(a, isGsplats);

  if (version === null) {
    if (a.format_type !== undefined && a.format_type !== null) {
      throw new UnsupportedFormatVersionError(
        `Store root declares format_type=${JSON.stringify(a.format_type)} but no ` +
          'format_version. A Luxar header carries both; this store is corrupt or was ' +
          'written by a foreign tool. Rebuild it with the current Luxar release.'
      );
    }
    return null;
  }

  const { outcome, message } = isGsplats
    ? checkFormatVersion(
        'gsplats',
        version,
        GSPLATS_FORMAT_VERSION,
        SUPPORTED_GSPLATS_FORMAT_VERSIONS
      )
    : checkFormatVersion('scene', version, SCENE_FORMAT_VERSION, SUPPORTED_SCENE_VERSIONS);

  if (outcome === 'refuse') {
    throw new UnsupportedFormatVersionError(message);
  }
  if (outcome === 'newer-minor') {
    log.warning(Modules.SCENE_LOADER, message);
    notifier.toast(message, 6000);
  }
  return outcome;
}
