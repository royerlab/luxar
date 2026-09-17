/**
 * The viewer's release version, as a plain string.
 *
 * Injected at build time from `package.json` through Vite's `define:`
 * (`viewerVersionDefine` in `tools/build-identity.ts`, applied by the
 * application, library and vitest configs), so there is no second copy of the
 * version to keep in step — `scripts/set_version.py` writes `package.json` and
 * `scripts/check_version_consistency.py` gates it against Python and
 * `CITATION.cff`.
 *
 * Distinct from `config/build-info.ts`: that is the diagnostic stamp (version +
 * commit + build time) and legitimately reports `'unknown'` when unstamped.
 * This is the public constant an embedder compares against, so it is always a
 * version-shaped string — a context with no define (a consumer bundling `src/`
 * with its own config) sees {@link DEV_VIEWER_VERSION} rather than `'unknown'`.
 */

import { UNKNOWN } from './config/build-info';

/** What {@link VIEWER_VERSION} reports when no build injected a version. */
export const DEV_VIEWER_VERSION = '0.0.0-dev';

/**
 * Map the raw define value to the exported version: any non-empty string other
 * than the build stamp's `'unknown'` placeholder passes through; everything
 * else (absent define, empty, placeholder) is the development fallback.
 * Separated from the constant so the fallback arm is testable in a build that
 * does inject the define.
 */
export function resolveViewerVersion(defined: unknown): string {
  if (typeof defined !== 'string') return DEV_VIEWER_VERSION;
  if (defined.length === 0 || defined === UNKNOWN) return DEV_VIEWER_VERSION;
  return defined;
}

/**
 * The viewer's release version — `package.json`'s `version` in every built
 * bundle (semver-normalized CalVer such as `2026.6.5`), or
 * {@link DEV_VIEWER_VERSION} when no build injected one.
 */
export const VIEWER_VERSION: string = resolveViewerVersion(
  // `typeof` on an undeclared identifier is the one legal reference before
  // the define exists; a bare read would be a ReferenceError in an embedder.
  typeof __LUXAR_VIEWER_VERSION__ === 'undefined' ? undefined : __LUXAR_VIEWER_VERSION__
);
