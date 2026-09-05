/**
 * Runtime view of the bundle's build identity.
 *
 * The stamp is injected by `tools/build-identity.ts` through Vite's `define:`
 * in both `vite.config.ts` (the application bundle that becomes `dist/`, and
 * from there the PyPI wheel, `luxar export` folders and the hosted site) and
 * `vite.lib.config.ts` (the npm library bundle).
 *
 * Read it through {@link buildInfo}, never off the global directly: the define
 * is absent in every context that does not go through those two configs —
 * vitest, a consumer bundling `src/` themselves, ts-node tooling — and a bare
 * reference to an undeclared identifier is a `ReferenceError`, not `undefined`.
 */

/**
 * Placeholder for a field the build could not resolve.
 *
 * Deliberately duplicated from `tools/build-identity.ts` rather than imported:
 * that module pulls `child_process` and `fs`, which must never reach a browser
 * bundle. `build-info.test.ts` imports both and asserts they agree, so the
 * duplication cannot drift silently.
 */
export const UNKNOWN = 'unknown';

export interface BuildInfo {
  /** Semver-normalized CalVer, e.g. `2026.6.5`, or `'unknown'`. */
  version: string;
  /** Short commit SHA, `<sha>-dirty`, or `'unknown'`. */
  commit: string;
  /** ISO-8601 UTC build instant, or `'unknown'`. */
  buildTime: string;
  /**
   * False when no stamp was injected — a dev/test/embedder context rather than
   * a build product. Callers reporting a version to a user should say so
   * instead of presenting `'unknown'` as if the build were broken.
   */
  stamped: boolean;
}

const UNSTAMPED: BuildInfo = {
  version: UNKNOWN,
  commit: UNKNOWN,
  buildTime: UNKNOWN,
  stamped: false,
};

function field(source: Record<string, unknown>, key: string): string {
  const value = source[key];
  return typeof value === 'string' && value.length > 0 ? value : UNKNOWN;
}

function read(): BuildInfo {
  // `typeof` on an undeclared identifier is the one reference that is legal
  // before the define exists; anything else throws in vitest and in embedders.
  if (typeof __LUXAR_BUILD__ === 'undefined') return UNSTAMPED;
  try {
    const parsed: unknown = JSON.parse(__LUXAR_BUILD__);
    if (typeof parsed !== 'object' || parsed === null) return UNSTAMPED;
    const source = parsed as Record<string, unknown>;
    return {
      version: field(source, 'version'),
      commit: field(source, 'commit'),
      buildTime: field(source, 'buildTime'),
      stamped: true,
    };
  } catch {
    // A malformed stamp is a build-tooling bug, but losing the whole viewer to
    // it would be absurd — a diagnostic aid must never be load-bearing.
    return UNSTAMPED;
  }
}

const cached: BuildInfo = read();

/** The identity of this bundle. Frozen at module load; never throws. */
export function buildInfo(): BuildInfo {
  return cached;
}

/** One-line human form, e.g. `2026.6.5 (77ced2609, built 2026-09-04T16:10:06Z)`. */
export function buildInfoLine(info: BuildInfo = cached): string {
  if (!info.stamped) return 'development build (unstamped)';
  return `${info.version} (${info.commit}, built ${info.buildTime})`;
}
