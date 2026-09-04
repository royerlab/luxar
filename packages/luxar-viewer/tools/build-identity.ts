/**
 * Build-time identity of the viewer bundle: version, commit, build timestamp.
 *
 * A bug report says "the viewer renders black". Without a stamp there is no way
 * to tie that to a revision, and the viewer reaches users through five channels
 * that are cut at different times — the dev server, the PyPI wheel
 * (`luxar/_viewer_dist`), `luxar export` offline folders, the npm library
 * bundle, and the hosted demo site. Three of those are copies of `dist/`, so
 * stamping at Vite build time reaches them all from one place.
 *
 * `version` comes from `package.json`, which `scripts/set_version.py` keeps in
 * lockstep with Python's `__version__` and `CITATION.cff` (gated by
 * `scripts/tests/test_set_version.py::test_the_committed_tree_is_consistent`).
 * That makes it the one version in the repo that cannot silently drift.
 *
 * NOTHING HERE MAY THROW. A build that dies because `git` is missing is a
 * strictly worse outcome than a bundle stamped `commit: "unknown"` — and the
 * missing-git case is the normal one for a source tarball, a Docker build with
 * no `.git`, or a shallow CI checkout of an archive. Every probe is wrapped and
 * degrades to `'unknown'`.
 */

import { execFileSync } from 'child_process';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';

/** Placeholder written whenever a probe cannot answer. Never an empty string. */
export const UNKNOWN = 'unknown';

/** The compile-time constant name that carries the stamp into the bundle. */
export const BUILD_DEFINE = '__LUXAR_BUILD__';

export interface BuildIdentity {
  /** Semver-normalized CalVer from package.json, e.g. `2026.6.5`. */
  version: string;
  /** Short commit SHA, `<sha>-dirty` when the tree has uncommitted changes. */
  commit: string;
  /** ISO-8601 UTC instant the bundle was built. */
  buildTime: string;
}

function viewerRoot(): string {
  return fileURLToPath(new URL('..', import.meta.url));
}

function packageVersion(root: string): string {
  try {
    const raw = readFileSync(new URL('package.json', `file://${root}`), 'utf8');
    const parsed: unknown = JSON.parse(raw);
    const version = (parsed as { version?: unknown }).version;
    return typeof version === 'string' && version.length > 0 ? version : UNKNOWN;
  } catch {
    return UNKNOWN;
  }
}

function gitCommit(root: string): string {
  const git = (args: string[]): string =>
    execFileSync('git', args, { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
  try {
    const sha = git(['rev-parse', '--short', 'HEAD']).trim();
    if (!sha) return UNKNOWN;
    // A dev build with uncommitted work must not be reported as that commit:
    // the SHA would send a bug report to source that is not what ran.
    const dirty = git(['status', '--porcelain']).trim().length > 0;
    return dirty ? `${sha}-dirty` : sha;
  } catch {
    return UNKNOWN;
  }
}

/**
 * Resolve the identity of the bundle being built.
 *
 * `now` is injectable so tests can pin the timestamp; production callers omit it.
 */
export function buildIdentity(now: Date = new Date()): BuildIdentity {
  const root = viewerRoot();
  return {
    version: packageVersion(root),
    commit: gitCommit(root),
    // Second precision: the millisecond field is noise in a bug report and
    // makes two builds of the same source look more different than they are.
    buildTime: `${now.toISOString().slice(0, 19)}Z`,
  };
}

/**
 * The `define:` block for a Vite config.
 *
 * One constant rather than three separate defines: a partially-applied stamp
 * (version injected, commit not) is a worse diagnostic than none, because it
 * looks authoritative. All three arrive or none do.
 *
 * Double-encoded on purpose. Vite substitutes a define's value as RAW SOURCE,
 * so a bare object literal would be spliced in unparenthesised and is a syntax
 * hazard wherever the identifier lands in statement position. Emitting a quoted
 * JSON *string* is always a well-formed expression; the runtime parses it once.
 */
export function buildDefine(identity: BuildIdentity = buildIdentity()): Record<string, string> {
  return { [BUILD_DEFINE]: JSON.stringify(JSON.stringify(identity)) };
}

/**
 * Render the stamp as an HTML `<meta>` tag.
 *
 * The runtime surface (`window.__luxarBuild`) needs the app to boot. This one
 * survives a bundle that fails to start, and is greppable in a `dist/` tree, an
 * unzipped wheel, or an emailed `luxar export` folder without running anything
 * — which is the state most bug reports arrive in.
 */
export function buildMetaTag(identity: BuildIdentity = buildIdentity()): string {
  const content = `${identity.version} ${identity.commit} ${identity.buildTime}`;
  return `<meta name="luxar-build" content="${content}" />`;
}

/** Vite plugin injecting {@link buildMetaTag} into `index.html`'s `<head>`. */
export function buildIdentityHtmlPlugin(identity: BuildIdentity = buildIdentity()) {
  return {
    name: 'luxar-build-identity',
    transformIndexHtml(html: string): string {
      return html.replace('</head>', `  ${buildMetaTag(identity)}\n  </head>`);
    },
  };
}
