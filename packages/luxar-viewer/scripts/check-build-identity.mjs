#!/usr/bin/env node
/**
 * Assert a built bundle actually carries its build stamp.
 *
 * `define:` is silent when it goes wrong. Drop the `define` block, rename the
 * constant, add a third Vite config that forgets it — every unit test still
 * passes, the build still succeeds, and the artifact ships with no revision on
 * it. The whole point of the stamp is the bug report six months from now, so
 * the only check with teeth is one that reads the emitted files.
 *
 * Both sides are derived from the tree: the expected version is read from
 * `package.json` (the file `scripts/set_version.py` stamps), and the observed
 * one from the artifact. Neither is written down here.
 *
 *   node scripts/check-build-identity.mjs dist        # application build
 *   node scripts/check-build-identity.mjs dist/lib    # npm library build
 *
 * Exits 1 with a specific message on the first failure, 0 when every applicable
 * surface carries a matching stamp.
 */

import { existsSync, readdirSync, readFileSync, statSync } from 'fs';
import { dirname, join, resolve } from 'path';
import { fileURLToPath } from 'url';

const VIEWER_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/** Marker the runtime stamp is stored under, mirroring `tools/build-identity.ts`. */
const STAMP_KEY = 'buildTime';
const META_NAME = 'luxar-build';
const UNKNOWN = 'unknown';

/** The version every surface must agree with. */
export function expectedVersion(root = VIEWER_ROOT) {
  const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
  if (typeof pkg.version !== 'string' || pkg.version.length === 0) {
    throw new Error('package.json has no version to check against');
  }
  return pkg.version;
}

/** Every `.js` file directly inside `dir` (the bundle output is flat per level). */
export function bundleFiles(dir) {
  const out = [];
  const walk = (d) => {
    for (const name of readdirSync(d)) {
      const full = join(d, name);
      if (statSync(full).isDirectory()) walk(full);
      else if (name.endsWith('.js')) out.push(full);
    }
  };
  walk(dir);
  return out;
}

/**
 * Find the stamp inside emitted JS.
 *
 * Escaping-insensitive on purpose: the app build emits the JSON stamp as a
 * plain string literal while the library build backslash-escapes its quotes.
 * A grep written against one of those spellings reports a confident FALSE
 * NEGATIVE against the other — which is exactly what happened while this stamp
 * was being added, and is the failure this normalisation exists to prevent.
 */
export function findStamp(source) {
  const flat = source.replace(/\\"/g, '"');
  const match = flat.match(/\{"version":"([^"]*)","commit":"([^"]*)","buildTime":"([^"]*)"\}/);
  return match ? { version: match[1], commit: match[2], buildTime: match[3] } : null;
}

/** Parse the `<meta name="luxar-build">` content, or null when absent. */
export function findMetaStamp(html) {
  const match = html.match(new RegExp(`<meta name="${META_NAME}" content="([^"]*)"`));
  if (!match) return null;
  const [version, commit, buildTime] = match[1].split(' ');
  return { version, commit, buildTime };
}

/**
 * Whether an unresolved commit is a defect here.
 *
 * A source tarball legitimately has no history, so `unknown` is the honest
 * answer there. But when a `.git` directory IS present and the stamp still says
 * `unknown`, the probe broke — and that is precisely the silent case worth
 * failing on.
 */
export function commitMustResolve(root = VIEWER_ROOT) {
  return existsSync(join(root, '..', '..', '.git'));
}

export function check(
  distDir,
  { root = VIEWER_ROOT, requireCommit = commitMustResolve(root) } = {}
) {
  const problems = [];
  const version = expectedVersion(root);

  if (!existsSync(distDir)) {
    return [`${distDir} does not exist — run the build first`];
  }

  const files = bundleFiles(distDir);
  // Fail closed: an empty scan is the "gate proves nothing" case, not a pass.
  if (files.length === 0) {
    return [`${distDir} contains no .js files — nothing was scanned, so nothing was proved`];
  }

  const stamped = files.map((f) => findStamp(readFileSync(f, 'utf8'))).filter(Boolean);
  if (stamped.length === 0) {
    problems.push(
      `no build stamp in any of the ${files.length} .js file(s) under ${distDir} — ` +
        `is \`define: buildDefine()\` still in the Vite config that produced it?`
    );
  }
  for (const stamp of stamped) {
    if (stamp.version !== version) {
      problems.push(`bundle stamp says version ${stamp.version}, package.json says ${version}`);
    }
    if (requireCommit && stamp.commit === UNKNOWN) {
      problems.push('bundle stamp has no commit, but this tree has git history');
    }
    if (!stamp[STAMP_KEY]) {
      problems.push('bundle stamp has no build time');
    }
  }

  // Only the application build emits an index.html; the library build has none.
  const indexHtml = join(distDir, 'index.html');
  if (existsSync(indexHtml)) {
    const meta = findMetaStamp(readFileSync(indexHtml, 'utf8'));
    if (!meta) {
      problems.push(
        `index.html carries no <meta name="${META_NAME}"> — it is the only surface that ` +
          `survives a bundle that fails to boot`
      );
    } else if (meta.version !== version) {
      problems.push(`index.html stamp says version ${meta.version}, package.json says ${version}`);
    } else if (requireCommit && meta.commit === UNKNOWN) {
      problems.push('index.html stamp has no commit, but this tree has git history');
    }
  }

  return problems;
}

function main() {
  const distDir = resolve(process.argv[2] ?? join(VIEWER_ROOT, 'dist'));
  const problems = check(distDir);
  if (problems.length > 0) {
    for (const p of problems) console.error(`❌ ${p}`);
    process.exit(1);
  }
  console.log(`✅ ${distDir} carries a build stamp matching package.json`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
