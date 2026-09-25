/**
 * Materialise a viewer build for any git ref, cached by commit SHA.
 *
 * A ref is extracted with `git archive` (no worktree is registered, nothing
 * touches the index) into `<cacheRoot>/<sha>/packages/luxar-viewer`, its
 * dependencies are installed from the lockfile, and `vite build` writes
 * `dist/`. The WASM module is the slow part of a from-scratch build, so when
 * the ref's Rust sources are byte-identical to the current checkout's, the
 * current checkout's `public/wasm` is copied instead of recompiled.
 *
 * The special ref `WORKTREE` builds the current checkout's working tree in
 * place, into a separate output directory, for iterating on uncommitted work.
 *
 * @module scripts/render-gate/builds
 */

import { spawnSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const VIEWER = 'packages/luxar-viewer';
const RUST = `${VIEWER}/src/wasm/rust`;
const STAMP = '.render-gate-build-ok';

function run(cmd, args, cwd, { quiet = false } = {}) {
  const res = spawnSync(cmd, args, {
    cwd,
    encoding: 'utf8',
    stdio: quiet ? ['ignore', 'pipe', 'pipe'] : ['ignore', 'inherit', 'inherit'],
    maxBuffer: 1 << 28,
  });
  if (res.status !== 0) {
    const detail = quiet ? `\n${res.stdout ?? ''}${res.stderr ?? ''}` : '';
    throw new Error(`${cmd} ${args.join(' ')} failed in ${cwd} (exit ${res.status})${detail}`);
  }
  return res.stdout ?? '';
}

/**
 * @param {string} repoRoot Root of the current checkout.
 * @param {string} ref Any git ref.
 * @returns {string} The full commit SHA.
 */
export function resolveSha(repoRoot, ref) {
  return run('git', ['rev-parse', '--verify', `${ref}^{commit}`], repoRoot, { quiet: true }).trim();
}

function rustUnchanged(repoRoot, sha) {
  const res = spawnSync('git', ['diff', '--quiet', sha, '--', RUST], { cwd: repoRoot });
  return res.status === 0;
}

function buildViewer(viewerDir, outDir, repoRoot, sha) {
  run('pnpm', ['install', '--frozen-lockfile', '--prefer-offline'], viewerDir, { quiet: true });
  const currentWasm = join(repoRoot, VIEWER, 'public', 'wasm');
  if (sha && existsSync(currentWasm) && rustUnchanged(repoRoot, sha)) {
    cpSync(currentWasm, join(viewerDir, 'public', 'wasm'), { recursive: true });
  } else if (!existsSync(join(viewerDir, 'public', 'wasm', 'luxar_wasm_bg.wasm'))) {
    run('bash', ['scripts/build-wasm.sh'], viewerDir);
  }
  run('pnpm', ['exec', 'vite', 'build', '--outDir', outDir, '--emptyOutDir'], viewerDir, {
    quiet: true,
  });
}

/**
 * Return a `dist/` directory for `ref`, building it on first use.
 *
 * @param {{ repoRoot: string, cacheRoot: string, ref: string, log?: (m: string) => void }} opts
 * @returns {{ ref: string, sha: string, distDir: string }}
 */
export function ensureBuild({ repoRoot, cacheRoot, ref, log = () => {} }) {
  if (ref === 'WORKTREE') {
    const viewerDir = join(repoRoot, VIEWER);
    const distDir = join(cacheRoot, 'worktree-dist');
    log(`building the working tree into ${distDir}`);
    buildViewer(viewerDir, distDir, repoRoot, null);
    return { ref, sha: 'WORKTREE', distDir };
  }
  const sha = resolveSha(repoRoot, ref);
  const root = join(cacheRoot, sha);
  const viewerDir = join(root, VIEWER);
  const distDir = join(viewerDir, 'dist');
  if (existsSync(join(root, STAMP))) return { ref, sha, distDir };

  log(`building ${ref} (${sha.slice(0, 10)}) into ${root}`);
  rmSync(root, { recursive: true, force: true });
  mkdirSync(root, { recursive: true });
  const archive = spawnSync('git', ['archive', '--format=tar', sha, VIEWER], {
    cwd: repoRoot,
    maxBuffer: 1 << 30,
  });
  if (archive.status !== 0) throw new Error(`git archive ${sha} failed`);
  const untar = spawnSync('tar', ['-x', '-C', root], { input: archive.stdout });
  if (untar.status !== 0) throw new Error(`extracting ${sha} failed`);
  buildViewer(viewerDir, distDir, repoRoot, sha);
  writeFileSync(join(root, STAMP), `${sha}\n`);
  return { ref, sha, distDir };
}
