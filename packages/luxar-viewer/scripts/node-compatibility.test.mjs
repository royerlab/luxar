import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const PKG = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const REPO = resolve(PKG, '../..');

function packageJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

describe('published Node compatibility', () => {
  it("matches Vite's supported Node range", () => {
    const manifest = packageJson(resolve(PKG, 'package.json'));
    const vite = packageJson(resolve(PKG, 'node_modules/vite/package.json'));
    expect(manifest.engines?.node).toBe(vite.engines?.node);
  });

  it('smoke-imports the Node 22 build on the lowest supported Node', () => {
    const manifest = packageJson(resolve(PKG, 'package.json'));
    const minimumVersion = /^\^(\d+\.\d+\.\d+)/.exec(manifest.engines?.node ?? '')?.[1];
    expect(minimumVersion).toBeDefined();

    const workflow = readFileSync(resolve(REPO, '.github/workflows/ci.yml'), 'utf8');
    const releaseStart = workflow.indexOf('  release-readiness:');
    const releaseEnd = workflow.indexOf('\n  wheel-viewer:', releaseStart);
    expect(releaseStart).toBeGreaterThan(-1);
    expect(releaseEnd).toBeGreaterThan(releaseStart);

    const releaseJob = workflow.slice(releaseStart, releaseEnd);
    const build = releaseJob.indexOf('run: pnpm run ci:release');
    const minimumNode = releaseJob.indexOf(`node-version: ${minimumVersion}`);
    const smoke = releaseJob.indexOf('run: node scripts/check-lib-exports.mjs');

    expect(build).toBeGreaterThan(-1);
    expect(minimumNode).toBeGreaterThan(build);
    expect(smoke).toBeGreaterThan(minimumNode);
  });
});
