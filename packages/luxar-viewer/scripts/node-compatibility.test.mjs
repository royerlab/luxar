import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const PKG = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const REPO = resolve(PKG, '../..');

describe('published Node compatibility', () => {
  it('declares Vite-compatible Node consumer ranges', () => {
    const manifest = JSON.parse(readFileSync(resolve(PKG, 'package.json'), 'utf8'));
    expect(manifest.engines?.node).toBe('^20.19.0 || >=22.12.0');
  });

  it('smoke-imports the Node 22 build on Node 20.19', () => {
    const workflow = readFileSync(resolve(REPO, '.github/workflows/ci.yml'), 'utf8');
    const releaseStart = workflow.indexOf('  release-readiness:');
    const releaseEnd = workflow.indexOf('\n  wheel-viewer:', releaseStart);
    expect(releaseStart).toBeGreaterThan(-1);
    expect(releaseEnd).toBeGreaterThan(releaseStart);

    const releaseJob = workflow.slice(releaseStart, releaseEnd);
    const build = releaseJob.indexOf('run: pnpm run ci:release');
    const minimumNode = releaseJob.indexOf('node-version: 20.19.0');
    const smoke = releaseJob.indexOf('run: node scripts/check-lib-exports.mjs');

    expect(build).toBeGreaterThan(-1);
    expect(minimumNode).toBeGreaterThan(build);
    expect(smoke).toBeGreaterThan(minimumNode);
  });
});
