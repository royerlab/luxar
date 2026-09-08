import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const PKG = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const REPO = resolve(PKG, '../..');
const SUPPORTED_NODE_RANGE = '^20.19.0 || >=22.12.0';

function packageJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

describe('published Node compatibility', () => {
  it('states the consumer range adopted from Vite 8.2', () => {
    const manifest = packageJson(resolve(PKG, 'package.json'));
    expect(manifest.engines?.node).toBe(SUPPORTED_NODE_RANGE);
  });

  it('smoke-imports the Node 22 build on the lowest supported Node', () => {
    const manifest = packageJson(resolve(PKG, 'package.json'));
    const minimumVersion = /^\^(\d+\.\d+\.\d+)/.exec(manifest.engines?.node ?? '')?.[1];
    expect(minimumVersion).toBeDefined();

    const workflow = readFileSync(resolve(REPO, '.github/workflows/ci.yml'), 'utf8');
    const releaseJob = /^ {2}release-readiness:\n[\s\S]*?(?=^ {2}[\w-]+:\n|(?![\s\S]))/m.exec(
      workflow
    )?.[0];
    expect(releaseJob).toBeDefined();

    const step = (name) => {
      const escapedName = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const block = new RegExp(
        `^ {6}- name: ${escapedName}\\n[\\s\\S]*?(?=^ {6}- name: |(?![\\s\\S]))`,
        'm'
      ).exec(releaseJob)?.[0];
      expect(block, `missing release-readiness step: ${name}`).toBeDefined();
      return block;
    };
    const guard = (block) => /^ {8}if: (.+)$/m.exec(block)?.[1];

    const build = step('Build library bundle (with WASM)');
    const minimumNode = step('Set up minimum supported Node');
    const smoke = step('Smoke-import library on minimum supported Node');

    expect(minimumNode).toContain(`node-version: ${minimumVersion}`);
    expect(smoke).toContain('run: node scripts/check-lib-exports.mjs');
    expect(guard(minimumNode)).toBe(guard(build));
    expect(guard(smoke)).toBe(guard(build));
    expect(releaseJob.indexOf(minimumNode)).toBeGreaterThan(releaseJob.indexOf(build));
    expect(releaseJob.indexOf(smoke)).toBeGreaterThan(releaseJob.indexOf(minimumNode));
  });
});
