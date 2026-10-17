import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, it } from 'vitest';

const script = 'scripts/ab-webgpu-vs-webgl.mjs';

it.each([
  { args: ['--help'], status: 0, output: /--min-ssim <value>/ },
  { args: ['--backend', 'webgpu'], status: 2, output: /unknown option --backend/ },
  { args: ['--dataset'], status: 2, output: /--dataset requires a value/ },
  { args: ['--dataset', '--bloom'], status: 2, output: /--dataset requires a value/ },
])('validates $args before starting servers or writing output', ({ args, status, output }) => {
  const root = mkdtempSync(join(tmpdir(), 'luxar-ab-cli-'));
  const out = join(root, 'output');
  try {
    const result = spawnSync(process.execPath, [script, '--out', out, ...args], {
      encoding: 'utf8',
      timeout: 20_000,
    });
    expect(result.error).toBeUndefined();
    expect(result.status).toBe(status);
    expect(result.stdout + result.stderr).toMatch(output);
    expect(existsSync(out)).toBe(false);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
