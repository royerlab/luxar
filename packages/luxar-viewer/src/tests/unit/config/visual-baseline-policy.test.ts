import { execFileSync } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const E2E_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../e2e');

function committedVisualBaselines(): string[] {
  return execFileSync('git', ['ls-files', '--', '*-snapshots/*.png'], {
    cwd: E2E_ROOT,
    encoding: 'utf8',
  })
    .split('\n')
    .filter(Boolean)
    .sort();
}

describe('visual baseline policy', () => {
  it('keeps only reproducible Linux Chromium baselines', () => {
    const baselines = committedVisualBaselines();
    expect(baselines.length).toBeGreaterThan(0);
    expect(baselines.filter((baseline) => !baseline.endsWith('-chromium-linux.png'))).toEqual([]);
  });

  it('ignores untracked platform snapshots', () => {
    const snapshotDir = path.join(E2E_ROOT, 'visual-baseline-policy.spec.ts-snapshots');
    const untrackedBaseline = path.join(snapshotDir, 'untracked-chromium-darwin.png');

    try {
      mkdirSync(snapshotDir, { recursive: true });
      writeFileSync(untrackedBaseline, 'not a tracked baseline');
      expect(committedVisualBaselines()).not.toContain(
        'visual-baseline-policy.spec.ts-snapshots/untracked-chromium-darwin.png'
      );
    } finally {
      rmSync(snapshotDir, { recursive: true });
    }
  });
});
