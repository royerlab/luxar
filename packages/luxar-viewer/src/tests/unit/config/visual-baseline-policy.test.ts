import { readdirSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const E2E_ROOT = path.resolve(process.cwd(), 'src/tests/e2e');

function committedVisualBaselines(): string[] {
  return (readdirSync(E2E_ROOT, { recursive: true }) as string[])
    .filter((entry) => entry.endsWith('.png') && entry.includes('-snapshots/'))
    .sort();
}

describe('visual baseline policy', () => {
  it('keeps only reproducible Linux Chromium baselines', () => {
    const baselines = committedVisualBaselines();
    expect(baselines.length).toBeGreaterThan(0);
    expect(baselines.filter((baseline) => !baseline.endsWith('-chromium-linux.png'))).toEqual([]);
  });
});
