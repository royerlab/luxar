import { readFileSync } from 'node:fs';
import { describe, it, expect } from 'vitest';
import { computeCacheBudgets } from '../../../cache/heap-budget';

const GiB = 1024 ** 3;

function readme(): string {
  return readFileSync(new URL('../../../cache/README.md', import.meta.url), 'utf8').replace(
    /\s+/g,
    ' '
  );
}

describe('cache README', () => {
  it('states the actual S-cache cap', () => {
    const capBytes = computeCacheBudgets(1e12).sliceBytes;
    const stated = [...readme().matchAll(/S-cache[^.]*?capped at[^.]*?(\d+(?:\.\d+)?) ?GiB/g)].map(
      (match) => Number(match[1])
    );

    expect(stated).toHaveLength(2);
    for (const gib of stated) expect(gib * GiB).toBe(capBytes);
  });

  it('documents validation for hash-less external datasets', () => {
    const externalDatasets = readme()
      .split('**External datasets**')[1]
      ?.split('## Performance')[0];

    expect(externalDatasets).toBeDefined();
    expect(externalDatasets).toContain('zattrs-hash');
    expect(externalDatasets).not.toMatch(/validation (?:is )?skipped/);
  });
});
