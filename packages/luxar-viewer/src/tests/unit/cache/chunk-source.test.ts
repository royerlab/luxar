import { describe, expect, it } from 'vitest';

import { ArchiveFaultError, archiveFaultFrom } from '../../../cache/chunk-source';

describe('archiveFaultFrom', () => {
  it('finds an archive fault through ordinary Error causes', () => {
    const fault = new ArchiveFaultError('archive unavailable', 'https://example.test/data.zip');
    const wrapped = new Error('outer', { cause: new Error('inner', { cause: fault }) });

    expect(archiveFaultFrom(wrapped)).toBe(fault);
  });

  it('returns undefined for unrelated and cyclic cause chains', () => {
    const cyclic = new Error('cyclic');
    cyclic.cause = cyclic;

    expect(archiveFaultFrom(new Error('ordinary'))).toBeUndefined();
    expect(archiveFaultFrom(cyclic)).toBeUndefined();
  });
});
