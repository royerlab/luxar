import { describe, expect, it, vi } from 'vitest';

interface VitestConfigModule {
  default: {
    test?: {
      coverage?: {
        exclude?: unknown[];
      };
    };
  };
}

describe('Vitest coverage boundary', () => {
  it('excludes test helpers with explicit recursive globs', async () => {
    const { default: config } = await vi.importActual<VitestConfigModule>(
      '../../../../vitest.config'
    );
    const exclusions = config.test?.coverage?.exclude ?? [];

    expect(exclusions).toContain('src/tests/**');
    expect(exclusions.every((entry) => typeof entry === 'string' && !entry.endsWith('/'))).toBe(
      true
    );
  });
});
