// Wall-clock benchmarks must not gate required PR checks. Keep them in the
// opt-in perf suite (`pnpm test:perf` / `pnpm bench:wasm`); see #2165.
import { readdirSync } from 'node:fs';
import { afterEach, describe, expect, it, vi } from 'vitest';

const WASM_TEST_PREFIX = 'src/tests/unit/wasm/';
const wasmTestDirectory = new URL('../wasm/', import.meta.url);

interface VitestConfigModule {
  default: {
    test?: {
      exclude?: unknown[];
      hookTimeout?: number;
      include?: unknown[];
      maxWorkers?: number;
      testTimeout?: number;
    };
  };
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
});

describe('Vitest performance suite boundary', () => {
  it('keeps every WASM performance test out of the required suite', async () => {
    const performanceTests = readdirSync(wasmTestDirectory)
      .filter((fileName) => /(?:benchmark|perf|performance).*\.test\.ts$/.test(fileName))
      .map((fileName) => `${WASM_TEST_PREFIX}${fileName}`)
      .sort();

    const [{ default: mainConfig }, { default: perfConfig }] = await Promise.all([
      vi.importActual<VitestConfigModule>('../../../../vitest.config'),
      vi.importActual<VitestConfigModule>('../../../../vitest.perf.config'),
    ]);
    const mainExcludes = mainConfig.test?.exclude ?? [];
    const perfIncludes = perfConfig.test?.include ?? [];

    expect(perfIncludes).toEqual(performanceTests);
    expect(mainExcludes).toEqual(expect.arrayContaining(performanceTests));
  });
});

describe('Vitest contention policy', () => {
  it.each([
    ['off CI', '', undefined],
    ['on CI', '1', 4],
  ] as const)(
    'keeps 60 s budgets %s while preserving the CI worker cap',
    async (_label, ci, maxWorkers) => {
      vi.stubEnv('CI', ci);
      vi.resetModules();

      const { default: config } = await vi.importActual<VitestConfigModule>(
        '../../../../vitest.config'
      );

      expect(config.test).toMatchObject({
        testTimeout: 60_000,
        hookTimeout: 60_000,
        maxWorkers,
      });
    }
  );
});
