// Wall-clock benchmarks must not gate required PR checks. Keep them in the
// opt-in perf suite (`pnpm test:perf` / `pnpm bench:wasm`); see #2165.
import { readdirSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import mainConfig from '../../../../vitest.config';
import perfConfig from '../../../../vitest.perf.config';

const WASM_TEST_PREFIX = 'src/tests/unit/wasm/';
const wasmTestDirectory = new URL('../wasm/', import.meta.url);

describe('Vitest performance suite boundary', () => {
  it('keeps every WASM performance test out of the required suite', () => {
    const performanceTests = readdirSync(wasmTestDirectory)
      .filter((fileName) => /(?:benchmark|perf|performance).*\.test\.ts$/.test(fileName))
      .map((fileName) => `${WASM_TEST_PREFIX}${fileName}`)
      .sort();

    const mainExcludes = mainConfig.test.exclude;
    const perfIncludes = perfConfig.test.include;

    expect(perfIncludes).toEqual(performanceTests);
    expect(mainExcludes).toEqual(expect.arrayContaining(performanceTests));
  });
});
