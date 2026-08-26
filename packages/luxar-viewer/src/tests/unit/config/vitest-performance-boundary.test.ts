import { readFileSync, readdirSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const WASM_TEST_PREFIX = 'src/tests/unit/wasm/';
const wasmTestDirectory = new URL('../wasm/', import.meta.url);
const mainConfigPath = new URL('../../../../vitest.config.ts', import.meta.url);
const perfConfigPath = new URL('../../../../vitest.perf.config.ts', import.meta.url);

function wasmTestPaths(configPath: URL, arrayName: 'exclude' | 'include'): string[] {
  const source = readFileSync(configPath, 'utf8');
  const array = source.match(new RegExp(`${arrayName}: \\[([\\s\\S]*?)\\n    \\]`))?.[1] ?? '';
  return [...array.matchAll(/'(?<path>src\/tests\/unit\/wasm\/[^']+\.test\.ts)'/g)]
    .map((match) => match.groups!.path)
    .sort();
}

describe('Vitest performance suite boundary', () => {
  it('keeps every WASM performance test out of the required suite', () => {
    const performanceTests = readdirSync(wasmTestDirectory)
      .filter((fileName) => /(?:benchmark|perf|performance).*\.test\.ts$/.test(fileName))
      .map((fileName) => `${WASM_TEST_PREFIX}${fileName}`)
      .sort();

    const mainExcludes = wasmTestPaths(mainConfigPath, 'exclude');
    const perfIncludes = wasmTestPaths(perfConfigPath, 'include');

    expect(perfIncludes).toEqual(performanceTests);
    expect(mainExcludes).toEqual(expect.arrayContaining(performanceTests));
  });
});
