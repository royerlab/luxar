/**
 * Programmatic Python -> TypeScript array round-trip contract tests.
 *
 * Python generates fixtures and a `roundtrip_expectations.json` file by decoding
 * every relevant fixture array with `luxar.encoding.ArrayDecoder`. These tests
 * load the same zarr arrays in Node.js and require the TypeScript ArrayDecoder to
 * reproduce the same flattened float32 values byte-for-byte.
 *
 * This is intentionally non-browser and non-WebGL so it can run in the fast
 * Vitest phase while still exercising real Python-written zarr stores.
 */

import { describe, it, expect } from 'vitest';
import { ArrayDecoder, ArrayRefRegistry } from '../../../data/utils/array-decoder';
import type { ArrayMetadata } from '../../../data/utils/array-decoder';
import * as zarr from 'zarrita';
import { FileSystemStore } from '@zarrita/storage';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { readFileSync } from 'fs';
import { createHash } from 'crypto';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FIXTURES_DIR = path.resolve(__dirname, '../../../../tests/fixtures');
const EXPECTATIONS_PATH = path.join(FIXTURES_DIR, 'roundtrip_expectations.json');

interface ArrayExpectation {
  path: string;
  encoding: string;
  storage_shape: number[];
  storage_dtype: string;
  decoded_shape: number[];
  decoded_dtype: string;
  expected_elements: number;
  flat_length: number;
  float32_sha256: string;
  samples: Array<{ index: number; value: number }>;
  stats: { min: number | null; max: number | null; mean: number | null };
}

interface FixtureExpectation {
  arrays: Record<string, ArrayExpectation>;
}

interface RoundTripExpectations {
  version: number;
  fixtures: Record<string, FixtureExpectation>;
}

const EXPECTATIONS = JSON.parse(readFileSync(EXPECTATIONS_PATH, 'utf-8')) as RoundTripExpectations;

async function loadArrayWithAttrs(
  datasetName: string,
  arrayPath: string
): Promise<{
  array: zarr.Array<zarr.DataType, zarr.Readable>;
  attrs: ArrayMetadata;
  rootLoc: zarr.Location<zarr.Readable>;
}> {
  const storePath = path.join(FIXTURES_DIR, datasetName);
  const rawStore = new FileSystemStore(storePath);
  const store = await zarr.tryWithConsolidated(rawStore);
  const rootLoc = zarr.root(store);
  const array = await zarr.open(rootLoc.resolve(arrayPath), { kind: 'array' });
  const attrs = array.attrs as unknown as ArrayMetadata;
  return { array, attrs, rootLoc };
}

function float32Sha256(values: Float32Array): string {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  return createHash('sha256').update(Buffer.from(bytes)).digest('hex');
}

function shapeProduct(shape: number[]): number {
  return shape.reduce((product, value) => product * value, 1);
}

describe('Python-TypeScript encoded array round-trip', () => {
  it('uses the expected fixture expectation schema', () => {
    expect(EXPECTATIONS.version).toBe(1);
    expect(Object.keys(EXPECTATIONS.fixtures).length).toBeGreaterThan(0);
  });

  for (const [fixtureName, fixture] of Object.entries(EXPECTATIONS.fixtures)) {
    describe(fixtureName, () => {
      for (const [arrayPath, expected] of Object.entries(fixture.arrays)) {
        it(`decodes ${arrayPath} (${expected.encoding}) like Python`, async () => {
          const { array, attrs, rootLoc } = await loadArrayWithAttrs(fixtureName, arrayPath);
          const decoder = new ArrayDecoder(new ArrayRefRegistry());

          // Do not pass expectedElements here. The decoder must rely on Python's
          // encoding metadata (not caller hints) for full-array round-trips.
          const decoded = await decoder.decode(array, attrs, undefined, rootLoc);

          expect(decoded.length).toBe(expected.flat_length);
          expect(decoded.length).toBe(shapeProduct(expected.decoded_shape));

          for (const sample of expected.samples) {
            expect(decoded[sample.index]).toBeCloseTo(sample.value, 6);
          }

          expect(float32Sha256(decoded)).toBe(expected.float32_sha256);
        });
      }
    });
  }
});
