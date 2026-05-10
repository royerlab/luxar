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
import { RangeLoader } from '../../../data/loaders/range-loader';
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

interface ArrayOperationExpectation {
  kind: 'full' | 'range';
  start?: number;
  end?: number;
  decoded_shape: number[];
  flat_length: number;
  float32_sha256: string;
  samples: Array<{ index: number; value: number }>;
  stats: { min: number | null; max: number | null; mean: number | null };
}

interface ArrayExpectation {
  path: string;
  encoding: string;
  storage_shape: number[];
  storage_dtype: string;
  decoded_shape: number[];
  decoded_dtype: string;
  expected_elements: number;
  shape_class: string;
  flat_length: number;
  float32_sha256: string;
  samples: Array<{ index: number; value: number }>;
  stats: { min: number | null; max: number | null; mean: number | null };
  operations: ArrayOperationExpectation[];
  contract_case?: { case_id?: string; semantic_type?: string; description?: string };
}

interface FixtureExpectation {
  arrays: Record<string, ArrayExpectation>;
}

interface ContractManifest {
  fixture_count: number;
  array_count: number;
  observed: Record<string, Record<string, number>>;
  required: Record<string, string[]>;
}

interface RoundTripExpectations {
  version: number;
  manifest: ContractManifest;
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
  store: zarr.Readable;
}> {
  const storePath = path.join(FIXTURES_DIR, datasetName);
  const rawStore = new FileSystemStore(storePath);
  const store = await zarr.tryWithConsolidated(rawStore);
  const rootLoc = zarr.root(store);
  const array = await zarr.open(rootLoc.resolve(arrayPath), { kind: 'array' });
  const attrs = array.attrs as unknown as ArrayMetadata;
  return { array, attrs, rootLoc, store };
}

function float32Sha256(values: Float32Array): string {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  return createHash('sha256').update(Buffer.from(bytes)).digest('hex');
}

function shapeProduct(shape: number[]): number {
  return shape.reduce((product, value) => product * value, 1);
}

function elementsPerItem(shape: number[]): number {
  if (shape.length <= 1) return 1;
  return shape.slice(1).reduce((product, value) => product * value, 1);
}

function assertSamples(
  values: Float32Array,
  expected: ArrayOperationExpectation | ArrayExpectation
): void {
  for (const sample of expected.samples) {
    expect(values[sample.index]).toBeCloseTo(sample.value, 6);
  }
}

function assertManifestCoverage(manifest: ContractManifest): void {
  for (const [category, requiredValues] of Object.entries(manifest.required)) {
    const observed = manifest.observed[category] ?? {};
    const missing = requiredValues.filter((value) => !observed[value]);
    expect(missing, `missing ${category} coverage`).toEqual([]);
  }
}

async function decodeRange(
  array: zarr.Array<zarr.DataType, zarr.Readable>,
  attrs: ArrayMetadata,
  store: zarr.Readable,
  operation: ArrayOperationExpectation
): Promise<Float32Array> {
  if (operation.kind !== 'range') {
    throw new Error(`decodeRange only accepts range operations, got ${operation.kind}`);
  }
  const start = operation.start ?? 0;
  const end = operation.end ?? start;
  if (start === end) return new Float32Array(0);

  const rangeLoader = new RangeLoader(new ArrayRefRegistry(), {
    workerThreshold: Number.MAX_SAFE_INTEGER,
  });
  rangeLoader.setVerbose(false);

  const output = new Float32Array(operation.flat_length);
  const written = await rangeLoader.loadRangesResolvingRef(
    array as unknown as zarr.Array<zarr.DataType, zarr.FetchStore>,
    attrs,
    [{ start, end }],
    output,
    end - start,
    elementsPerItem(operation.decoded_shape),
    store,
    'array-roundtrip'
  );
  expect(written).toBe(operation.flat_length);
  return output;
}

describe('Python-TypeScript encoded array round-trip', () => {
  it('uses the expected fixture expectation schema and coverage manifest', () => {
    expect(EXPECTATIONS.version).toBe(2);
    expect(Object.keys(EXPECTATIONS.fixtures).length).toBeGreaterThan(0);
    expect(EXPECTATIONS.manifest.fixture_count).toBe(Object.keys(EXPECTATIONS.fixtures).length);
    expect(EXPECTATIONS.manifest.array_count).toBeGreaterThan(100);
    assertManifestCoverage(EXPECTATIONS.manifest);
  });

  for (const [fixtureName, fixture] of Object.entries(EXPECTATIONS.fixtures)) {
    describe(fixtureName, () => {
      for (const [arrayPath, expected] of Object.entries(fixture.arrays)) {
        it(`decodes ${arrayPath} (${expected.encoding}) like Python`, async () => {
          const { array, attrs, rootLoc, store } = await loadArrayWithAttrs(fixtureName, arrayPath);
          const decoder = new ArrayDecoder(new ArrayRefRegistry());

          // Do not pass expectedElements here. The decoder must rely on Python's
          // encoding metadata (not caller hints) for full-array round-trips.
          const decoded = await decoder.decode(array, attrs, undefined, rootLoc);

          expect(decoded.length).toBe(expected.flat_length);
          expect(decoded.length).toBe(shapeProduct(expected.decoded_shape));
          assertSamples(decoded, expected);
          expect(float32Sha256(decoded)).toBe(expected.float32_sha256);

          const fullOperation = expected.operations.find((operation) => operation.kind === 'full');
          expect(fullOperation?.float32_sha256).toBe(expected.float32_sha256);

          for (const operation of expected.operations.filter((item) => item.kind === 'range')) {
            const rangeDecoded = await decodeRange(array, attrs, store, operation);
            expect(rangeDecoded.length).toBe(operation.flat_length);
            expect(rangeDecoded.length).toBe(shapeProduct(operation.decoded_shape));
            assertSamples(rangeDecoded, operation);
            expect(float32Sha256(rangeDecoded)).toBe(operation.float32_sha256);
          }
        });
      }
    });
  }
});
