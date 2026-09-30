/**
 * Lifetime of a data worker's blosc codec (`workers/data-worker/decode/blosc.ts`).
 *
 * Each worker is its own JS realm, so it necessarily holds its own blosc WASM
 * instance; what must NOT happen is re-instantiating it per decode call, or a
 * cold (never-warmed) worker decoding differently from a warm one. The module
 * is re-imported per test so each test starts as a fresh worker.
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { NativeBloscCtor } from '../../../../data/codecs/worker-blosc';

const counts = { loads: 0, instances: 0, failNextLoad: false };

vi.mock('../../../../data/zarr', async (importOriginal) => {
  const real = await importOriginal<typeof import('../../../../data/zarr')>();
  return {
    ...real,
    loadNativeBlosc: async (): Promise<NativeBloscCtor> => {
      counts.loads++;
      if (counts.failNextLoad) {
        counts.failNextLoad = false;
        throw new Error('codec chunk fetch failed');
      }
      const Ctor = await real.loadNativeBlosc();
      return {
        fromConfig: (config: Record<string, unknown>) => {
          counts.instances++;
          return Ctor.fromConfig(config);
        },
      };
    },
  };
});

const here = path.dirname(fileURLToPath(import.meta.url));
const GOLDEN = path.join(here, '../../data/codecs/golden-chunks');
// A format-3 fixture without a delta: its only chunk is `c/0` (1-D) or `c/0/0`.
const FIXTURE = fs
  .readdirSync(GOLDEN)
  .sort()
  .find((n) => fs.existsSync(path.join(GOLDEN, n, 'zarr.json')) && !n.includes('delta'));

function chunkBytes(): Uint8Array {
  const dir = path.join(GOLDEN, FIXTURE!);
  const meta = JSON.parse(fs.readFileSync(path.join(dir, 'zarr.json'), 'utf8'));
  const shape: number[] = meta.chunk_grid.configuration.chunk_shape;
  return new Uint8Array(fs.readFileSync(path.join(dir, 'c', ...shape.map(() => '0'))));
}

type WorkerModule = typeof import('../../../../workers/data-worker/decode/blosc');

async function freshWorker(): Promise<WorkerModule> {
  vi.resetModules();
  return import('../../../../workers/data-worker/decode/blosc');
}

describe('data-worker blosc codec lifetime', () => {
  beforeEach(() => {
    counts.loads = 0;
    counts.instances = 0;
    counts.failNextLoad = false;
  });

  afterEach(() => {
    vi.resetModules();
  });

  it('instantiates the codec once per worker: warm-up and every later batch share it', async () => {
    const worker = await freshWorker();
    const bytes = chunkBytes();
    await worker.warmCodecs();
    await worker.warmCodecs();
    for (let i = 0; i < 3; i++) {
      await worker.decodeBloscBatch([{ bytes: bytes.slice(), delta: null }]);
    }
    expect(counts.loads).toBe(1);
    expect(counts.instances).toBe(1);
  });

  it('a COLD worker (never warmed) decodes its first batch exactly like a warm one', async () => {
    const warm = await freshWorker();
    await warm.warmCodecs();
    const [expected] = await warm.decodeBloscBatch([{ bytes: chunkBytes(), delta: null }]);

    const cold = await freshWorker();
    counts.loads = 0;
    counts.instances = 0;
    // Two concurrent batches racing the lazy instantiation still share one codec.
    const [a, b] = await Promise.all([
      cold.decodeBloscBatch([{ bytes: chunkBytes(), delta: null }]),
      cold.decodeBloscBatch([{ bytes: chunkBytes(), delta: null }]),
    ]);
    expect(a[0]).toEqual(expected);
    expect(b[0]).toEqual(expected);
    expect(counts.loads).toBe(1);
    expect(counts.instances).toBe(1);
  });

  it('a failed codec load does not poison the worker: the next call retries', async () => {
    const worker = await freshWorker();
    counts.failNextLoad = true;
    await expect(worker.warmCodecs()).rejects.toThrow('codec chunk fetch failed');
    const [result] = await worker.decodeBloscBatch([{ bytes: chunkBytes(), delta: null }]);
    expect(result).toHaveProperty('data');
    expect(counts.loads).toBe(2);
    expect(counts.instances).toBe(1);
  });
});
