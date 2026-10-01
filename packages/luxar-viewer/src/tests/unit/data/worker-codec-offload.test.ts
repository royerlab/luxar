/**
 * Blosc chunk decompression is OFFLOADED to the data-worker pool.
 *
 * Measured in real Chrome on a cold h2afva playback tick: blosc/zstd ran on the
 * MAIN thread (53 ms/tick) and the luxar_delta decode after it (36 ms/tick)
 * while 13 of 14 data workers idled. These tests pin the contract that replaces
 * that: once the pool has a usable worker, a zarr read of a blosc-compressed
 * chunk runs NO blosc decode on the main thread; the compressed bytes go to a
 * worker and the decoded bytes come back. The fake workers run the codec
 * in-process (a flag distinguishes "inside the worker" from the main thread).
 */

import * as path from 'path';
import { fileURLToPath } from 'url';

import { FileSystemStore } from '@zarrita/storage';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

import * as zarr from '../../../data/zarr';
import { setWorkerCodecsEnabled } from '../../../data/codecs/worker-blosc';
import { hasUrlFlag, URL_PARAM_KEYS } from '../../../config/url-params';
import { disposeWorkerPool, getWorkerPool, warmWorkerCodecs } from '../../../workers/worker-pool';
import {
  BloscDecodeDispatcher,
  MAX_BATCH_BYTES,
  MAX_BATCH_CHUNKS,
  MIN_OFFLOAD_DECODED_BYTES,
  type CodecPoolPort,
} from '../../../workers/worker-pool/codec-dispatch';

const here = path.dirname(fileURLToPath(import.meta.url));
const GOLDEN = path.join(here, 'codecs/golden-chunks');

interface NativeCodec {
  decode(data: Uint8Array): Promise<Uint8Array>;
}
interface NativeCtor {
  fromConfig(config: object): NativeCodec;
  prototype: NativeCodec;
}

type WorkerItem = { bytes: Uint8Array; delta: { cols: number; bits: number } | null };

let Native: NativeCtor;
/** True only while a FAKE WORKER is running a decode (vs the main thread). */
let inWorker = false;
let mainThreadDecodes = 0;

async function loadNative(): Promise<NativeCtor> {
  return (await zarr.loadNativeBlosc()) as unknown as NativeCtor;
}

/** A fake pool worker whose `decodeBloscBatch` decodes in-process. */
function makeFakeWorker() {
  const calls: WorkerItem[][] = [];
  const api = {
    warmCodecs: vi.fn(async () => undefined),
    decodeBloscBatch: vi.fn(async (items: WorkerItem[]) => {
      calls.push(items);
      const codec = Native.fromConfig({});
      const out = [];
      for (const item of items) {
        inWorker = true;
        const pending = codec.decode(item.bytes);
        inWorker = false;
        out.push({ data: await pending });
      }
      return out;
    }),
  };
  return { instance: { worker: { terminate: vi.fn() }, api, activeQueries: 0 }, calls, api };
}

function injectWorkers(instances: unknown[]): void {
  const pool = getWorkerPool() as unknown as {
    workers: unknown[];
    initPromise: Promise<void>;
  };
  pool.workers = instances;
  pool.initPromise = Promise.resolve();
}

/** Inject workers and let their codec warm-up finish (offload needs a warm worker). */
async function injectWarmWorkers(instances: { api: { warmCodecs: () => Promise<unknown> } }[]) {
  injectWorkers(instances);
  warmWorkerCodecs();
  await vi.waitFor(() => {
    for (const w of instances) expect(w.api.warmCodecs).toHaveBeenCalled();
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
}

async function readGolden(name: string) {
  const store = new FileSystemStore(path.join(GOLDEN, name));
  const array = await zarr.openArray(zarr.root(store));
  return zarr.readArray(array);
}

describe('blosc decode offload to the data-worker pool', () => {
  beforeAll(async () => {
    Native = await loadNative();
    const original = Native.prototype.decode;
    vi.spyOn(Native.prototype, 'decode').mockImplementation(function (
      this: NativeCodec,
      data: Uint8Array
    ) {
      if (!inWorker) mainThreadDecodes++;
      return original.call(this, data);
    });
  });

  beforeEach(() => {
    disposeWorkerPool();
    mainThreadDecodes = 0;
  });

  afterEach(() => {
    disposeWorkerPool();
  });

  it('runs no blosc decode on the main thread once a pool worker is warm', async () => {
    const fake = makeFakeWorker();
    await injectWarmWorkers([fake.instance]);

    const { data } = await readGolden('v3_u16_zstd_shuffle');

    expect(mainThreadDecodes).toBe(0);
    expect(fake.api.decodeBloscBatch).toHaveBeenCalledTimes(1);
    expect(data.length).toBe(5120 * 3);
  });

  it('decodes on the main thread when no pool exists', async () => {
    const { data } = await readGolden('v3_u16_zstd_shuffle');
    expect(mainThreadDecodes).toBe(1);
    expect(data.length).toBe(5120 * 3);
  });

  it('keeps a small real chunk on the main thread even with a pool', async () => {
    const fake = makeFakeWorker();
    injectWorkers([fake.instance]);
    await readGolden('v2_u16_zstd_bitshuffle'); // 45x3 uint16 = 270 decoded bytes
    expect(fake.api.decodeBloscBatch).not.toHaveBeenCalled();
    expect(mainThreadDecodes).toBe(1);
  });

  it('decodes on the main thread while the pool has no usable worker yet', async () => {
    getWorkerPool(); // pool exists (backend installed) but no worker published
    await readGolden('v3_u16_zstd_shuffle');
    expect(mainThreadDecodes).toBe(1);
  });

  it('the ?mainThreadCodecs kill switch keeps decodes on the main thread', async () => {
    const fake = makeFakeWorker();
    injectWorkers([fake.instance]);
    setWorkerCodecsEnabled(false);
    try {
      await readGolden('v3_u16_zstd_shuffle');
    } finally {
      setWorkerCodecsEnabled(true);
    }
    expect(mainThreadDecodes).toBe(1);
    expect(fake.api.decodeBloscBatch).not.toHaveBeenCalled();
    expect(hasUrlFlag(URL_PARAM_KEYS.mainThreadCodecs, '?src=x&mainThreadCodecs')).toBe(true);
    expect(hasUrlFlag(URL_PARAM_KEYS.mainThreadCodecs, '?src=x')).toBe(false);
  });

  it('falls back to the main thread when the worker call fails', async () => {
    const fake = makeFakeWorker();
    fake.api.decodeBloscBatch.mockRejectedValueOnce(new Error('worker crashed'));
    await injectWarmWorkers([fake.instance]);
    const { data } = await readGolden('v3_u16_zstd_shuffle');
    expect(fake.api.decodeBloscBatch).toHaveBeenCalledTimes(1);
    expect(mainThreadDecodes).toBe(1);
    expect(data.length).toBe(5120 * 3);
  });

  it('a pool-wide abort does not abandon an in-flight decode', async () => {
    const fake = makeFakeWorker();
    await injectWarmWorkers([fake.instance]);
    const controller = new AbortController();
    getWorkerPool().setAbortSignal(controller.signal);
    const pending = readGolden('v3_u16_zstd_shuffle');
    controller.abort();
    await pending;
    expect(mainThreadDecodes).toBe(0);
    expect(fake.api.decodeBloscBatch).toHaveBeenCalledTimes(1);
  });

  it('the first big read decodes on the main thread while ONE worker warms', async () => {
    const fake = makeFakeWorker();
    const other = makeFakeWorker();
    let finishWarm: () => void = () => {};
    fake.api.warmCodecs.mockImplementation(
      () => new Promise<undefined>((resolve) => (finishWarm = () => resolve(undefined)))
    );
    injectWorkers([fake.instance, other.instance]);

    await readGolden('v3_u16_zstd_shuffle');
    expect(mainThreadDecodes).toBe(1);
    expect(fake.api.decodeBloscBatch).not.toHaveBeenCalled();
    expect(fake.api.warmCodecs).toHaveBeenCalledTimes(1);
    // The second worker waits for the first (it will hit the HTTP cache).
    expect(other.api.warmCodecs).not.toHaveBeenCalled();
    // A warm-up is not a query: the warming worker still reads as idle.
    expect(fake.instance.activeQueries).toBe(0);
    expect(getWorkerPool().getIdleWorkerCount()).toBe(2);

    finishWarm();
    await vi.waitFor(() => expect(other.api.warmCodecs).toHaveBeenCalledTimes(1));
    await readGolden('v3_u16_zstd_shuffle');
    expect(mainThreadDecodes).toBe(1);
  });
});

describe('BloscDecodeDispatcher batching', () => {
  type Item = { bytes: Uint8Array };

  /** A blosc-framed stand-in: only the header's decoded size (bytes 4..7) matters here. */
  function frame(decodedBytes: number): Uint8Array {
    const bytes = new Uint8Array(32);
    new DataView(bytes.buffer).setUint32(4, decodedBytes, true);
    return bytes;
  }

  function makePort(idle: number) {
    const batches: Item[][] = [];
    const port: CodecPoolPort = {
      isInitialized: () => true,
      getIdleWarmWorkerCount: () => idle,
      ensureCodecsWarm: () => true,
      runDecode: (_op, fn) =>
        fn({
          decodeBloscBatch: async (items: Item[]) => {
            batches.push(items);
            return items.map((item) => ({ data: item.bytes }));
          },
        } as never),
    };
    return { port, batches };
  }

  function decodeMany(
    dispatcher: BloscDecodeDispatcher,
    n: number,
    size = MIN_OFFLOAD_DECODED_BYTES
  ) {
    return Promise.all(
      Array.from({ length: n }, () => {
        const job = dispatcher.decode({ bytes: frame(size), delta: null });
        if (!job) throw new Error('dispatcher declined');
        return job;
      })
    );
  }

  it('declines (returns null) while the pool has no usable worker', () => {
    const { port } = makePort(4);
    const dispatcher = new BloscDecodeDispatcher({ ...port, isInitialized: () => false });
    expect(dispatcher.decode({ bytes: frame(10), delta: null })).toBeNull();
  });

  it('declines an above-floor chunk while no worker codec is warm', () => {
    const { port, batches } = makePort(4);
    const dispatcher = new BloscDecodeDispatcher({ ...port, ensureCodecsWarm: () => false });
    expect(dispatcher.decode({ bytes: frame(MIN_OFFLOAD_DECODED_BYTES), delta: null })).toBeNull();
    expect(batches).toEqual([]);
  });

  it('one idle worker: 20 same-microtask decodes -> 3 messages (8 + 8 + 4)', async () => {
    const { port, batches } = makePort(1);
    const out = await decodeMany(new BloscDecodeDispatcher(port), 20);
    expect(out).toHaveLength(20);
    expect(batches.map((b) => b.length)).toEqual([MAX_BATCH_CHUNKS, MAX_BATCH_CHUNKS, 4]);
  });

  it('declines a chunk too small to be worth the round trip', () => {
    const { port, batches } = makePort(4);
    const dispatcher = new BloscDecodeDispatcher(port);
    expect(
      dispatcher.decode({ bytes: frame(MIN_OFFLOAD_DECODED_BYTES - 1), delta: null })
    ).toBeNull();
    expect(batches).toEqual([]);
  });

  it('sends the worker a COPY of the borrowed (L1-owned) bytes', async () => {
    const { port, batches } = makePort(1);
    const original = frame(MIN_OFFLOAD_DECODED_BYTES);
    await new BloscDecodeDispatcher(port).decode({ bytes: original, delta: null });
    const sent = batches[0][0].bytes;
    expect(sent).not.toBe(original);
    expect(sent.buffer).not.toBe(original.buffer);
    expect(sent).toEqual(original);
  });

  it('spreads a flush evenly over the idle workers', async () => {
    const { port, batches } = makePort(3);
    await decodeMany(new BloscDecodeDispatcher(port), 12);
    expect(batches.map((b) => b.length)).toEqual([4, 4, 4]);
  });

  it('caps a message at MAX_BATCH_BYTES of decoded output', async () => {
    const { port, batches } = makePort(1);
    await decodeMany(new BloscDecodeDispatcher(port), 3, 5 * 1024 * 1024);
    expect(batches.map((b) => b.length)).toEqual([1, 1, 1]);
    expect(MAX_BATCH_BYTES).toBe(8 * 1024 * 1024);
  });

  it('decodes issued in different microtasks-chains go in separate messages', async () => {
    const { port, batches } = makePort(1);
    const dispatcher = new BloscDecodeDispatcher(port);
    await decodeMany(dispatcher, 2);
    await decodeMany(dispatcher, 2);
    expect(batches.map((b) => b.length)).toEqual([2, 2]);
  });

  it('rejects only the chunk the worker reported an error for', async () => {
    const port: CodecPoolPort = {
      isInitialized: () => true,
      getIdleWarmWorkerCount: () => 1,
      ensureCodecsWarm: () => true,
      runDecode: async () => [{ error: 'bad chunk' }, { data: new Uint8Array([1]) }] as never,
    };
    const dispatcher = new BloscDecodeDispatcher(port);
    const [a, b] = await Promise.allSettled([
      dispatcher.decode({ bytes: frame(MIN_OFFLOAD_DECODED_BYTES), delta: null }),
      dispatcher.decode({ bytes: frame(MIN_OFFLOAD_DECODED_BYTES), delta: null }),
    ]);
    expect(a).toMatchObject({ status: 'rejected' });
    expect(b).toEqual({ status: 'fulfilled', value: new Uint8Array([1]) });
  });
});
