/**
 * Exactness gate for the worker-offloaded blosc codec: a chunk decoded through
 * a data worker must be BYTE-IDENTICAL to the main-thread pipeline.
 *
 * Fixtures (`codecs/golden-chunks/`) are one-chunk arrays holding REAL chunks,
 * byte for byte:
 *  - `v3_*_zstd_*` / `v2_*_zstd_*`: copied from the render-gate stores
 *    (`datasets/gate/`, format 3) and the example stores (format 2) — zstd with
 *    shuffle / bitshuffle / noshuffle, uint8/16/32/64 + float32, and the
 *    `luxar_delta_v1` u8 / u16 filter as the Python writer emits it;
 *  - `*_lz4_*`, `*_blosclz_*`, `v3_u16_delta_zstd_bitshuffle`, `v2_*_delta_*`:
 *    written by zarr-python + numcodecs for the configurations no real store
 *    uses today (lz4, blosclz, bitshuffle at format 3, delta at format 2).
 * The delta filter is only defined for uint8/uint16 codes (see
 * `data/codecs/luxar-delta.ts`), so there is no signed / 32-bit delta case.
 *
 * Three decodes are compared per fixture:
 *  1. the main-thread zarrita pipeline (no backend: native blosc, then the
 *     delta codec) — the reference;
 *  2. an INDEPENDENT reference: native blosc on the raw chunk file + a fresh
 *     `LuxarDeltaCodec` decode, bypassing zarrita;
 *  3. the full pipeline with the worker decode function (`decodeBloscBatch`)
 *     run in-process as the backend — which exercises the fused delta and the
 *     delta pass-through that must keep it from being applied twice.
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

import { FileSystemStore } from '@zarrita/storage';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';

import * as zarr from '../../../data/zarr';
import { LuxarDeltaCodec } from '../../../data/codecs/luxar-delta';
import {
  WorkerBloscCodec,
  resolveFusedDelta,
  setBloscDecodeBackend,
  type BloscDecodeRequest,
  type NativeBloscCtor,
} from '../../../data/codecs/worker-blosc';
import { decodeBloscBatch } from '../../../workers/data-worker/decode/blosc';

const here = path.dirname(fileURLToPath(import.meta.url));
const GOLDEN = path.join(here, 'codecs/golden-chunks');
const FIXTURES = fs.readdirSync(GOLDEN).sort();

interface FixtureMeta {
  format: 2 | 3;
  dtype: string;
  chunkFile: string;
  cname: string;
  shuffle: string;
  delta: { cols: number; bits: number } | null;
}

const SHUFFLE_NAMES: Record<number, string> = { 0: 'noshuffle', 1: 'shuffle', 2: 'bitshuffle' };

function readMeta(name: string): FixtureMeta {
  const dir = path.join(GOLDEN, name);
  if (fs.existsSync(path.join(dir, 'zarr.json'))) {
    const m = JSON.parse(fs.readFileSync(path.join(dir, 'zarr.json'), 'utf8'));
    const blosc = m.codecs.find((c: { name: string }) => c.name === 'blosc').configuration;
    const delta = m.codecs.find((c: { name: string }) => c.name === 'luxar_delta_v1');
    const chunkShape: number[] = m.chunk_grid.configuration.chunk_shape;
    return {
      format: 3,
      dtype: m.data_type,
      chunkFile: ['c', ...chunkShape.map(() => '0')].join('/'),
      cname: blosc.cname,
      shuffle: blosc.shuffle,
      delta: delta ? delta.configuration : null,
    };
  }
  const m = JSON.parse(fs.readFileSync(path.join(dir, '.zarray'), 'utf8'));
  const delta = (m.filters ?? []).find((f: { id: string }) => f.id === 'luxar_delta_v1');
  return {
    format: 2,
    dtype: m.dtype,
    chunkFile: m.chunks.map(() => '0').join('.'),
    cname: m.compressor.cname,
    shuffle: SHUFFLE_NAMES[m.compressor.shuffle],
    delta: delta ? { cols: delta.cols, bits: delta.bits } : null,
  };
}

function bytesOf(view: ArrayBufferView): Uint8Array {
  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

async function readThroughPipeline(name: string): Promise<Uint8Array> {
  // A fresh array per read: zarrita resolves the codec chain per array, on
  // first decode, so each read sees the backend installed at that moment.
  const array = await zarr.openArray(zarr.root(new FileSystemStore(path.join(GOLDEN, name))));
  const { data } = await zarr.readArray(array);
  return bytesOf(data as ArrayBufferView).slice();
}

let Native: NativeBloscCtor;

describe('worker blosc codec — byte identity with the main-thread pipeline', () => {
  beforeAll(async () => {
    Native = await zarr.loadNativeBlosc();
  });

  afterEach(() => {
    setBloscDecodeBackend(null);
  });

  it('the fixture set covers every blosc + delta configuration the gate asks for', () => {
    const metas = FIXTURES.map(readMeta);
    const has = (pred: (m: FixtureMeta) => boolean): boolean => metas.some(pred);
    for (const cname of ['zstd', 'lz4', 'blosclz'])
      expect(has((m) => m.cname === cname)).toBe(true);
    for (const shuffle of ['noshuffle', 'shuffle', 'bitshuffle']) {
      expect(has((m) => m.shuffle === shuffle)).toBe(true);
    }
    for (const bits of [8, 16]) {
      for (const format of [2, 3]) {
        expect(has((m) => m.delta?.bits === bits && m.format === format)).toBe(true);
      }
    }
    expect(has((m) => m.delta !== null && m.cname === 'lz4')).toBe(true);
    expect(has((m) => m.delta !== null && m.cname === 'zstd')).toBe(true);
  });

  it.each(FIXTURES)(
    '%s: worker decode == main-thread decode == independent reference',
    async (name) => {
      const meta = readMeta(name);

      // 1. Main-thread pipeline (no backend installed).
      const mainThread = await readThroughPipeline(name);

      // 2. Independent reference, bypassing zarrita entirely.
      const raw = new Uint8Array(fs.readFileSync(path.join(GOLDEN, name, meta.chunkFile)));
      let reference = await Native.fromConfig({}).decode(raw);
      if (meta.delta) {
        const bits = meta.delta.bits as 8 | 16;
        const codes =
          bits === 8 ? reference : new Uint16Array(reference.buffer, 0, reference.byteLength / 2);
        reference = bytesOf(
          new LuxarDeltaCodec(meta.delta.cols, bits).decode({ data: codes, shape: [], stride: [] })
            .data
        );
      }
      expect(mainThread).toEqual(reference);

      // 3. Through the worker decode function, run in-process as the backend.
      const requests: BloscDecodeRequest[] = [];
      setBloscDecodeBackend({
        decode: async (request) => {
          requests.push(request);
          const [result] = await decodeBloscBatch([request]);
          if ('error' in result) throw new Error(result.error);
          return result.data;
        },
      });
      const viaWorker = await readThroughPipeline(name);

      expect(requests).toHaveLength(1);
      // The delta is fused into the worker decode exactly when the chain has one.
      expect(requests[0].delta).toEqual(meta.delta);
      expect(viaWorker.byteLength).toBe(mainThread.byteLength);
      expect(viaWorker).toEqual(mainThread);
    }
  );

  it('the worker batch decode equals native blosc chunk by chunk (no delta)', async () => {
    const items = FIXTURES.map((name) => {
      const meta = readMeta(name);
      return new Uint8Array(fs.readFileSync(path.join(GOLDEN, name, meta.chunkFile)));
    });
    const results = await decodeBloscBatch(
      items.map((bytes) => ({ bytes: bytes.slice(), delta: null }))
    );
    for (let i = 0; i < items.length; i++) {
      const expected = await Native.fromConfig({}).decode(items[i]);
      expect(results[i]).toEqual({ data: expected });
    }
  });

  it('a corrupt chunk fails alone inside a batch', async () => {
    const meta = readMeta(FIXTURES[0]);
    const good = new Uint8Array(fs.readFileSync(path.join(GOLDEN, FIXTURES[0], meta.chunkFile)));
    // A delta whose `cols` cannot divide the decoded chunk throws inside the
    // worker (blosc itself reports garbage input by returning empty output,
    // identically on both threads, rather than by throwing).
    const decodedLength = (await Native.fromConfig({}).decode(good)).length;
    const [bad, ok] = await decodeBloscBatch([
      { bytes: good.slice(), delta: { cols: decodedLength + 1, bits: 8 } },
      { bytes: good.slice(), delta: null },
    ]);
    expect(bad).toEqual({ error: expect.stringContaining('luxar_delta_v1') });
    expect(ok).toEqual({ data: await Native.fromConfig({}).decode(good) });
  });

  it.each([
    ['a rejecting backend', () => Promise.reject(new Error('worker died'))],
    ['a declining backend', () => null],
  ])('%s falls back to the main-thread decode, byte-identically', async (_label, decode) => {
    for (const name of FIXTURES.filter((n) => readMeta(n).delta !== null)) {
      setBloscDecodeBackend(null);
      const mainThread = await readThroughPipeline(name);
      setBloscDecodeBackend({ decode });
      expect(await readThroughPipeline(name)).toEqual(mainThread);
    }
  });

  it('does not fuse a delta that is not decoded immediately after blosc', () => {
    const meta = { dataType: 'uint16', shape: [16, 2] };
    const delta = { name: 'luxar_delta_v1', configuration: { cols: 2, bits: 16 } };
    const blosc = { name: 'blosc', configuration: {} };
    const little = { name: 'bytes', configuration: { endian: 'little' } };
    const big = { name: 'bytes', configuration: { endian: 'big' } };
    const other = { name: 'transpose', configuration: {} };
    expect(resolveFusedDelta({ ...meta, codecs: [delta, little, blosc] })).toEqual({
      cols: 2,
      bits: 16,
    });
    expect(resolveFusedDelta({ ...meta, codecs: [delta, blosc] })).toEqual({ cols: 2, bits: 16 });
    expect(resolveFusedDelta({ ...meta, codecs: [delta, big, blosc] })).toBeNull();
    expect(resolveFusedDelta({ ...meta, codecs: [delta, other, little, blosc] })).toBeNull();
    expect(resolveFusedDelta({ ...meta, codecs: [little, blosc] })).toBeNull();
    // An invalid delta config is left for the delta codec to reject.
    expect(
      resolveFusedDelta({
        ...meta,
        codecs: [{ ...delta, configuration: { cols: 3, bits: 16 } }, little, blosc],
      })
    ).toBeNull();
    expect(WorkerBloscCodec.fromConfig({}, { ...meta, codecs: [little, blosc] }).delta).toBeNull();
  });
});
