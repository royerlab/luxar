import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';

import { parseRange, startServer } from './server.mjs';

let root;
let dist;
let data;
const BODY = Buffer.from('0123456789abcdefghij'); // 20 bytes

beforeAll(() => {
  root = mkdtempSync(join(tmpdir(), 'gate-server-'));
  dist = join(root, 'dist');
  data = join(root, 'datasets');
  mkdirSync(dist);
  mkdirSync(join(data, 'gate'), { recursive: true });
  writeFileSync(join(dist, 'index.html'), '<html></html>');
  writeFileSync(join(data, 'gate', 'blob.bin'), BODY);
});

afterAll(() => rmSync(root, { recursive: true, force: true }));

async function withServer(options, fn) {
  const server = await startServer({ distDir: dist, dataRoot: data, port: 0, ...options });
  try {
    await fn(server);
  } finally {
    await server.close();
  }
}

describe('parseRange', () => {
  it('parses closed, open and suffix ranges and rejects unsatisfiable ones', () => {
    expect(parseRange('bytes=2-5', 20)).toEqual({ start: 2, end: 5 });
    expect(parseRange('bytes=15-', 20)).toEqual({ start: 15, end: 19 });
    expect(parseRange('bytes=10-99', 20)).toEqual({ start: 10, end: 19 });
    expect(parseRange('bytes=-4', 20)).toEqual({ start: 16, end: 19 });
    expect(parseRange('bytes=-400', 20)).toEqual({ start: 0, end: 19 });
    expect(parseRange('bytes=20-', 20)).toBe('unsatisfiable');
    expect(parseRange('bytes=5-2', 20)).toBe('unsatisfiable');
    expect(parseRange('bytes=-0', 20)).toBe('unsatisfiable');
    expect(parseRange(undefined, 20)).toBeNull();
    expect(parseRange('bytes=1-2,4-5', 20)).toBeNull();
  });
});

describe('startServer', () => {
  it('default mode ignores Range and sends the unchanged minimal headers', async () => {
    await withServer({}, async (s) => {
      const r = await fetch(`${s.origin}/datasets/gate/blob.bin`, {
        headers: { Range: 'bytes=0-3' },
      });
      expect(r.status).toBe(200);
      expect(Buffer.from(await r.arrayBuffer())).toEqual(BODY);
      expect(r.headers.get('cache-control')).toBe('no-store');
      expect(r.headers.get('accept-ranges')).toBeNull();
      expect(r.headers.get('etag')).toBeNull();
    });
  });

  it('answers a single range with 206 and Content-Range', async () => {
    await withServer({ range: true }, async (s) => {
      const r = await fetch(`${s.origin}/datasets/gate/blob.bin`, {
        headers: { Range: 'bytes=2-5' },
      });
      expect(r.status).toBe(206);
      expect(r.headers.get('content-range')).toBe('bytes 2-5/20');
      expect(r.headers.get('accept-ranges')).toBe('bytes');
      expect(await r.text()).toBe('2345');
    });
  });

  it('answers a suffix range with the file tail', async () => {
    await withServer({ range: true }, async (s) => {
      const r = await fetch(`${s.origin}/datasets/gate/blob.bin`, {
        headers: { Range: 'bytes=-3' },
      });
      expect(r.status).toBe(206);
      expect(r.headers.get('content-range')).toBe('bytes 17-19/20');
      expect(await r.text()).toBe('hij');
    });
  });

  it('answers an unsatisfiable range with 416', async () => {
    await withServer({ range: true }, async (s) => {
      const r = await fetch(`${s.origin}/datasets/gate/blob.bin`, {
        headers: { Range: 'bytes=50-60' },
      });
      expect(r.status).toBe(416);
      expect(r.headers.get('content-range')).toBe('bytes */20');
    });
  });

  it('sends an ETag and answers a matching If-None-Match with 304', async () => {
    await withServer({ etag: true, cacheControl: 'max-age=300' }, async (s) => {
      const url = `${s.origin}/datasets/gate/blob.bin`;
      const first = await fetch(url);
      const tag = first.headers.get('etag');
      expect(tag).toMatch(/^"[0-9a-f]+-[0-9a-f]+"$/);
      expect(first.headers.get('cache-control')).toBe('max-age=300');
      await first.arrayBuffer();
      const again = await fetch(url, { headers: { 'If-None-Match': tag } });
      expect(again.status).toBe(304);
      const weak = await fetch(url, { headers: { 'If-None-Match': `W/${tag}` } });
      expect(weak.status).toBe(304);
      const stale = await fetch(url, { headers: { 'If-None-Match': '"nope"' } });
      expect(stale.status).toBe(200);
      await stale.arrayBuffer();
    });
  });

  it('delays every response by latencyMs', async () => {
    await withServer({ latencyMs: 150 }, async (s) => {
      const t0 = performance.now();
      const r = await fetch(`${s.origin}/datasets/gate/blob.bin`);
      await r.arrayBuffer();
      expect(performance.now() - t0).toBeGreaterThanOrEqual(140);
    });
  });

  it('logs requests, exposes the log over HTTP and resets it', async () => {
    await withServer({ range: true }, async (s) => {
      await (
        await fetch(`${s.origin}/datasets/gate/blob.bin`, { headers: { Range: 'bytes=0-9' } })
      ).arrayBuffer();
      // Client body completion can precede the server's response finish event.
      await vi.waitFor(() => expect(s.log[0].endMs).not.toBeNull(), { timeout: 5000 });
      await (await fetch(`${s.origin}/datasets/gate/missing.bin`)).arrayBuffer();
      await vi.waitFor(() => expect(s.log[1].endMs).not.toBeNull(), { timeout: 5000 });
      const log = await (await fetch(`${s.origin}/__gate/requests`)).json();
      expect(log).toHaveLength(2);
      expect(log[0]).toMatchObject({
        url: '/datasets/gate/blob.bin',
        range: 'bytes=0-9',
        status: 206,
        bytes: 10,
        inflightAtStart: 1,
      });
      expect(log[0].endMs).toBeGreaterThanOrEqual(log[0].startMs);
      expect(log[1]).toMatchObject({ status: 404, bytes: 0 });
      expect(s.stats()).toEqual({ requests: 2, bytes: 10, maxInflight: 1 });
      const reset = await fetch(`${s.origin}/__gate/reset`, { method: 'POST' });
      expect(reset.status).toBe(204);
      expect(await (await fetch(`${s.origin}/__gate/requests`)).json()).toEqual([]);
      expect(s.stats().requests).toBe(0);
    });
  });

  it('throttles through the shared token bucket', async () => {
    await withServer({ bytesPerSec: 100 }, async (s) => {
      const t0 = performance.now();
      await (await fetch(`${s.origin}/datasets/gate/blob.bin`)).arrayBuffer();
      // 20 bytes at 100 B/s: ~200 ms.
      expect(performance.now() - t0).toBeGreaterThanOrEqual(150);
    });
  });
});
