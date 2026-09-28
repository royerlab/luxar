/**
 * Static server for the render gate: one per build, serving the build's
 * `dist/` at `/` and the checkout's `datasets/` at `/datasets/`. Same origin
 * for viewer and data, so no CORS and no browser flags.
 *
 * With no options it is the minimal server the exact and perf suites have
 * always used (plain HTTP/1.1, `Cache-Control: no-store`, `Range` ignored);
 * those responses are unchanged byte for byte. Options turn it into a
 * hosted-link simulator for the `hosted` suites:
 *
 * - `range`: a single `Range: bytes=a-b` / `bytes=a-` / suffix `bytes=-n`
 *   answers 206 with `Content-Range` (416 when unsatisfiable), every file
 *   response advertises `Accept-Ranges: bytes` (a `.zarr.zip` store needs it);
 * - `latencyMs`: delay before the response headers of every request;
 * - `bytesPerSec`: ONE token bucket shared by every response of this server
 *   (a link, not a per-connection cap);
 * - `h2`: HTTP/2 over TLS (`allowHTTP1` too) with a self-signed certificate
 *   generated once into `certDir` (the browser needs
 *   `--ignore-certificate-errors`), removing HTTP/1.1's 6-connection limit;
 * - `etag`: strong `ETag` from size + mtime, `If-None-Match` answers 304;
 * - `cacheControl`: the `Cache-Control` value (default `no-store`).
 *
 * Every server keeps a request log, exposed at `GET /__gate/requests` (JSON)
 * and cleared by `POST /__gate/reset` (or `reset()` in process); `stats()`
 * summarises it. The log never changes a response.
 *
 * @module scripts/render-gate/server
 */

import { execFileSync } from 'node:child_process';
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import http2 from 'node:http2';
import { dirname, extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.woff2': 'font/woff2',
};

const here = dirname(fileURLToPath(import.meta.url));
/** `<repo>/delme/gate-certs`: this file lives at packages/luxar-viewer/scripts/render-gate. */
export const DEFAULT_CERT_DIR = resolve(here, '../../../..', 'delme', 'gate-certs');

function safeJoin(root, urlPath) {
  const decoded = decodeURIComponent(urlPath);
  const full = normalize(join(root, decoded));
  return full === root || full.startsWith(root + sep) ? full : null;
}

function fileAt(path) {
  try {
    const st = statSync(path);
    if (st.isFile()) return { path, size: st.size, mtime: st.mtimeMs };
    if (st.isDirectory()) {
      const index = join(path, 'index.html');
      const ist = statSync(index);
      if (ist.isFile()) return { path: index, size: ist.size, mtime: ist.mtimeMs };
    }
  } catch {
    /* missing */
  }
  return null;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Parse a single-range `Range` header against a file size.
 *
 * @param {string|undefined} header The request's `Range` header.
 * @param {number} size File size in bytes.
 * @returns {null | { start: number, end: number } | 'unsatisfiable'} `null`
 *   when there is no (or no parseable single) range: serve the whole file.
 */
export function parseRange(header, size) {
  const m = /^bytes=(\d*)-(\d*)$/.exec((header ?? '').trim());
  if (!m || (m[1] === '' && m[2] === '')) return null;
  if (m[1] === '') {
    const n = Number(m[2]);
    if (n === 0 || size === 0) return 'unsatisfiable';
    return { start: Math.max(0, size - n), end: size - 1 };
  }
  const start = Number(m[1]);
  const end = m[2] === '' ? size - 1 : Math.min(size - 1, Number(m[2]));
  if (start >= size || end < start) return 'unsatisfiable';
  return { start, end };
}

/**
 * A self-signed certificate for 127.0.0.1, generated once with openssl and
 * reused across runs.
 *
 * @param {string} certDir Directory holding `key.pem` / `cert.pem`.
 * @returns {{ key: Buffer, cert: Buffer }}
 */
export function ensureCert(certDir = DEFAULT_CERT_DIR) {
  const key = join(certDir, 'key.pem');
  const cert = join(certDir, 'cert.pem');
  if (!existsSync(key) || !existsSync(cert)) {
    mkdirSync(certDir, { recursive: true });
    execFileSync(
      'openssl',
      [
        'req',
        '-x509',
        '-nodes',
        '-newkey',
        'rsa:2048',
        '-subj',
        '/CN=127.0.0.1',
        '-days',
        '3650',
        '-keyout',
        key,
        '-out',
        cert,
      ],
      { stdio: 'ignore' }
    );
  }
  return { key: readFileSync(key), cert: readFileSync(cert) };
}

/** A shared token bucket: `take(n)` resolves when `n` bytes may be sent. */
function tokenBucket(bytesPerSec) {
  let nextFree = 0;
  return {
    async take(n) {
      if (!bytesPerSec) return;
      const now = performance.now();
      const start = Math.max(now, nextFree);
      nextFree = start + (n / bytesPerSec) * 1000;
      const wait = nextFree - now;
      if (wait > 1) await sleep(wait);
    },
    reset() {
      nextFree = 0;
    },
  };
}

function etagMatches(header, tag) {
  if (!header) return false;
  if (header.trim() === '*') return true;
  return header.split(',').some((t) => t.trim().replace(/^W\//, '') === tag);
}

/**
 * Start a server.
 *
 * @param {{ distDir: string, dataRoot: string, port: number, range?: boolean,
 *   latencyMs?: number, bytesPerSec?: number, h2?: boolean, etag?: boolean,
 *   cacheControl?: string, certDir?: string }} opts
 * @returns {Promise<{ origin: string, log: object[], now: () => number,
 *   stats: () => { requests: number, bytes: number, maxInflight: number },
 *   reset: () => void, close: () => Promise<void> }>}
 */
export function startServer({
  distDir,
  dataRoot,
  port,
  range = false,
  latencyMs = 0,
  bytesPerSec = 0,
  h2 = false,
  etag = false,
  cacheControl = 'no-store',
  certDir = DEFAULT_CERT_DIR,
}) {
  const plain = !range && !latencyMs && !bytesPerSec && !h2 && !etag;
  const t0 = performance.now();
  const now = () => performance.now() - t0;
  const log = [];
  let inflight = 0;
  let maxInflight = 0;
  const bucket = tokenBucket(bytesPerSec);
  const reset = () => {
    log.length = 0;
    maxInflight = inflight;
    bucket.reset();
  };
  const stats = () => ({
    requests: log.length,
    bytes: log.reduce((s, e) => s + e.bytes, 0),
    maxInflight,
  });

  function control(req, res, pathname) {
    if (pathname === '/__gate/requests') {
      const body = JSON.stringify(log);
      res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
      res.end(body);
      return true;
    }
    if (pathname === '/__gate/reset' && req.method === 'POST') {
      reset();
      res.writeHead(204);
      res.end();
      return true;
    }
    return false;
  }

  /** The unchanged minimal response (no options). */
  function servePlain(req, res, file, entry) {
    res.writeHead(200, {
      'Content-Type': TYPES[extname(file.path)] ?? 'application/octet-stream',
      'Content-Length': file.size,
      'Cache-Control': 'no-store',
      'Access-Control-Allow-Origin': '*',
    });
    entry.status = 200;
    if (req.method === 'HEAD') {
      res.end();
      return;
    }
    entry.bytes = file.size;
    createReadStream(file.path).pipe(res);
  }

  async function serveHosted(req, res, file, entry) {
    const tag = `"${file.size.toString(16)}-${Math.floor(file.mtime).toString(16)}"`;
    const base = { 'Cache-Control': cacheControl, 'Access-Control-Allow-Origin': '*' };
    if (etag) base.ETag = tag;
    if (etag && etagMatches(req.headers['if-none-match'], tag)) {
      entry.status = 304;
      res.writeHead(304, base);
      res.end();
      return;
    }
    const r = range ? parseRange(req.headers.range, file.size) : null;
    if (r === 'unsatisfiable') {
      entry.status = 416;
      res.writeHead(416, { ...base, 'Content-Range': `bytes */${file.size}` });
      res.end();
      return;
    }
    const start = r ? r.start : 0;
    const end = r ? r.end : file.size - 1;
    const headers = {
      ...base,
      'Content-Type': TYPES[extname(file.path)] ?? 'application/octet-stream',
      'Content-Length': Math.max(0, end - start + 1),
    };
    if (range) headers['Accept-Ranges'] = 'bytes';
    if (r) headers['Content-Range'] = `bytes ${start}-${end}/${file.size}`;
    entry.status = r ? 206 : 200;
    res.writeHead(entry.status, headers);
    if (req.method === 'HEAD' || file.size === 0) {
      res.end();
      return;
    }
    const rs = createReadStream(file.path, { start, end, highWaterMark: 16384 });
    for await (const chunk of rs) {
      await bucket.take(chunk.length);
      if (res.destroyed) break;
      entry.bytes += chunk.length;
      if (!res.write(chunk)) await new Promise((ok) => res.once('drain', ok));
    }
    res.end();
  }

  async function handler(req, res) {
    const url = new URL(req.url ?? '/', 'http://localhost');
    if (url.pathname.startsWith('/__gate/') && control(req, res, url.pathname)) return;
    inflight++;
    maxInflight = Math.max(maxInflight, inflight);
    const entry = {
      t: Date.now(),
      url: url.pathname + url.search,
      range: req.headers.range ?? null,
      status: 0,
      bytes: 0,
      startMs: now(),
      endMs: null,
      inflightAtStart: inflight,
    };
    log.push(entry);
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      inflight--;
      entry.endMs = now();
    };
    res.once('finish', finish);
    res.once('close', finish);
    try {
      const isData = url.pathname.startsWith('/datasets/');
      const root = isData ? dataRoot : distDir;
      const rel = isData ? url.pathname.slice('/datasets'.length) : url.pathname;
      const target = safeJoin(root, rel);
      const file = target ? fileAt(target) : null;
      if (latencyMs) await sleep(latencyMs);
      if (!file) {
        entry.status = 404;
        res.writeHead(404, { 'Access-Control-Allow-Origin': '*' });
        res.end();
        return;
      }
      if (plain) servePlain(req, res, file, entry);
      else await serveHosted(req, res, file, entry);
    } catch (e) {
      entry.error = String(e?.message ?? e);
      if (!res.headersSent) res.writeHead(500);
      res.end();
    }
  }

  const sessions = new Set();
  const server = h2
    ? http2.createSecureServer({ ...ensureCert(certDir), allowHTTP1: true }, handler)
    : createServer(handler);
  if (h2) {
    server.on('session', (s) => {
      sessions.add(s);
      s.once('close', () => sessions.delete(s));
    });
  }
  return new Promise((ok, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', () =>
      ok({
        origin: `${h2 ? 'https' : 'http'}://127.0.0.1:${port}`,
        log,
        now,
        stats,
        reset,
        close: () =>
          new Promise((r) => {
            server.close(() => r());
            for (const s of sessions) s.destroy();
            server.closeAllConnections?.();
          }),
      })
    );
  });
}
