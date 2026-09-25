/**
 * Minimal static server for the render gate: one per build, serving the
 * build's `dist/` at `/` and the checkout's `datasets/` at `/datasets/`.
 * Same origin for viewer and data, so no CORS and no browser flags.
 *
 * @module scripts/render-gate/server
 */

import { createReadStream, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, sep } from 'node:path';

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

function safeJoin(root, urlPath) {
  const decoded = decodeURIComponent(urlPath);
  const full = normalize(join(root, decoded));
  return full === root || full.startsWith(root + sep) ? full : null;
}

function fileAt(path) {
  try {
    const st = statSync(path);
    if (st.isFile()) return { path, size: st.size };
    if (st.isDirectory()) {
      const index = join(path, 'index.html');
      const ist = statSync(index);
      if (ist.isFile()) return { path: index, size: ist.size };
    }
  } catch {
    /* missing */
  }
  return null;
}

/**
 * Start a server.
 *
 * @param {{ distDir: string, dataRoot: string, port: number }} opts
 * @returns {Promise<{ origin: string, close: () => Promise<void> }>}
 */
export function startServer({ distDir, dataRoot, port }) {
  const server = createServer((req, res) => {
    const url = new URL(req.url ?? '/', 'http://localhost');
    const isData = url.pathname.startsWith('/datasets/');
    const root = isData ? dataRoot : distDir;
    const rel = isData ? url.pathname.slice('/datasets'.length) : url.pathname;
    const target = safeJoin(root, rel);
    const file = target ? fileAt(target) : null;
    if (!file) {
      res.writeHead(404, { 'Access-Control-Allow-Origin': '*' });
      res.end();
      return;
    }
    res.writeHead(200, {
      'Content-Type': TYPES[extname(file.path)] ?? 'application/octet-stream',
      'Content-Length': file.size,
      'Cache-Control': 'no-store',
      'Access-Control-Allow-Origin': '*',
    });
    if (req.method === 'HEAD') {
      res.end();
      return;
    }
    createReadStream(file.path).pipe(res);
  });
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', () =>
      resolve({
        origin: `http://127.0.0.1:${port}`,
        close: () => new Promise((r) => server.close(() => r())),
      })
    );
  });
}
