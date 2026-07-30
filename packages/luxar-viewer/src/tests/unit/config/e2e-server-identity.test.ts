import { createServer as createHTTPServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { createServer as createViteServer } from 'vite';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  assertCheckoutServerIdentity,
  assertHTTPResource,
  checkoutIdentityPlugin,
  createE2EServerMetadata,
  ensureCheckoutIdentity,
  requireE2EServerMetadata,
} from '../../../../tools/e2e-server-identity';

let temporaryRoot: string;
let projectRoot: string;
let viewerRoot: string;

async function findFreePort(): Promise<number> {
  const reservation = createHTTPServer();
  await new Promise<void>((resolve) => reservation.listen(0, '127.0.0.1', resolve));
  const port = (reservation.address() as AddressInfo).port;
  await new Promise<void>((resolve, reject) =>
    reservation.close((error) => (error ? reject(error) : resolve()))
  );
  return port;
}

beforeEach(() => {
  temporaryRoot = mkdtempSync(path.join(tmpdir(), 'luxar-e2e-identity-'));
  projectRoot = path.join(temporaryRoot, 'checkout');
  viewerRoot = path.join(projectRoot, 'packages/luxar-viewer');
  mkdirSync(viewerRoot, { recursive: true });
});

afterEach(() => {
  rmSync(temporaryRoot, { recursive: true, force: true });
});

describe('E2E server identity', () => {
  it('creates a stable ignored marker unique to the checkout path', () => {
    const first = ensureCheckoutIdentity(projectRoot, viewerRoot);
    const second = ensureCheckoutIdentity(projectRoot, viewerRoot);

    expect(second).toEqual(first);
    expect(first.identity).toMatch(/^[a-f0-9]{24}$/);
    expect(readFileSync(first.markerFile, 'utf8')).toBe(first.markerBody);
    expect(first.viewerPath).toBe(`/.luxar-e2e-identities/${first.identity}.txt`);
    expect(first.dataPath).toBe(
      `/packages/luxar-viewer/.luxar-e2e-identities/${first.identity}.txt`
    );

    writeFileSync(first.markerFile, 'corrupt marker\n', 'utf8');
    ensureCheckoutIdentity(projectRoot, viewerRoot);
    expect(readFileSync(first.markerFile, 'utf8')).toBe(first.markerBody);

    const otherProject = path.join(temporaryRoot, 'other-checkout');
    const otherViewer = path.join(otherProject, 'packages/luxar-viewer');
    mkdirSync(otherViewer, { recursive: true });
    const other = ensureCheckoutIdentity(otherProject, otherViewer);
    expect(other.identity).not.toBe(first.identity);
  });

  it('rejects a viewer root outside the project root', () => {
    const unrelatedViewer = path.join(temporaryRoot, 'unrelated-viewer');
    mkdirSync(unrelatedViewer, { recursive: true });

    expect(() => ensureCheckoutIdentity(projectRoot, unrelatedViewer)).toThrow(
      /Viewer root must be inside the project root/
    );
  });

  it('canonicalizes symlink aliases to the same checkout identity', () => {
    const direct = ensureCheckoutIdentity(projectRoot, viewerRoot);
    const aliasRoot = path.join(temporaryRoot, 'checkout-alias');
    symlinkSync(projectRoot, aliasRoot, 'dir');

    const alias = ensureCheckoutIdentity(aliasRoot, path.join(aliasRoot, 'packages/luxar-viewer'));
    expect(alias.identity).toBe(direct.identity);
    expect(alias.projectRoot).toBe(direct.projectRoot);
    expect(alias.markerFile).toBe(direct.markerFile);
  });

  it('publishes only this checkout identity from the Vite middleware', async () => {
    const checkout = ensureCheckoutIdentity(projectRoot, viewerRoot);
    const port = await findFreePort();
    const vite = await createViteServer({
      configFile: false,
      logLevel: 'silent',
      root: viewerRoot,
      plugins: [checkoutIdentityPlugin(checkout)],
      server: { host: '127.0.0.1', port, strictPort: true },
    });

    try {
      await vite.listen();
      const address = vite.httpServer?.address() as AddressInfo;
      const baseURL = `http://127.0.0.1:${address.port}`;

      const matching = await fetch(new URL(checkout.viewerPath, baseURL));
      expect(matching.status).toBe(200);
      expect(await matching.text()).toBe(checkout.markerBody);

      const head = await fetch(new URL(checkout.viewerPath, baseURL), { method: 'HEAD' });
      expect(head.status).toBe(200);
      expect(await head.text()).toBe('');

      const foreign = await fetch(
        new URL('/.luxar-e2e-identities/000000000000000000000000.txt', baseURL)
      );
      expect(foreign.status).toBe(404);
    } finally {
      await vite.close();
    }
  });

  it('passes exact identity metadata to global setup', () => {
    const checkout = ensureCheckoutIdentity(projectRoot, viewerRoot);
    const metadata = createE2EServerMetadata(
      checkout,
      'http://localhost:5173',
      'http://localhost:9000'
    );

    expect(metadata.viewerIdentityURL).toBe(
      `http://localhost:5173/.luxar-e2e-identities/${checkout.identity}.txt`
    );
    expect(metadata.dataIdentityURL).toBe(
      `http://localhost:9000/packages/luxar-viewer/.luxar-e2e-identities/${checkout.identity}.txt`
    );
    expect(requireE2EServerMetadata({ luxarE2E: metadata })).toEqual(metadata);
    expect(() => requireE2EServerMetadata({})).toThrow(/missing metadata\.luxarE2E/);
    expect(() =>
      requireE2EServerMetadata({
        luxarE2E: {
          ...metadata,
          checkout: { ...metadata.checkout, markerBody: 42 },
        },
      })
    ).toThrow(/invalid server identity shape/);
  });

  it('requires both a successful status and the exact identity body', async () => {
    const checkout = ensureCheckoutIdentity(projectRoot, viewerRoot);
    let status = 200;
    const server = createHTTPServer((_request, response) => {
      response.writeHead(status, { 'Content-Type': 'text/plain' });
      response.end(checkout.markerBody);
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));

    try {
      const address = server.address() as AddressInfo;
      const url = `http://127.0.0.1:${address.port}${checkout.viewerPath}`;
      await expect(assertCheckoutServerIdentity('Viewer', url, checkout)).resolves.toBeUndefined();

      status = 404;
      await expect(assertCheckoutServerIdentity('Viewer', url, checkout)).rejects.toThrow(
        /HTTP 404/
      );
    } finally {
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve()))
      );
    }
  });

  it('rejects a catch-all server that returns 200 with the wrong body', async () => {
    const checkout = ensureCheckoutIdentity(projectRoot, viewerRoot);
    const server = createHTTPServer((_request, response) => {
      response.writeHead(200, { 'Content-Type': 'text/html' });
      response.end('<html>foreign checkout</html>');
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));

    try {
      const address = server.address() as AddressInfo;
      const url = `http://127.0.0.1:${address.port}${checkout.viewerPath}`;
      await expect(assertCheckoutServerIdentity('Viewer', url, checkout)).rejects.toThrow(
        /does not belong to this checkout/
      );
      await expect(assertCheckoutServerIdentity('Viewer', url, checkout)).rejects.toThrow(
        new RegExp(`lsof -nP -iTCP:${address.port}`)
      );
    } finally {
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve()))
      );
    }
  });

  it('checks dataset availability through HTTP rather than only on disk', async () => {
    const server = createHTTPServer((request, response) => {
      response.statusCode = request.url === '/available/' ? 200 : 404;
      response.end();
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));

    try {
      const address = server.address() as AddressInfo;
      const baseURL = `http://127.0.0.1:${address.port}`;
      await expect(assertHTTPResource('fixture', `${baseURL}/available/`)).resolves.toBeUndefined();
      await expect(assertHTTPResource('fixture', `${baseURL}/missing/`)).rejects.toThrow(
        /is not reachable over HTTP/
      );
    } finally {
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve()))
      );
    }
  });
});
