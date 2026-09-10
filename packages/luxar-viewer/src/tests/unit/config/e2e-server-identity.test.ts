import { createServer as createHTTPServer } from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import type { Connect } from 'vite';
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

/** Distinctive body served when the identity middleware calls `next()`. */
const FALL_THROUGH_BODY = 'fell through to the next middleware\n';

/**
 * Per-request budget, matching the production helpers in `tools/e2e-server-identity.ts`
 * (`AbortSignal.timeout(5000)`).
 *
 * A hung request therefore fails with a named `TimeoutError` well inside vitest's 15s
 * per-test budget, rather than surfacing as an anonymous 15s test timeout: the requests
 * are sequential and the first abort fails the test, so the budgets never accumulate in
 * practice.
 */
const FETCH_TIMEOUT_MS = 5000;

/** What the identity middleware did when driven directly, without a socket. */
interface DirectResponse {
  statusCode: number | undefined;
  /** One entry per `end()` call, holding that call's own argument list. */
  endCalls: unknown[][];
}

/**
 * Drive the identity middleware with request/response stubs instead of over HTTP.
 *
 * Node's http server discards a HEAD response's body no matter what the handler
 * wrote, so `fetch` cannot observe the plugin's HEAD branch at all. Stubs keep the
 * body observable. Only what `checkoutIdentityPlugin` touches is stubbed: the
 * request's `method` and `url`, and the response's `setHeader`, `statusCode`, `end`.
 * `statusCode` starts `undefined` instead of Node's real `200` default on purpose,
 * so the assertions also pin that the plugin sets the status explicitly.
 */
function driveMiddleware(
  middleware: Connect.NextHandleFunction,
  method: string,
  url: string
): DirectResponse {
  const endCalls: unknown[][] = [];
  const response = {
    statusCode: undefined as number | undefined,
    // The plugin sets both headers before it branches; the fetch assertions cover
    // them, so this stub only has to exist.
    setHeader(): void {},
    end(...args: unknown[]): void {
      endCalls.push(args);
    },
  };
  // The middleware reads only the members stubbed above, so two narrow casts stand
  // in for Node's full IncomingMessage / ServerResponse contracts.
  middleware(
    { method, url } as unknown as IncomingMessage,
    response as unknown as ServerResponse,
    () => {
      expect.fail(`${method} ${url} must not fall through to the next middleware`);
    }
  );
  return { statusCode: response.statusCode, endCalls };
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

  // The plugin's middleware is exercised on a plain `node:http` server, not a
  // real Vite dev server: loading Vite plus its optimizer/watcher startup is
  // load-sensitive and can overrun the 15s per-test budget under coverage and
  // whole-suite contention. Real Vite is exercised by Playwright, including the
  // mobile-only PR job, and only for the matching-path GET
  // that both `playwright.config.ts`'s `webServer[0].url` readiness probe and
  // `src/tests/e2e/global-setup.ts` request; HEAD, the foreign-identity 404, and
  // the pass-through are covered here alone.
  it('publishes only this checkout identity, and passes everything else through', async () => {
    const checkout = ensureCheckoutIdentity(projectRoot, viewerRoot);

    const plugin = checkoutIdentityPlugin(checkout);
    const configureServer = plugin.configureServer;
    const hook = typeof configureServer === 'function' ? configureServer : configureServer?.handler;
    if (typeof hook !== 'function') {
      expect.fail('checkoutIdentityPlugin must expose a configureServer hook');
    }

    let registered: Connect.NextHandleFunction | undefined;
    type MiddlewareStandIn = { use(middleware: Connect.NextHandleFunction): MiddlewareStandIn };
    const middlewares: MiddlewareStandIn = {
      use(middleware) {
        registered = middleware;
        return middlewares; // `Connect.Server.use()` returns the server for chaining.
      },
    };
    const middlewareHost = { middlewares };
    // The hook only reaches for `server.middlewares.use`, so the stand-in above
    // is sufficient; one cast keeps Vite's full `ViteDevServer` + plugin-`this`
    // contract out of a test that does not need either.
    await (hook as (server: unknown) => unknown)(middlewareHost);
    // Registration must happen during the `configureServer` call itself: Vite
    // installs a returned post-hook's middlewares *after* its static and
    // html-fallback handlers, and for the matching path `serveStaticMiddleware`
    // serves the marker file straight off disk (sirv runs in `dev: true` mode, so
    // dotfile paths are not excluded), so a post-hook middleware never runs for it.
    expect(registered).toBeTypeOf('function');
    const identityMiddleware = registered as Connect.NextHandleFunction;

    // HEAD is asserted through the stubs because a real socket cannot show it: the
    // GET below is the control proving those same stubs do observe a body chunk.
    const directHead = driveMiddleware(identityMiddleware, 'HEAD', checkout.viewerPath);
    expect(directHead.statusCode).toBe(200);
    expect(directHead.endCalls).toHaveLength(1);
    expect(directHead.endCalls[0][0]).toBeUndefined();
    const directGet = driveMiddleware(identityMiddleware, 'GET', checkout.viewerPath);
    expect(directGet.statusCode).toBe(200);
    expect(directGet.endCalls).toHaveLength(1);
    expect(directGet.endCalls[0][0]).toBe(checkout.markerBody);

    const server = createHTTPServer((request, response) => {
      identityMiddleware(request, response, () => {
        response.writeHead(418, { 'Content-Type': 'text/plain' });
        response.end(FALL_THROUGH_BODY);
      });
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));

    try {
      const address = server.address() as AddressInfo;
      const baseURL = `http://127.0.0.1:${address.port}`;

      const matching = await fetch(new URL(checkout.viewerPath, baseURL), {
        signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      });
      expect(matching.status).toBe(200);
      expect(matching.headers.get('cache-control')).toBe('no-store');
      expect(matching.headers.get('content-type')).toBe('text/plain; charset=utf-8');
      expect(await matching.text()).toBe(checkout.markerBody);

      const head = await fetch(new URL(checkout.viewerPath, baseURL), {
        method: 'HEAD',
        signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      });
      // The status travels over a real socket; the body does not (Node suppresses it
      // for HEAD), so the empty body is asserted through the middleware stub above.
      expect(head.status).toBe(200);

      // The plugin writes both headers before it branches, so the 404 carries them too.
      const foreign = await fetch(
        new URL('/.luxar-e2e-identities/000000000000000000000000.txt', baseURL),
        { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) }
      );
      expect(foreign.status, "another checkout's identity path must 404").toBe(404);
      expect(foreign.headers.get('cache-control')).toBe('no-store');
      expect(foreign.headers.get('content-type')).toBe('text/plain; charset=utf-8');
      expect(await foreign.text()).toBe('Unknown Luxar E2E checkout identity\n');

      // Anything outside the identity prefix must reach the next middleware.
      const unrelated = await fetch(new URL('/index.html', baseURL), {
        signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      });
      expect(unrelated.status, 'a non-identity path must fall through untouched').toBe(418);
      expect(await unrelated.text()).toBe(FALL_THROUGH_BODY);
      // The plugin must not leak its own headers onto unrelated responses.
      expect(unrelated.headers.get('cache-control')).toBeNull();
    } finally {
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve()))
      );
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
