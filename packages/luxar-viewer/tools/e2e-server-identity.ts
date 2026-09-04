import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import * as path from 'node:path';
import type { Plugin } from 'vite';

const IDENTITY_DIRECTORY = '.luxar-e2e-identities';
const IDENTITY_PREFIX = `/${IDENTITY_DIRECTORY}/`;
const IDENTITY_LENGTH = 24;

/** Checkout-specific server identity shared by Vite, Playwright, and preflight. */
export interface CheckoutIdentity {
  identity: string;
  projectRoot: string;
  viewerRoot: string;
  markerBody: string;
  markerFile: string;
  viewerPath: string;
  dataPath: string;
}

/** Metadata passed from a Playwright config to its global setup. */
export interface E2EServerMetadata {
  checkout: CheckoutIdentity;
  viewerIdentityURL: string;
  dataIdentityURL: string;
  dataBaseURL: string;
}

/** Canonicalize an existing path so symlink aliases identify the same checkout. */
function canonicalPath(value: string): string {
  return realpathSync(path.resolve(value));
}

/** Convert a path relative to the repository root into a URL path. */
function relativeURLPath(projectRoot: string, file: string): string {
  const relative = path.relative(projectRoot, file);
  if (relative === '' || relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`E2E identity marker must be inside the project root: ${file}`);
  }
  return `/${relative.split(path.sep).map(encodeURIComponent).join('/')}`;
}

/**
 * Create the deterministic marker used to distinguish sibling clones/worktrees.
 *
 * The marker is ignored by git and persists for the checkout's lifetime. Any
 * static server rooted at the repository can therefore prove its identity.
 */
export function ensureCheckoutIdentity(projectRoot: string, viewerRoot: string): CheckoutIdentity {
  const canonicalProjectRoot = canonicalPath(projectRoot);
  const canonicalViewerRoot = canonicalPath(viewerRoot);
  const viewerRelative = path.relative(canonicalProjectRoot, canonicalViewerRoot);
  if (viewerRelative === '' || viewerRelative.startsWith('..') || path.isAbsolute(viewerRelative)) {
    throw new Error(`Viewer root must be inside the project root: ${canonicalViewerRoot}`);
  }

  const identity = createHash('sha256')
    .update(canonicalProjectRoot)
    .digest('hex')
    .slice(0, IDENTITY_LENGTH);
  const markerBody = `${identity}\n`;
  const markerDirectory = path.join(canonicalViewerRoot, IDENTITY_DIRECTORY);
  const markerFile = path.join(markerDirectory, `${identity}.txt`);

  mkdirSync(markerDirectory, { recursive: true });
  if (!existsSync(markerFile) || readFileSync(markerFile, 'utf8') !== markerBody) {
    writeFileSync(markerFile, markerBody, 'utf8');
  }

  return {
    identity,
    projectRoot: canonicalProjectRoot,
    viewerRoot: canonicalViewerRoot,
    markerBody,
    markerFile,
    viewerPath: `${IDENTITY_PREFIX}${identity}.txt`,
    dataPath: relativeURLPath(canonicalProjectRoot, markerFile),
  };
}

/** Build serializable metadata consumed by the E2E global setup. */
export function createE2EServerMetadata(
  checkout: CheckoutIdentity,
  viewerBaseURL: string,
  dataBaseURL: string
): E2EServerMetadata {
  return {
    checkout,
    viewerIdentityURL: new URL(checkout.viewerPath, viewerBaseURL).toString(),
    dataIdentityURL: new URL(checkout.dataPath, dataBaseURL).toString(),
    dataBaseURL,
  };
}

/** Read and validate Luxar's custom metadata from a Playwright FullConfig. */
export function requireE2EServerMetadata(metadata: Record<string, unknown>): E2EServerMetadata {
  const candidate = metadata.luxarE2E;
  if (typeof candidate !== 'object' || candidate === null) {
    throw new Error('Playwright config is missing metadata.luxarE2E server identity settings');
  }

  const value = candidate as Partial<E2EServerMetadata>;
  const checkout = value.checkout as Partial<CheckoutIdentity> | undefined;
  if (
    typeof checkout?.identity !== 'string' ||
    typeof checkout.projectRoot !== 'string' ||
    typeof checkout.markerBody !== 'string' ||
    typeof value.viewerIdentityURL !== 'string' ||
    typeof value.dataIdentityURL !== 'string' ||
    typeof value.dataBaseURL !== 'string'
  ) {
    throw new Error('Playwright metadata.luxarE2E has an invalid server identity shape');
  }

  return candidate as E2EServerMetadata;
}

/**
 * Vite endpoint that returns 404 for another checkout's identity path.
 *
 * The exact marker body is checked again by global setup. That second check
 * also rejects unrelated catch-all servers that return index.html with 200.
 */
export function checkoutIdentityPlugin(checkout: CheckoutIdentity): Plugin {
  // One middleware for both the dev server and `vite preview`, so a perf run
  // against the production bundle (LUXAR_PERF_PREVIEW=1) passes the same
  // checkout-identity check as a dev-server run.
  const middleware = (
    request: { url?: string; method?: string },
    response: {
      setHeader(name: string, value: string): void;
      statusCode: number;
      end(body?: string): void;
    },
    next: () => void
  ): void => {
    const pathname = new URL(request.url ?? '/', 'http://localhost').pathname;
    if (!pathname.startsWith(IDENTITY_PREFIX)) {
      next();
      return;
    }

    response.setHeader('Cache-Control', 'no-store');
    response.setHeader('Content-Type', 'text/plain; charset=utf-8');
    if (pathname !== checkout.viewerPath) {
      response.statusCode = 404;
      response.end('Unknown Luxar E2E checkout identity\n');
      return;
    }

    response.statusCode = 200;
    if (request.method === 'HEAD') {
      response.end();
    } else {
      response.end(checkout.markerBody);
    }
  };
  return {
    name: 'luxar-e2e-checkout-identity',
    configurePreviewServer(server) {
      server.middlewares.use(middleware);
    },
    configureServer(server) {
      server.middlewares.use(middleware);
    },
  };
}

/** Fail with checkout/port guidance if a server returns the wrong marker. */
export async function assertCheckoutServerIdentity(
  label: string,
  url: string,
  checkout: CheckoutIdentity
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(url, {
      cache: 'no-store',
      signal: AbortSignal.timeout(5000),
    });
  } catch (error) {
    throw new Error(serverMismatchMessage(label, url, checkout, `unreachable: ${String(error)}`), {
      cause: error,
    });
  }

  const body = await response.text();
  if (!response.ok || body !== checkout.markerBody) {
    const received = response.ok
      ? `HTTP ${response.status} with identity body ${JSON.stringify(body.slice(0, 120))}`
      : `HTTP ${response.status} ${response.statusText}`;
    throw new Error(serverMismatchMessage(label, url, checkout, received));
  }
}

/** Fail if a locally present dataset is not reachable from the serving root. */
export async function assertHTTPResource(label: string, url: string): Promise<void> {
  let response: Response;
  try {
    response = await fetch(url, {
      method: 'HEAD',
      cache: 'no-store',
      signal: AbortSignal.timeout(5000),
    });
  } catch (error) {
    throw new Error(`${label} is not reachable over HTTP: ${url} (${String(error)})`, {
      cause: error,
    });
  }

  if (!response.ok) {
    throw new Error(
      `${label} is not reachable over HTTP: ${url} -> ` +
        `HTTP ${response.status} ${response.statusText}`
    );
  }
}

function serverMismatchMessage(
  label: string,
  url: string,
  checkout: CheckoutIdentity,
  received: string
): string {
  const port = new URL(url).port;
  return (
    `${label} server does not belong to this checkout: ${checkout.projectRoot}\n` +
    `Identity probe: ${url}\n` +
    `Received: ${received}\n` +
    `Another checkout or stale process may own port ${port}. Inspect it with:\n` +
    `  lsof -nP -iTCP:${port} -sTCP:LISTEN\n` +
    'Stop that process (after coordinating with its owner), then rerun Playwright.'
  );
}
