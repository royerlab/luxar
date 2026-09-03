// @vitest-environment jsdom
/**
 * Unit tests for core/app/init/environment-guards.ts (G2).
 *
 * Both guards are "fail-fast" assertions called from the front of
 * `LuxarApp.init()`. The contract is precise:
 *   - `assertBrowserEnvironment` throws when window or document is undefined.
 *   - `assertThreeRevision` throws when THREE.REVISION is lower than the
 *     minimum or is unparseable.
 *
 * Tests use `vi.stubGlobal` for the window/document branch and a
 * `vi.doMock`-based re-import for THREE.REVISION (which is a frozen
 * module export in v0.184+, so direct assignment is rejected by V8).
 */

import { describe, it, expect, afterEach, vi } from 'vitest';
import {
  assertBrowserEnvironment,
  assertThreeRevision,
} from '../../../../../core/app/init/environment-guards';

describe('assertBrowserEnvironment', () => {
  // jsdom env: window + document are both defined by default.
  it('returns silently when window and document are both defined', () => {
    expect(() => assertBrowserEnvironment()).not.toThrow();
  });

  it('throws a clear error when window is undefined', () => {
    const original = globalThis.window;
    delete (globalThis as { window?: Window }).window;
    try {
      expect(() => assertBrowserEnvironment()).toThrow(/browser environment/i);
      expect(() => assertBrowserEnvironment()).toThrow(/window and document must be defined/i);
    } finally {
      globalThis.window = original;
    }
  });

  it('throws a clear error when document is undefined', () => {
    const original = globalThis.document;
    delete (globalThis as { document?: Document }).document;
    try {
      expect(() => assertBrowserEnvironment()).toThrow(/browser environment/i);
    } finally {
      globalThis.document = original;
    }
  });
});

/**
 * Re-import environment-guards with a specific THREE.REVISION value
 * mocked in. `vi.doMock` (non-hoisted) lets us swap the THREE namespace
 * per-test without leaking into the other suites in this file.
 */
async function importGuardsWithRevision(
  revision: string | undefined
): Promise<typeof import('../../../../../core/app/init/environment-guards')> {
  vi.resetModules();
  vi.doMock('three', async () => {
    const actual = await vi.importActual<typeof import('three')>('three');
    return {
      ...actual,
      REVISION: revision as never,
    };
  });
  return import('../../../../../core/app/init/environment-guards');
}

describe('assertThreeRevision', () => {
  afterEach(() => {
    vi.doUnmock('three');
    vi.resetModules();
  });

  it('returns silently when REVISION >= default minimum (185)', async () => {
    const { assertThreeRevision: fn } = await importGuardsWithRevision('185');
    expect(() => fn()).not.toThrow();
  });

  it('returns silently when REVISION is well above minimum', async () => {
    const { assertThreeRevision: fn } = await importGuardsWithRevision('999');
    expect(() => fn()).not.toThrow();
  });

  it('throws when REVISION is below default minimum', async () => {
    const { assertThreeRevision: fn } = await importGuardsWithRevision('184');
    expect(() => fn()).toThrow(/three@>=0\.185\.0/);
    expect(() => fn()).toThrow(/found r184/);
  });

  it('throws when REVISION is far below minimum', async () => {
    const { assertThreeRevision: fn } = await importGuardsWithRevision('150');
    expect(() => fn()).toThrow(/three@>=0\.185\.0/);
  });

  it('honors a custom minRevision override', async () => {
    const { assertThreeRevision: fn } = await importGuardsWithRevision('190');
    // 180 is BELOW the default floor, so a passing call proves the override is
    // read rather than the default silently applying.
    expect(() => fn(180)).not.toThrow();
    expect(() => fn(190)).not.toThrow();
    expect(() => fn(191)).toThrow(/three@>=0\.191\.0/);
  });

  it('error message includes the actual revision so users can diagnose', async () => {
    const { assertThreeRevision: fn } = await importGuardsWithRevision('170');
    let captured = '';
    try {
      fn();
    } catch (e) {
      captured = (e as Error).message;
    }
    expect(captured).toMatch(/r170/);
    expect(captured).toMatch(/Update the three peer dependency/i);
  });

  it('treats a non-numeric REVISION as below-min (NaN guard)', async () => {
    const { assertThreeRevision: fn } = await importGuardsWithRevision('not-a-number');
    // parseInt yields NaN which Number.isFinite rejects → throw.
    expect(() => fn()).toThrow(/three@>=0\.185\.0/);
  });

  it('boundary: exactly the minimum revision is accepted (off-by-one guard)', async () => {
    const { assertThreeRevision: fn } = await importGuardsWithRevision('185');
    // `< minRevision` (not `<=`) so 185 must pass for minRevision=185.
    expect(() => fn(185)).not.toThrow();
  });

  it('default floor tracks the minor of the `three` peer range', async () => {
    // Nothing else couples these: the peer range is what npm enforces for an
    // embedder, the floor is what the viewer enforces at runtime, and a bump that
    // moves one and forgets the other leaves the guard admitting a revision the
    // package refuses to install. Read the manifest and compare.
    // `import.meta.url` is not a file: URL under the jsdom environment this file
    // runs in, so the manifest is located from the process cwd — which is the
    // package dir for every entry point the repo actually uses, and the repo root
    // if someone runs vitest with an explicit `--root`. Both are accepted rather
    // than leaving the test to fail on the caller's choice of cwd.
    const { readFileSync } = await import('node:fs');
    const { join } = await import('node:path');
    const candidates = [
      join(process.cwd(), 'package.json'),
      join(process.cwd(), 'packages', 'luxar-viewer', 'package.json'),
    ];
    let manifest: { name?: string; peerDependencies?: Record<string, string> } | undefined;
    for (const candidate of candidates) {
      try {
        const parsed = JSON.parse(readFileSync(candidate, 'utf8')) as typeof manifest;
        if (parsed?.name === '@luxar/viewer') {
          manifest = parsed;
          break;
        }
      } catch {
        // Not this one — try the next candidate.
      }
    }
    expect(manifest, `viewer package.json not found from cwd ${process.cwd()}`).toBeDefined();
    const range = manifest!.peerDependencies?.three;
    expect(range).toBeDefined();
    const peerMinor = Number(/^[~^]?0\.(\d+)\./.exec(range!)?.[1]);
    expect(Number.isFinite(peerMinor)).toBe(true);

    // This pins the default to EQUAL the peer minor, in both directions: the
    // revision below it must throw (so the default cannot be lower), and the
    // thrown message must name `peerMinor` (so it cannot be higher either).
    expect(() => assertThreeRevision()).not.toThrow();
    const { assertThreeRevision: fn } = await importGuardsWithRevision(String(peerMinor - 1));
    expect(() => fn()).toThrow(new RegExp(`three@>=0\\.${peerMinor}\\.0`));
  });

  it('current bundled THREE revision passes the default guard (sanity)', () => {
    // The real module imported at the top of this file uses the real
    // THREE. If our peer-dep declaration is honored, the bundled
    // revision must be >= 185.
    expect(() => assertThreeRevision()).not.toThrow();
  });
});
