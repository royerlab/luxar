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

  it('returns silently when REVISION >= default minimum (184)', async () => {
    const { assertThreeRevision: fn } = await importGuardsWithRevision('184');
    expect(() => fn()).not.toThrow();
  });

  it('returns silently when REVISION is well above minimum', async () => {
    const { assertThreeRevision: fn } = await importGuardsWithRevision('999');
    expect(() => fn()).not.toThrow();
  });

  it('throws when REVISION is below default minimum', async () => {
    const { assertThreeRevision: fn } = await importGuardsWithRevision('183');
    expect(() => fn()).toThrow(/three@>=0\.184\.0/);
    expect(() => fn()).toThrow(/found r183/);
  });

  it('throws when REVISION is far below minimum', async () => {
    const { assertThreeRevision: fn } = await importGuardsWithRevision('150');
    expect(() => fn()).toThrow(/three@>=0\.184\.0/);
  });

  it('honors a custom minRevision override', async () => {
    const { assertThreeRevision: fn } = await importGuardsWithRevision('190');
    expect(() => fn(184)).not.toThrow();
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
    expect(() => fn()).toThrow(/three@>=0\.184\.0/);
  });

  it('boundary: exactly the minimum revision is accepted (off-by-one guard)', async () => {
    const { assertThreeRevision: fn } = await importGuardsWithRevision('184');
    // `< minRevision` (not `<=`) so 184 must pass for minRevision=184.
    expect(() => fn(184)).not.toThrow();
  });

  it('current bundled THREE revision passes the default guard (sanity)', () => {
    // The real module imported at the top of this file uses the real
    // THREE. If our peer-dep declaration is honored, the bundled
    // revision must be >= 184.
    expect(() => assertThreeRevision()).not.toThrow();
  });
});
