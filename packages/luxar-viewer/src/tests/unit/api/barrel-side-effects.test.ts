// @vitest-environment jsdom
/**
 * Asserts that importing the public package barrel (`src/index.ts`) is
 * side-effect-free.
 *
 * This is the contract that makes Luxar safely embeddable: a third-party
 * page doing `import { LuxarApp } from '@luxar/viewer'` must NOT find its
 * console silently monkey-patched, its `:root` CSS variables overwritten,
 * or its `<body>` mutated by a singleton waking up at module-load.
 *
 * If you find yourself relaxing one of these assertions, instead push the
 * side effect down into LuxarApp.init() (gated by an option) or expose it
 * as an explicit factory the consumer must call.
 */

import { describe, expect, it, beforeAll, vi } from 'vitest';
import { existsSync, readFileSync } from 'fs';
import { join } from 'path';

/**
 * The public value-export list shared with `scripts/check-lib-exports.mjs`.
 * Read through `fs` rather than a JSON import so the test's view of the file
 * is the file itself, not a module-cache copy. `import.meta.url` is not a
 * `file:` URL under the jsdom environment this file runs in, so the path is
 * resolved from the process cwd — the viewer package when vitest runs from
 * its own config, the repo root under `make test-fast`.
 */
function publicApiExports(): string[] {
  const candidates = [
    join(process.cwd(), 'scripts', 'public-api-exports.json'),
    join(process.cwd(), 'packages', 'luxar-viewer', 'scripts', 'public-api-exports.json'),
  ];
  const path = candidates.find((candidate) => existsSync(candidate));
  if (path === undefined) {
    throw new Error(`public-api-exports.json not found at ${candidates.join(' or ')}`);
  }
  const parsed: unknown = JSON.parse(readFileSync(path, 'utf8'));
  const exportsList = (parsed as { exports?: unknown }).exports;
  if (!Array.isArray(exportsList)) throw new Error('public-api-exports.json has no exports array');
  return exportsList as string[];
}

type ConsoleMethod = 'log' | 'warn' | 'error' | 'info' | 'debug';
const CONSOLE_METHODS: readonly ConsoleMethod[] = ['log', 'warn', 'error', 'info', 'debug'];

interface BarrelTestGlobals {
  __preBarrelConsole?: Record<ConsoleMethod, typeof console.log>;
}

describe('Public barrel side effects', () => {
  beforeAll(async () => {
    // Reset module cache so the barrel import is observed FRESH in this
    // process. Without this, a mutant that schedules a one-shot side
    // effect via `if (!globalThis.__luxarLoaded) { ... mutate(); }` would
    // slip through if any earlier test (in this or another file) already
    // touched the barrel. [api.md/C2]
    vi.resetModules();

    // Snapshot ALL five console methods BEFORE the barrel is touched. The
    // bootstrap path (core/bootstrap.ts → consoleInterceptor.patch())
    // patches log/warn/error/info/debug; checking only `log` would let a
    // regression that patches only `warn` slip through. [api.md/C1]
    const snapshot = {} as Record<ConsoleMethod, typeof console.log>;
    for (const m of CONSOLE_METHODS) {
      snapshot[m] = console[m];
    }
    (globalThis as BarrelTestGlobals).__preBarrelConsole = snapshot;

    // This is the suite's only cold full-graph probe. A timeout here is reported
    // as infrastructure failure and skips all 14 embeddability assertions; if it
    // recurs, investigate a load-dependent barrel stall rather than raising the budget.
    // Whatever is loaded here is what consumers see.
    await import('../../../index');
  });

  it('does not patch any of the five console methods (log/warn/error/info/debug)', () => {
    const pre = (globalThis as BarrelTestGlobals).__preBarrelConsole;
    expect(pre).toBeDefined();
    for (const m of CONSOLE_METHODS) {
      expect(console[m], `console.${m} was patched by barrel import`).toBe(pre![m]);
    }
  });

  it('does not inject any --luxar-* CSS variables on documentElement', () => {
    const inlineStyle = document.documentElement.getAttribute('style') ?? '';
    expect(inlineStyle).not.toMatch(/--luxar-/);
  });

  it('does not append any luxar-* DOM elements to document.body', () => {
    const luxarChildren = Array.from(document.body.children).filter((el) =>
      (el.id || '').startsWith('luxar-')
    );
    expect(luxarChildren).toEqual([]);
  });

  // [R11/C-G15][P5] Scan `document.head` too. A regression that injected
  // `<style id="luxar-…">` or `<link id="luxar-…">` into the document
  // head at module-load (a common pattern for CSS-in-JS libraries that
  // forgot to gate behind init()) would slip past the body-only check.
  // Match by id-prefix AND by class-prefix (CSS-in-JS libs frequently
  // assign a class rather than an id).
  it('does not inject any luxar-* <style> / <link> elements into document.head', () => {
    const headChildren = Array.from(document.head.children);
    const offenders = headChildren.filter((el) => {
      const id = (el.id || '').startsWith('luxar-');
      const className = (el.getAttribute('class') || '')
        .split(/\s+/)
        .some((c) => c.startsWith('luxar-'));
      return id || className;
    });
    expect(offenders).toEqual([]);
  });

  it('does not set window.__luxarDebug', () => {
    expect(window.__luxarDebug).toBeUndefined();
  });

  // api.md O1 / Phase E1 fix: previous version bundled all four assertions
  // (function-typed, prototype-object, init, dispose) into a single `it`.
  // A mutant that drops only `dispose` from the prototype would surface as
  // a single failure with a generic "exposes LuxarApp..." name, requiring
  // the diff to read the assertion to know what regressed. Splitting via
  // `it.each` makes the failing line name itself (`exposes LuxarApp: dispose`).
  // Each row pins ONE shape of the contract:
  //   - typeof === 'function'  — pinned by api.md W1/G5
  //   - typeof prototype === 'object'  — guards `export const LuxarApp = () => ...`
  //   - prototype.init is a function  — documented lifecycle
  //   - prototype.dispose is a function  — documented lifecycle
  it.each<{
    label: string;
    check: (mod: typeof import('../../../index')) => unknown;
    expected: string;
  }>([
    {
      label: 'typeof LuxarApp === function',
      check: (m) => typeof m.LuxarApp,
      expected: 'function',
    },
    {
      label: 'typeof LuxarApp.prototype === object',
      check: (m) => typeof m.LuxarApp.prototype,
      expected: 'object',
    },
    {
      label: 'typeof LuxarApp.prototype.init === function',
      check: (m) => typeof m.LuxarApp.prototype.init,
      expected: 'function',
    },
    {
      label: 'typeof LuxarApp.prototype.dispose === function',
      check: (m) => typeof m.LuxarApp.prototype.dispose,
      expected: 'function',
    },
  ])('exposes LuxarApp as a constructor: $label', async ({ check, expected }) => {
    const mod = await import('../../../index');
    expect(check(mod)).toBe(expected);
  });

  it('three URL/bootstrap entrypoints are runtime functions exported from the barrel [api.md O3]', async () => {
    // api.md G1 fix: previous version pinned only 5 of 8 value exports.
    // Add the 3 missing: normalizeDataSourceUrl + the rendering helpers.
    const mod = await import('../../../index');
    expect(typeof mod.bootstrapStandalone).toBe('function');
    expect(typeof mod.readUrlParams).toBe('function');
    expect(typeof mod.normalizeDataSourceUrl).toBe('function');
  });

  it('exposes rendering helpers (blending + colormap) as callable functions', async () => {
    // api.md G1 fix: pin the 5 rendering helpers re-exported from index.ts
    // (getCompleteBlendingState, applyBlendingStateToMaterial,
    // supportsScalarColormap, applyColormapTextureToMaterial,
    // applyScalarRangeToMaterial). Each is a STABLE public API per the
    // module's own docstring; a regression that drops one is a breaking
    // change that should be caught here.
    const mod = await import('../../../index');
    expect(typeof mod.getCompleteBlendingState).toBe('function');
    expect(typeof mod.applyBlendingStateToMaterial).toBe('function');
    expect(typeof mod.supportsScalarColormap).toBe('function');
    expect(typeof mod.applyColormapTextureToMaterial).toBe('function');
    expect(typeof mod.applyScalarRangeToMaterial).toBe('function');
  });

  // api.md O2 / Phase E2 fix: previous version bundled the StorageKeys
  // container shape (typeof === 'object', not null, non-empty) with a
  // per-entry contract check (each value is a `luxar.`-prefixed string,
  // or a function returning one) inside a single `it`. A broken entry
  // and a wholly-replaced container both surfaced as one generic
  // "exposes StorageKeys..." failure. Split into:
  //   (a) one `it` that pins the container shape (3 assertions);
  //   (b) one `it.each` that iterates over the entries and names the
  //       offending entry-key on failure.
  it('exposes StorageKeys as a non-null, non-empty object', async () => {
    const mod = await import('../../../index');
    expect(typeof mod.StorageKeys).toBe('object');
    expect(mod.StorageKeys).not.toBeNull();
    expect(Object.keys(mod.StorageKeys).length).toBeGreaterThan(0);
  });

  // Per-entry contract: every value must be either a `luxar.`-prefixed
  // string, or a (sceneId) => string that returns a `luxar.`-prefixed
  // string. Failures now name the specific entry (e.g.
  // "StorageKeys.rendering is a luxar.* string or factory").
  it('every StorageKeys entry is a luxar.* string or a (sceneId) => luxar.* factory', async () => {
    const mod = await import('../../../index');
    const entries = Object.entries(mod.StorageKeys);
    expect(entries.length).toBeGreaterThan(0);
    for (const [key, value] of entries) {
      if (typeof value === 'function') {
        const built = (value as (s: string) => string)('test-scene');
        expect(typeof built, `StorageKeys.${key}('test-scene') type`).toBe('string');
        expect(built.startsWith('luxar.'), `StorageKeys.${key}('test-scene') prefix`).toBe(true);
      } else {
        expect(typeof value, `StorageKeys.${key} type`).toBe('string');
        expect((value as string).startsWith('luxar.'), `StorageKeys.${key} prefix`).toBe(true);
      }
    }
  });

  it('does not leak additional unexpected exports (public surface lock)', async () => {
    // api.md G3 fix: a mutation that adds a leaky export would silently
    // expand the public surface and never fail any test. Pin the FULL
    // value-export set against scripts/public-api-exports.json — the SAME
    // list `scripts/check-lib-exports.mjs` holds the built dist/lib bundle
    // to, so the source barrel and the shipped bundle cannot disagree about
    // what is public. New public APIs must update that file.
    const mod = await import('../../../index');
    const valueExports = Object.keys(mod).sort();
    expect(valueExports).toEqual(publicApiExports());
  });

  it('keeps scripts/public-api-exports.json sorted and duplicate-free', () => {
    // The list is the canonical form both consumers compare against; an
    // unsorted or duplicated entry would still "match" a sorted actual set
    // by accident in one consumer and not the other.
    const listed = publicApiExports();
    expect(listed).toEqual([...new Set(listed)].sort());
    expect(listed.length).toBeGreaterThan(0);
  });
});
