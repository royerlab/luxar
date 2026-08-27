// @vitest-environment jsdom
/**
 * Asserts that importing the public package barrel (`src/index.ts`) is
 * side-effect-free.
 *
 * This is the contract that makes Luxar safely embeddable: a third-party
 * page doing `import { LuxarApp } from 'luxar-viewer'` must NOT find its
 * console silently monkey-patched, its `:root` CSS variables overwritten,
 * or its `<body>` mutated by a singleton waking up at module-load.
 *
 * If you find yourself relaxing one of these assertions, instead push the
 * side effect down into LuxarApp.init() (gated by an option) or expose it
 * as an explicit factory the consumer must call.
 */

import { describe, expect, it, beforeAll, vi } from 'vitest';

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

    // Now import the barrel. Whatever is loaded here is what consumers see.
    await import('../../../index');
    // This hook transforms the ENTIRE public module graph from cold — after
    // `vi.resetModules()` there is nothing cached to reuse — so it is bounded by
    // Vite's transform throughput, not by anything the assertions do. Against the
    // 15 s local `hookTimeout` it fails as "Hook timed out in 15000ms" on a
    // loaded machine, taking all 14 embeddability assertions silently with it:
    // a hook failure reports as an infrastructure error, not as a barrel
    // regression, so the suite reads as noise rather than as a finding.
    // Observed doing exactly that while the rest of the suite was green.
    //
    // Keep the exceptional 90 s budget here rather than lifting the shared
    // ceilings further: 15 s stays strict locally, while CI's 60 s ceiling
    // covers contention for ordinary hooks without masking longer hangs.
  }, 90_000);

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
    // value-export set; new public APIs must update this list.
    const mod = await import('../../../index');
    const valueExports = Object.keys(mod).sort();
    expect(valueExports).toEqual(
      [
        'InputContext',
        'KeyAction',
        'LuxarApp',
        'LuxarLayer',
        'StorageKeys',
        'applyBlendingStateToMaterial',
        'applyColormapTextureToMaterial',
        'applyScalarRangeToMaterial',
        'bootstrapStandalone',
        'getCompleteBlendingState',
        'normalizeDataSourceUrl',
        'readUrlParams',
        'supportsScalarColormap',
      ].sort()
    );
  });
});
