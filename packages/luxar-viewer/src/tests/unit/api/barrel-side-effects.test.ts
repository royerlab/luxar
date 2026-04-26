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

import { describe, expect, it, beforeAll } from 'vitest';

describe('Public barrel side effects', () => {
  beforeAll(async () => {
    // Snapshot console BEFORE the barrel is touched. Vitest may have
    // wired its own console.* (jsdom polyfills, mock globals); we snapshot
    // whatever's there so the assertion compares apples to apples.
    (globalThis as { __preBarrelConsoleLog?: typeof console.log }).__preBarrelConsoleLog =
      console.log;

    // Now import the barrel. Whatever is loaded here is what consumers see.
    await import('../../../index');
  });

  it('does not patch console.log', () => {
    const pre = (globalThis as { __preBarrelConsoleLog?: typeof console.log })
      .__preBarrelConsoleLog;
    expect(console.log).toBe(pre);
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

  it('does not set window.__luxarDebug', () => {
    expect(window.__luxarDebug).toBeUndefined();
  });

  it('exposes LuxarApp, LuxarAppOptions, bootstrapStandalone, readUrlParams, StorageKeys', async () => {
    const mod = await import('../../../index');
    expect(typeof mod.LuxarApp).toBe('function');
    expect(typeof mod.bootstrapStandalone).toBe('function');
    expect(typeof mod.readUrlParams).toBe('function');
    expect(typeof mod.StorageKeys).toBe('object');
  });
});
