/**
 * Every `localStorage` touch in production code goes through `StorageKeys`.
 *
 * `utils/storage-keys.ts` is documented as the single source of truth for the
 * viewer's `localStorage` keys, but nothing enforced it: `ui/control-rail.ts`
 * shipped two hyphenated `luxar-control-rail-*` keys beside the `luxar.*`
 * registry for months. This scan reads the source tree and fails on any
 * `localStorage.getItem/setItem/removeItem(` whose first argument is not a
 * `StorageKeys.<entry>` reference (a static key or the `rendering(sceneId)`
 * factory) — an alias such as `this.STORAGE_KEY` is rejected too, because a
 * reader grepping the registry would not find it.
 *
 * Both directions are pinned: the scanner must flag a synthetic violation, and
 * it must find the real call sites (a scan of zero files proves nothing).
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const SRC_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');

/** Production `.ts` files under `src/`: no tests, no declarations. */
function listProductionSources(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) {
      if (entry === 'tests' || entry === 'node_modules') continue;
      listProductionSources(full, out);
    } else if (
      entry.endsWith('.ts') &&
      !entry.endsWith('.d.ts') &&
      !entry.endsWith('.test.ts') &&
      !entry.endsWith('.spec.ts')
    ) {
      out.push(full);
    }
  }
  return out;
}

const CALL_RE = /localStorage\s*\.\s*(getItem|setItem|removeItem)\s*\(\s*([^,)]*)/g;

/** Offending `localStorage.<op>(<arg>` snippets in `source`; empty when clean. */
export function findUnregisteredStorageKeyUses(source: string): string[] {
  const offenders: string[] = [];
  for (const match of source.matchAll(CALL_RE)) {
    const arg = match[2].trim();
    if (!/^StorageKeys\.\w+/.test(arg)) offenders.push(`localStorage.${match[1]}(${arg}`);
  }
  return offenders;
}

describe('localStorage keys come from StorageKeys (source scan)', () => {
  const files = listProductionSources(SRC_ROOT);

  it('scans the production sources and finds the known localStorage call sites', () => {
    const touching = files.filter((f) => CALL_RE.test(readFileSync(f, 'utf8')));
    CALL_RE.lastIndex = 0;
    // The registry's own consumers — if this list shrinks to nothing the
    // scan is looking in the wrong place, not at a clean tree.
    const names = touching.map((f) => path.relative(SRC_ROOT, f));
    expect(names).toEqual(
      expect.arrayContaining([
        path.join('ui', 'control-rail.ts'),
        path.join('ui', 'rendering-controls', 'settings-persistence.ts'),
        path.join('config', 'user-settings.ts'),
        path.join('themes', 'theme-manager.ts'),
      ])
    );
  });

  it('every localStorage.getItem/setItem/removeItem argument is a StorageKeys reference', () => {
    const offenders: string[] = [];
    for (const file of files) {
      for (const hit of findUnregisteredStorageKeyUses(readFileSync(file, 'utf8'))) {
        offenders.push(`${path.relative(SRC_ROOT, file)}: ${hit}`);
      }
    }
    expect(offenders, 'add the key to utils/storage-keys.ts and reference it').toEqual([]);
  });

  it('flags a literal key, an alias, and a multi-line call (scanner self-test)', () => {
    expect(
      findUnregisteredStorageKeyUses(`
        localStorage.getItem('luxar.rogue');
        localStorage.setItem(this.STORAGE_KEY, v);
        localStorage.removeItem(
          KEY
        );
      `)
    ).toEqual([
      "localStorage.getItem('luxar.rogue'",
      'localStorage.setItem(this.STORAGE_KEY',
      'localStorage.removeItem(KEY',
    ]);
  });

  it('accepts static entries and the rendering(sceneId) factory', () => {
    expect(
      findUnregisteredStorageKeyUses(`
        localStorage.getItem(StorageKeys.theme);
        localStorage.setItem(
          StorageKeys.audio,
          JSON.stringify(prefs)
        );
        localStorage.removeItem(StorageKeys.rendering(sceneId));
      `)
    ).toEqual([]);
  });
});
