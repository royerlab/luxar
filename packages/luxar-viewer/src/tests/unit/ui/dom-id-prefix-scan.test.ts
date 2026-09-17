/**
 * Every DOM `id="…"` the viewer's UI writes is `luxar-` prefixed.
 *
 * The viewer is embedded in host pages, and an element id is page-global: a
 * bare `id="manual-path"` (the dataset browser's manual-entry field until this
 * scan existed) collides with any host element of the same name and breaks
 * both `label[for]` pairings. The `luxar-` prefix is the same rule the CSS
 * classes and window events already follow. This scan reads `src/ui/**` and
 * fails on any `id="…"` (or `id='…'`) literal that does not start with
 * `luxar-`; `data-*-id="…"` attributes are not element ids and are skipped.
 *
 * Both directions are pinned: the scanner must flag a synthetic violation and
 * must find the real ids (a scan of zero files proves nothing).
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const UI_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../ui');

function listUiSources(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    if (statSync(full).isDirectory()) listUiSources(full, out);
    else if (entry.endsWith('.ts') && !entry.endsWith('.d.ts')) out.push(full);
  }
  return out;
}

/** `id="…"` / `id='…'` literals that are element ids, not `data-*-id` attrs. */
const ID_RE = /(?<![\w-])id=(["'])([^"']*)\1/g;

/** Every id literal in `source` that lacks the `luxar-` prefix. */
export function findUnprefixedDomIds(source: string): string[] {
  const offenders: string[] = [];
  for (const match of source.matchAll(ID_RE)) {
    if (!match[2].startsWith('luxar-')) offenders.push(match[2]);
  }
  return offenders;
}

describe('DOM ids in src/ui are luxar- prefixed (source scan)', () => {
  const files = listUiSources(UI_ROOT);

  it('finds id literals to check (the scan is not vacuous)', () => {
    let total = 0;
    for (const file of files) total += [...readFileSync(file, 'utf8').matchAll(ID_RE)].length;
    expect(total).toBeGreaterThanOrEqual(5);
  });

  it('every id="…" literal starts with luxar-', () => {
    const offenders: string[] = [];
    for (const file of files) {
      for (const id of findUnprefixedDomIds(readFileSync(file, 'utf8'))) {
        offenders.push(`${path.relative(UI_ROOT, file)}: id="${id}"`);
      }
    }
    expect(offenders).toEqual([]);
  });

  it('flags a bare id and skips data-*-id attributes (scanner self-test)', () => {
    expect(
      findUnprefixedDomIds(`
        <input id="manual-path" />
        <div id='luxar-ok' data-tab-id="overview" data-toggle-id="x"></div>
        <button id="luxar-dataset-browser-manual-load"></button>
      `)
    ).toEqual(['manual-path']);
  });
});
