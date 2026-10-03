/**
 * The staleness scan behind `ensureWasmBuilt` (see `src/tests/global-setup.ts`).
 *
 * A detector that never detects is worse than none: it would report every
 * gitignored `public/wasm/` build as current and leave the opaque
 * "x is not a function" failure exactly where it was. So both directions are
 * pinned here — a current wrapper reports nothing, a wrapper that dropped a
 * kernel names it — plus the emit form itself against the REAL artifact when
 * one is built (always the case in CI, which builds WASM before the suite).
 */

import { existsSync, readFileSync } from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { missingExportsIn } from '../../global-setup';
import { REQUIRED_WASM_EXPORTS } from '../../../wasm/required-exports';

const VIEWER_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');
const WASM_JS_PATH = path.resolve(VIEWER_ROOT, 'public/wasm/luxar_wasm.js');

/** Current parameter count of a kernel whose signature changed (else 2). */
const ARITY: Record<string, number> = { project_gsplats_nd_to_3d: 18 };

/** A wrapper declaring `names` the way wasm-pack emits free functions. */
function wrapperDeclaring(names: readonly string[], arity = ARITY): string {
  return names
    .map((name) => {
      const params = Array.from({ length: arity[name] ?? 2 }, (_, i) => `p${i}`).join(', ');
      return `export function ${name}(${params}) {\n  return wasm.${name}(${params});\n}`;
    })
    .join('\n\n');
}

describe('missingExportsIn', () => {
  it('reports nothing for a wrapper declaring every required export', () => {
    expect(missingExportsIn(wrapperDeclaring(REQUIRED_WASM_EXPORTS))).toEqual([]);
  });

  it('names the kernel a build predates', () => {
    const [dropped, ...kept] = REQUIRED_WASM_EXPORTS;
    expect(missingExportsIn(wrapperDeclaring(kept))).toEqual([dropped]);
  });

  it('reports every required export for an empty/truncated wrapper', () => {
    expect(missingExportsIn('')).toEqual([...REQUIRED_WASM_EXPORTS]);
  });

  it('does not accept a mere MENTION of the name as a declaration', () => {
    // The wrapper names every kernel it forwards in an internal `wasm.<name>()`
    // call and in its JSDoc, so a substring scan would call any build current.
    const [name] = REQUIRED_WASM_EXPORTS;
    const mentionsOnly = `/** See [\`${name}\`] below. */\nconst f = () => wasm.${name}(0);\n`;
    expect(missingExportsIn(mentionsOnly)).toContain(name);
  });

  it('names a kernel whose wrapper predates its current signature', () => {
    // A stale build still exports project_gsplats_nd_to_3d, without the
    // trailing out_source_indices parameter.
    const stale = wrapperDeclaring([...REQUIRED_WASM_EXPORTS, 'project_gsplats_nd_to_3d'], {
      project_gsplats_nd_to_3d: 17,
    });
    expect(missingExportsIn(stale)).toEqual(['project_gsplats_nd_to_3d']);
  });

  it('does not confuse a longer name that starts with a required one', () => {
    const [name] = REQUIRED_WASM_EXPORTS;
    expect(missingExportsIn(wrapperDeclaring([`${name}_v2`]))).toContain(name);
  });

  // The emit form is the one assumption the scan makes about wasm-pack. If a
  // toolchain upgrade ever changed it, every build would read as stale and the
  // suite would rebuild on every run — so check it against a real artifact.
  it.skipIf(!existsSync(WASM_JS_PATH))('matches the exports of a freshly built wrapper', () => {
    expect(missingExportsIn(readFileSync(WASM_JS_PATH, 'utf8'))).toEqual([]);
  });
});
