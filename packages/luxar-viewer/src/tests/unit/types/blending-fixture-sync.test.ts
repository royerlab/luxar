/**
 * Guard: the Python fixture generator's BLENDING_MODES twin stays
 * member-equal to the TS SSOT tuple.
 *
 * `tests/fixtures/generate_test_data.py` cannot import
 * `types/blending.ts`, so it hard-codes its own mode list (plus the
 * per-mode color/angle/z tables the blending E2E fixtures derive from).
 * Membership divergence fails SILENTLY: a 7th mode added to the TS
 * tuple but not the Python list would ship with zero E2E coverage —
 * the per-mode E2E loops iterate whatever the fixtures contain.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { BLENDING_MODES } from '../../../types/blending';

describe('fixture-generator mode list ↔ TS BLENDING_MODES tuple', () => {
  const generatorPath = resolve(
    dirname(fileURLToPath(import.meta.url)),
    '../../../../tests/fixtures/generate_test_data.py'
  );

  it('the Python BLENDING_MODES twin is member-equal to the TS tuple', () => {
    const source = readFileSync(generatorPath, 'utf-8');
    const match = source.match(/^BLENDING_MODES\s*=\s*\[([^\]]*)\]/m);
    expect(match, 'BLENDING_MODES list not found in generate_test_data.py').toBeTruthy();
    const pythonModes = Array.from(match![1].matchAll(/"([^"]+)"/g), (m) => m[1]);
    // Set equality — ORDER may differ (TS is panel-dropdown order,
    // Python is fixture order); membership must not.
    expect(new Set(pythonModes)).toEqual(new Set(BLENDING_MODES));
  });

  it('every mode has fixture color/angle/z table entries', () => {
    const source = readFileSync(generatorPath, 'utf-8');
    for (const table of ['BLENDING_MODE_COLORS', 'BLENDING_MODE_ANGLES', 'BLENDING_MODE_Z']) {
      const block = source.match(new RegExp(`${table}\\s*=\\s*\\{([^}]*)\\}`, 'm'));
      expect(block, `${table} not found`).toBeTruthy();
      const keys = Array.from(block![1].matchAll(/"([^"]+)"\s*:/g), (m) => m[1]);
      expect(new Set(keys), `${table} keys`).toEqual(new Set(BLENDING_MODES));
    }
  });
});
