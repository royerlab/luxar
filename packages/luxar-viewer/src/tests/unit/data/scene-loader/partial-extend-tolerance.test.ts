/**
 * `PARTIAL_EXTEND_TOLERANCE` must be the only place the flag is decided.
 *
 * `GEOMETRY_DESCRIPTORS` is documented as the single source of truth for what
 * each geometry kind needs from the shared scene-loader machinery — but twelve
 * call sites hardcoded `applyPartialExtendTolerance` and only two read the
 * table. Hardcoding is not a compile error, and a wrong value is not a crash:
 * it is a clipping result quietly too generous or too tight, on one geometry
 * type, in one code path. Nothing would have named it.
 *
 * Four checks, in order of what they can catch:
 *
 * 1. Every drawable kind has an entry, derived from the format contract's
 *    vocabulary rather than from a list written here.
 * 2. Lines remains the one kind that opts out.
 * 3. The descriptor's field equals the table, so the two cannot drift.
 * 4. No production source under `src/` writes the literal in a code position.
 *    This is the one that would have caught the original state, and the only
 *    one that keeps catching a NEW hardcoded site.
 */

import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { PARTIAL_EXTEND_TOLERANCE } from '../../../../data/scene-loader/partial-extend-tolerance';
import { GEOMETRY_DESCRIPTORS } from '../../../../data/scene-loader/geometry-descriptors';
import { LOADER_TYPES } from '../../../../types/format-contract';

const SRC_ROOT = join(__dirname, '..', '..', '..', '..');

/** Every `.ts` under `src/`, excluding tests and the table's own module. */
function productionSources(dir: string, acc: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      if (entry === 'tests' || entry === 'node_modules') continue;
      productionSources(full, acc);
    } else if (entry.endsWith('.ts') && !entry.endsWith('.test.ts')) {
      acc.push(full);
    }
  }
  return acc;
}

describe('PARTIAL_EXTEND_TOLERANCE', () => {
  it('covers every drawable geometry kind', () => {
    // Derived from the format contract, so a kind added there without an entry
    // here fails rather than silently defaulting.
    expect(LOADER_TYPES.length).toBeGreaterThanOrEqual(4);
    for (const kind of LOADER_TYPES) {
      expect(
        PARTIAL_EXTEND_TOLERANCE,
        `PARTIAL_EXTEND_TOLERANCE has no entry for '${kind}'`
      ).toHaveProperty(kind);
      expect(typeof PARTIAL_EXTEND_TOLERANCE[kind]).toBe('boolean');
    }
  });

  it('records lines as the one kind that opts out', () => {
    // Spelled out rather than derived: this is the asymmetry the whole table
    // exists for, and a table that became uniformly `true` would satisfy every
    // structural check above while being wrong.
    expect(PARTIAL_EXTEND_TOLERANCE.lines).toBe(false);
    expect(PARTIAL_EXTEND_TOLERANCE.points).toBe(true);
    expect(PARTIAL_EXTEND_TOLERANCE.gsplats).toBe(true);
    expect(PARTIAL_EXTEND_TOLERANCE.mesh).toBe(true);
  });

  it('agrees with GEOMETRY_DESCRIPTORS for every kind', () => {
    for (const kind of LOADER_TYPES) {
      expect(
        GEOMETRY_DESCRIPTORS[kind].applyPartialExtendTolerance,
        `GEOMETRY_DESCRIPTORS.${kind} disagrees with the table`
      ).toBe(PARTIAL_EXTEND_TOLERANCE[kind]);
    }
  });

  it('is the only place the flag is decided in production code', () => {
    const files = productionSources(SRC_ROOT);
    // Fail closed: an empty or tiny scan would make this pass vacuously, which
    // is exactly how a source-scanning gate stops working.
    expect(files.length).toBeGreaterThan(200);

    const offenders: string[] = [];
    for (const file of files) {
      if (file.endsWith('partial-extend-tolerance.ts')) continue;
      for (const [i, line] of readFileSync(file, 'utf8').split('\n').entries()) {
        // Code position only: `applyPartialExtendTolerance: true,` as an object
        // property. Prose mentioning the flag inside a `//` or ` * ` comment is
        // fine and there is plenty of it — those comments are what explain WHY
        // lines differs.
        const trimmed = line.trim();
        if (trimmed.startsWith('*') || trimmed.startsWith('//')) continue;
        if (/applyPartialExtendTolerance:\s*(true|false)\b/.test(line)) {
          offenders.push(`${file.slice(SRC_ROOT.length + 1)}:${i + 1}: ${trimmed}`);
        }
      }
    }
    expect(
      offenders,
      'these sites hardcode applyPartialExtendTolerance instead of reading ' +
        'PARTIAL_EXTEND_TOLERANCE:\n  ' +
        offenders.join('\n  ')
    ).toEqual([]);
  });
});
