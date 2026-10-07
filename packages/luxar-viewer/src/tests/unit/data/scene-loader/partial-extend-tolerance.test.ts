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
 * 2. The flag each kind's descriptor carries does what the geometry-behaviour
 *    matrix (`partialExtendTolerance`) declares: a partially-extended node's
 *    tolerance widens for every kind but lines, through the real
 *    `deriveNodeViewState`.
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
import { deriveNodeViewState } from '../../../../data/scene-loader/view-state/derive-node-view-state';
import { EXTEND_TO_ALL_TOLERANCE } from '../../../../data/scene-loader/view-state/extend-tolerance';
import type { ViewState } from '../../../../data/data-loader-types';
import { defineBehaviourConformance } from '../../../_conformance/define-behaviour-conformance';

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

/**
 * A node extended through `time` but not `channel`, viewed with both hidden:
 * the partial case, where only the descriptor's flag decides whether `time`'s
 * tolerance widens.
 */
function derivePartiallyExtended(kind: (typeof LOADER_TYPES)[number]): ViewState {
  const base: ViewState = {
    displayDims: [2, 3],
    slicePosition: [5, 1, 0, 0],
    tolerance: [0.5, 0.5, 0, 0],
    dimensions: [
      { name: 'time', unit: '', scale: 1 },
      { name: 'channel', unit: '', scale: 1 },
      { name: 'z', unit: 'um', scale: 1 },
      { name: 'y', unit: 'um', scale: 1 },
    ],
  };
  return deriveNodeViewState('/node', { extend_to_all: ['time'] }, base, null, {
    applyPartialExtendTolerance: GEOMETRY_DESCRIPTORS[kind].applyPartialExtendTolerance,
  }).viewState;
}

defineBehaviourConformance('partialExtendTolerance', {
  holds(kind) {
    expect(derivePartiallyExtended(kind).tolerance).toEqual([EXTEND_TO_ALL_TOLERANCE, 0.5, 0, 0]);
  },
  enforced: {
    'no-op': (kind) => {
      expect(derivePartiallyExtended(kind).tolerance).toEqual([0.5, 0.5, 0, 0]);
    },
  },
});

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
