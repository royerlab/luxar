/**
 * The runner that holds one row of the geometry-behaviour matrix against the
 * code.
 *
 * A row's probe file calls {@link defineBehaviourConformance} once, with the
 * row id and a {@link BehaviourProbe}. It emits one test per geometry type:
 *
 * - a `'yes'` cell runs `probe.holds(type)`, which drives real code through
 *   that type's adapter and asserts the behaviour;
 * - an absent cell runs the probe's checker for the cell's declared
 *   `enforcedBy`, which asserts the refusal message, the hidden predicate, or
 *   the unchanged state. A probe that has no checker for a declared
 *   enforcement fails the cell, so an absent cell can never pass by not being
 *   looked at.
 *
 * Follows the shape of `tests/unit/data/_shared/refinement-loop-contract.ts`:
 * one shared contract, thin per-type adapters, called from the file that owns
 * the fixtures. That is also why the probes are not all in one file — several
 * rows need module mocks (`vi.mock`) that are per file.
 *
 * @module tests/_conformance/define-behaviour-conformance
 */

import { describe, it } from 'vitest';

import { GEOMETRY_TYPES, type GeometryTypeName } from '../../types/format-contract';
import {
  behaviourRow,
  isAbsent,
  type Absent,
  type BehaviourId,
  type Enforcement,
} from './geometry-behaviours';

/** Asserts one absent cell's enforcement for `type`. */
export type EnforcementCheck = (type: GeometryTypeName, cell: Absent) => void | Promise<void>;

/** What a row's probe file supplies. */
export interface BehaviourProbe {
  /** Assert the behaviour holds for `type` (a `'yes'` cell). */
  holds(type: GeometryTypeName): void | Promise<void>;
  /** One checker per enforcement this row's absent cells declare. */
  enforced?: Partial<Record<Enforcement, EnforcementCheck>>;
}

/** The test title for one cell: the type and what it declares. */
function cellTitle(type: GeometryTypeName, id: BehaviourId): string {
  const cell = behaviourRow(id).cells[type];
  return isAbsent(cell) ? `${type}: absent, ${cell.enforcedBy}` : `${type}: yes`;
}

/**
 * Emit the conformance tests for row `id`: one per geometry type, each
 * asserting what the matrix declares for it.
 */
export function defineBehaviourConformance(id: BehaviourId, probe: BehaviourProbe): void {
  const row = behaviourRow(id);
  describe(`geometry behaviour ${id}`, () => {
    it.each(GEOMETRY_TYPES.map((type) => [cellTitle(type, id), type] as const))(
      '%s',
      async (_title, type) => {
        const cell = row.cells[type];
        if (!isAbsent(cell)) {
          await probe.holds(type);
          return;
        }
        const check = probe.enforced?.[cell.enforcedBy];
        if (!check) {
          throw new Error(
            `${id}.${type} is declared absent (enforcedBy '${cell.enforcedBy}') but the ` +
              `probe in ${row.probedIn} has no '${cell.enforcedBy}' checker, so nothing ` +
              'verifies the declaration'
          );
        }
        await check(type, cell);
      }
    );
  });
}
