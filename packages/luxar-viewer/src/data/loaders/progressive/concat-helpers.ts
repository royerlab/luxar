/**
 * Shared concatenation helpers for the progressive (additive-LOD) loaders.
 *
 * The points / lines / gsplats progressive loaders each concatenate a set of
 * per-LOD typed-array fields. The "allocate `total * perItem`, copy at a
 * running offset" loop and the "all-or-nothing optional field" gate were
 * duplicated across all three; they live here once. Geometry-specific
 * wrinkles (segment-index offsetting, colour fill-with-white) stay in the
 * individual loaders.
 *
 * The helpers are structurally generic over the concrete typed-array type
 * `A` (Float32Array, Uint8/16/32Array, Float16Array, …) so each loader keeps
 * its own narrow field types (e.g. `ColorArray`, `ScalarArray`) with the
 * source dtype preserved.
 *
 * @module data/loaders/progressive/concat-helpers
 */

/** Structural shape common to every typed array we concatenate. */
export interface ConcatTypedArray {
  readonly length: number;
  set(array: ArrayLike<number>, offset?: number): void;
}

/**
 * Concatenate a **required** typed-array field across parts.
 *
 * Allocates a fresh array of `sum(countOf(part)) * perItem` elements (using
 * the first part's constructor, so the dtype is preserved) and copies each
 * part in order.
 *
 * LADDER-DTYPE CONTRACT: every part must carry the field with the SAME
 * typed-array dtype. `TypedArray.set` converts by VALUE, not by semantics —
 * copying a later Uint8Array level (0..255 normalized) into a Float32Array
 * output (0..1) would silently write 255× values, and Float32 into Uint8
 * truncates to garbage. The Python writer always emits one dtype per field
 * across a ladder, so a mismatch means malformed data; fail fast with a
 * descriptive error instead of rendering corruption.
 *
 * @param parts - Per-LOD data parts (assumed length ≥ 1).
 * @param get - Extract the field from a part.
 * @param countOf - Element count of a part (rows; multiplied by `perItem`).
 * @param perItem - Components per element (e.g. 3 for positions, 1 for widths).
 * @param label - Field name for the dtype-mismatch error message.
 */
export function concatRequiredField<A extends ConcatTypedArray, P>(
  parts: P[],
  get: (p: P) => A,
  countOf: (p: P) => number,
  perItem = 1,
  label = 'field'
): A {
  const total = parts.reduce((s, p) => s + countOf(p), 0);
  const ctor = (get(parts[0]) as unknown as { constructor: new (n: number) => A }).constructor;
  for (let i = 1; i < parts.length; i++) {
    const other = (get(parts[i]) as unknown as { constructor: unknown }).constructor;
    if (other !== ctor) {
      throw new Error(
        `concatRequiredField: LOD level ${i} carries '${label}' as ` +
          `${(other as { name?: string }).name} but level 0 uses ` +
          `${(ctor as { name?: string }).name} — ladder levels must share each ` +
          "field's dtype (TypedArray.set converts by value, not semantics)."
      );
    }
  }
  const out = new ctor(total * perItem);
  let offset = 0;
  for (const p of parts) {
    out.set(get(p) as unknown as ArrayLike<number>, offset * perItem);
    offset += countOf(p);
  }
  return out;
}

/**
 * Concatenate an **optional** typed-array field with all-or-nothing policy:
 * the field is produced only when **every** part carries it; otherwise
 * `undefined` is returned (the attribute is dropped for the merged result).
 *
 * Preserves the source dtype by constructing from the first part's array.
 *
 * ZERO-ROW PARTS ARE ABSTAINERS (#1456). A part with `countOf(p) === 0`
 * contributes no rows, so it has no opinion about which attributes the merged
 * result carries — it is excluded from both the presence gate and the copy.
 * This matters because the canonical empty payloads OMIT their optional fields
 * rather than emitting zero-length arrays, so a slice that culls one level of an
 * additive ladder to zero would otherwise strip those fields from the WHOLE
 * ladder — including the levels that DO have data, whose values the node
 * adapters then replace with constant fills. Which fields, per geometry:
 *
 * - Points: `createEmptyPointsData` omits `colors`, `radii`, `sharpness` and
 *   `scalars` — all four route through here, so all four were lost (white,
 *   radius 0.5, sharpness 0.5, and `hasScalars` cleared = dead colormap).
 * - Lines: `createEmptyLinesData` omits only `scalars` (there is no `radii`
 *   field at all, and `colors`/`sharpness` are present-but-null and use the
 *   bespoke find-first + fill-for-missing path in `concatenateLinesData`, not
 *   this helper), so only the colormap was lost.
 * - GSplats: unaffected — its empty payload carries zero-length `Float32Array`s
 *   for the required fields and null colours on the fill-for-missing path; it
 *   never calls this helper.
 *
 * Excluding zero-row parts is also what keeps the dtype contract honest: such a
 * part copies nothing, so comparing its (possibly absent, possibly differently
 * typed) array against the real levels' could only produce a spurious throw.
 *
 * When EVERY part is zero-row the gate falls back to all the parts, so a wholly
 * empty ladder yields exactly what it did before — `undefined` if the parts
 * omit the field, else a zero-length array of the declared dtype.
 *
 * COUPLED CONTRACT: the append fast path's commit gates
 * (`commit-points-geometry.ts` / `commit-lines-geometry.ts`) compare each
 * optional field's PRESENCE against the committed parent precisely because of
 * this all-or-nothing drop — a new level without the field flips the merged
 * result from real values to the adapter's constant fill. If this policy ever
 * changes (e.g. fill-with-default), revisit those presence conjuncts. Those
 * gates stay correct under the abstainer rule: a ladder whose first committed
 * level is empty publishes no optional fields, and the pass that adds a
 * non-empty level flips presence on — a difference the gates already read as
 * "not a pure append" and answer with a full rewrite.
 */
export function concatOptionalField<A extends ConcatTypedArray, P>(
  parts: P[],
  get: (p: P) => A | null | undefined,
  countOf: (p: P) => number,
  perItem = 1,
  label = 'field'
): A | undefined {
  const contributing = parts.filter((p) => countOf(p) > 0);
  const voting = contributing.length > 0 ? contributing : parts;
  if (!voting.every((p) => get(p) != null)) return undefined;
  return concatRequiredField(voting, (p) => get(p) as A, countOf, perItem, label);
}
