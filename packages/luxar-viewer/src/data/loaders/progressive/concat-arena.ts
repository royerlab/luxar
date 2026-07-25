/**
 * Growable concat ARENA for the progressive (additive-LOD) loaders, plus the
 * per-result append-span registry.
 *
 * ## Why an arena
 *
 * The progressive loaders used to rebuild their full LOD concatenation from
 * scratch every time a level landed: appending level *k* re-allocated and
 * re-copied levels `0..k` — O(k·N) total main-thread copy work across a
 * ladder, measured as the frame-stall source (commit bd039a8ce). The arena
 * amortizes this: each loader keeps ONE capacity-managed buffer per field per
 * reset generation and appends only the NEW level's bytes, so streaming a
 * whole ladder copies O(N_total) overall.
 *
 * ## View-with-copy-on-grow (the aliasing decision)
 *
 * Snapshot results are `subarray(0, used)` VIEWS over the arena buffer, not
 * per-snapshot copies — a copy-per-snapshot would be O(N_total) per level and
 * reinstate exactly the cost this module removes. Views are safe because:
 *
 * - **Prefix immutability**: appends only ever WRITE at entries ≥ the length
 *   of every previously handed-out view, and growth re-allocates (the old
 *   buffer — and every view over it — is left untouched with its final
 *   values). No consumer can observe a mutation through an earlier snapshot.
 * - **No transfers**: nothing downstream transfers a loader-result buffer.
 *   The projection step structured-clones its inputs into the worker (an
 *   explicit "do not add a transfer list" contract in
 *   `data-processor-gsplats.ts`), and the depth-sort coordinator — which DOES
 *   transfer (`registerNode(transfer(..., [buffer.buffer]))` would detach the
 *   whole arena) — is only ever handed freshly-allocated arrays: gsplats pass
 *   the projection output (`positions.slice(...)` on the standard-3D path,
 *   fresh output buffers otherwise), and the points/lines commits use thunks
 *   documented as "MUST allocate fresh" for precisely this reason.
 * - **Read-only consumers**: commit gates compare snapshot results by object
 *   IDENTITY (`committed-data` / `prefix-lineage` WeakMaps); texel writers and
 *   projection kernels only read. (Accumulator `subarray` views already flow
 *   through these paths today via the single-LOD as-is result.)
 *
 * Known cost of views: a structured clone of a TypedArray view clones its
 * entire underlying ArrayBuffer, so worker-projection of a mid-ladder
 * snapshot ships the arena's slack too (≤ 50% under the 1.5× growth policy;
 * 0 after the final-level trim).
 *
 * ## Capacity policy
 *
 * The per-view level sizes are spatial-query results, unknown before each
 * level loads (there is no manifest total to pre-size from), so:
 * - first allocation: exact batch size;
 * - growth: `max(needed, ceil(capacity × 1.5))` — amortized O(N_total);
 * - `exact` growth (final level of the ladder): exactly `needed`;
 * - trim-on-ladder-complete: slack > {@link ARENA_TRIM_SLACK_FRACTION} is
 *   released by re-allocating to exact size, so steady-state memory equals
 *   the old exact-size concat (0% overhead once the ladder completes).
 *
 * ## Append spans
 *
 * Every fresh concat result is stamped (identity-keyed WeakMap, same pattern
 * as `types/prefix-lineage.ts`) with the {@link AppendSpan} it adds relative
 * to its prefix parent — the input for a later worker-input-residency lever
 * that wants to ship only the appended suffix.
 *
 * @module data/loaders/progressive/concat-arena
 */

/** Growth factor for arena re-allocation when a level outgrows capacity. */
export const ARENA_GROWTH_FACTOR = 1.5;

/**
 * Slack fraction above which the final-level trim re-allocates to exact
 * size. Slack at or below this stays (a realloc would copy O(N_total) to
 * reclaim almost nothing); the memory hard gate for the arena is ≤ 3%
 * steady-state overhead, so 2% leaves headroom.
 */
export const ARENA_TRIM_SLACK_FRACTION = 0.02;

// ---------------------------------------------------------------------------
// Append spans
// ---------------------------------------------------------------------------

/**
 * The element range a concat result APPENDS relative to its prefix parent
 * (the previous same-generation memoized result). `fromElement` is 0 for the
 * first result of a generation (it extends nothing — pair with
 * `getPrefixParent` to distinguish "first" from "append").
 *
 * Elements are the geometry's commit count unit: splats (gsplats), points
 * (points), segments (lines). Lines additionally carry the vertex-space span
 * (`positions`/`widths`/`colors` stride by vertices, not segments).
 */
export interface AppendSpan {
  /** First element index the result adds (== the prefix parent's count). */
  fromElement: number;
  /** Number of elements added (0 when a new level was empty). */
  elementCount: number;
  /** Lines only: first vertex index added. */
  fromVertex?: number;
  /** Lines only: number of vertices added. */
  vertexCount?: number;
}

/** result object → the span it appends relative to its prefix parent. */
const appendSpans = new WeakMap<object, AppendSpan>();

/**
 * Record the append span of a freshly memoized concat result. Called by the
 * loaders' `concatenateMemoized` alongside `setPrefixParent`.
 */
export function setAppendSpan(result: object, span: AppendSpan): void {
  appendSpans.set(result, span);
}

/**
 * The span `result` appends relative to its prefix parent, or `undefined`
 * for objects that never passed through a loader's memoized concat (e.g. a
 * slice-cache-restored clone).
 */
export function getAppendSpan(result: object): AppendSpan | undefined {
  return appendSpans.get(result);
}

// ---------------------------------------------------------------------------
// ArenaField — one growable typed-array field
// ---------------------------------------------------------------------------

/**
 * Structural shape of the typed arrays an {@link ArenaField} manages.
 * Extends `concat-helpers.ts`'s `ConcatTypedArray` with the fill/subarray
 * surface the arena needs; kept structural so one field type spans
 * Float32Array, Uint8/16/32Array, Float16Array, and unions thereof.
 */
export interface ArenaTypedArray {
  readonly length: number;
  set(array: ArrayLike<number>, offset?: number): void;
  fill(value: number, start?: number, end?: number): unknown;
  subarray(begin?: number, end?: number): unknown;
}

/** Copy-work counters (test/verification hook; negligible runtime cost). */
export interface ArenaFieldStats {
  /** Entries written by append/appendWith/appendFill — O(N_k) per level. */
  appendedEntries: number;
  /** Entries re-copied by growth/trim re-allocations — amortized O(N_total). */
  reallocCopiedEntries: number;
}

/**
 * One capacity-managed typed-array field of a ladder arena. Tracks length in
 * ELEMENTS (rows); each element is `perItem` array entries. Appends write at
 * the current end; growth re-allocates (never mutating bytes an earlier
 * `view()` can see — see the module doc's aliasing contract).
 */
export class ArenaField<A extends ArenaTypedArray> {
  private buf: A;
  private usedElements = 0;
  private readonly ctor: new (n: number) => A;
  readonly perItem: number;
  /** Copy-work accounting (exposed for the O(N_k)-per-append tests). */
  readonly stats: ArenaFieldStats = { appendedEntries: 0, reallocCopiedEntries: 0 };

  constructor(ctor: new (n: number) => A, perItem: number, initialCapacityElements = 0) {
    this.ctor = ctor;
    this.perItem = perItem;
    this.buf = new ctor(initialCapacityElements * perItem);
  }

  /** The field's element constructor (dtype identity for ladder checks). */
  get elementCtor(): new (n: number) => A {
    return this.ctor;
  }

  /** Elements currently written. */
  get lengthElements(): number {
    return this.usedElements;
  }

  /** Elements the current buffer can hold. */
  get capacityElements(): number {
    return this.buf.length / this.perItem;
  }

  /**
   * Ensure capacity for `totalElements`. Growth allocates
   * `max(needed, ceil(capacity × ARENA_GROWTH_FACTOR))` entries — or exactly
   * `needed` when `exact` (the ladder's final level, where headroom would be
   * pure waste). Re-allocation copies only the used prefix; the old buffer
   * (and any views over it) is left untouched.
   */
  ensureCapacity(totalElements: number, exact = false): void {
    const needed = totalElements * this.perItem;
    if (this.buf.length >= needed) return;
    const grown = exact
      ? needed
      : Math.max(needed, Math.ceil(this.buf.length * ARENA_GROWTH_FACTOR));
    this.realloc(grown);
  }

  private realloc(entries: number): void {
    const next = new this.ctor(entries);
    next.set(this.buf.subarray(0, this.usedElements * this.perItem) as ArrayLike<number>, 0);
    this.stats.reallocCopiedEntries += this.usedElements * this.perItem;
    this.buf = next;
  }

  /**
   * Append `elements` elements copied from `src` (same `TypedArray.set`
   * semantics as the reference concat: `src.length` entries are written).
   */
  append(src: ArrayLike<number>, elements: number): void {
    this.ensureCapacity(this.usedElements + elements);
    this.buf.set(src, this.usedElements * this.perItem);
    this.stats.appendedEntries += src.length;
    this.usedElements += elements;
  }

  /**
   * Append `elements` elements via a custom writer (lines segment-index
   * offsetting). `write` receives the raw buffer and the base ENTRY index.
   */
  appendWith(elements: number, write: (buf: A, baseEntry: number) => void): void {
    this.ensureCapacity(this.usedElements + elements);
    write(this.buf, this.usedElements * this.perItem);
    this.stats.appendedEntries += elements * this.perItem;
    this.usedElements += elements;
  }

  /** Append `elements` elements of a constant fill value (white/default fill). */
  appendFill(value: number, elements: number): void {
    this.ensureCapacity(this.usedElements + elements);
    const base = this.usedElements * this.perItem;
    this.buf.fill(value, base, base + elements * this.perItem);
    this.stats.appendedEntries += elements * this.perItem;
    this.usedElements += elements;
  }

  /**
   * Release over-allocation once the ladder is complete: re-allocates to
   * exact size when slack exceeds {@link ARENA_TRIM_SLACK_FRACTION} of the
   * used entries. Earlier views keep the old (still-correct) buffer.
   */
  trimToFit(): void {
    const used = this.usedElements * this.perItem;
    const slack = this.buf.length - used;
    if (slack <= 0) return;
    if (used > 0 && slack / used <= ARENA_TRIM_SLACK_FRACTION) return;
    this.realloc(used);
  }

  /**
   * A `subarray(0, used)` view of the current contents. A NEW view object
   * per call (memoized-result identity stays per-lodCount), always at
   * byteOffset 0. Later appends write beyond this view's length; growth
   * re-allocates — either way this view's contents never change.
   */
  view(): A {
    return this.buf.subarray(0, this.usedElements * this.perItem) as A;
  }
}

// ---------------------------------------------------------------------------
// Shared ladder validation + optional-field machinery
// ---------------------------------------------------------------------------

/**
 * Ladder-dtype fail-fast against an established constructor — the
 * incremental twin of `concatRequiredField`'s check, with its exact message
 * (level indices are ABSOLUTE ladder levels; only `parts[from..)` with
 * `i ≥ 1` are checked — level 0 defines the ctor).
 */
export function validateLadderFieldDtype<P, A extends ArenaTypedArray>(
  parts: P[],
  from: number,
  get: (p: P) => A,
  ctor: new (n: number) => A,
  label: string
): void {
  for (let i = Math.max(from, 1); i < parts.length; i++) {
    const other = get(parts[i]).constructor;
    if (other !== ctor) {
      throw new Error(
        `concatRequiredField: LOD level ${i} carries '${label}' as ` +
          `${(other as { name?: string }).name} but level 0 uses ` +
          `${(ctor as { name?: string }).name} — ladder levels must share each ` +
          "field's dtype (TypedArray.set converts by value, not semantics)."
      );
    }
  }
}

/** {@link OptionalLadderField.plan} outcome. */
export type OptionalFieldPlan<A extends ArenaTypedArray> =
  | 'skip'
  | 'drop'
  | { ctor: new (n: number) => A };

/**
 * An ALL-OR-NOTHING optional ladder field (the incremental twin of
 * `concatOptionalField`): produced only while EVERY level carries the
 * attribute; the first level without it drops the merged field for the rest
 * of the generation. `plan()` (throwing validation, no mutation) and
 * `apply()` (non-throwing mutation) are split so a loader arena can validate
 * its whole batch before writing anything.
 */
export class OptionalLadderField<A extends ArenaTypedArray> {
  field: ArenaField<A> | null = null;
  /** Some level lacked the attribute — the merged field is dropped for good. */
  dropped = false;

  /**
   * Decide the field's fate for a batch: `skip` (already dropped), `drop`
   * (a batch level lacks it), or `{ctor}` (all carry it; dtype validated
   * against the established ctor with the reference message).
   */
  plan<P>(
    parts: P[],
    from: number,
    get: (p: P) => A | null | undefined,
    label: string
  ): OptionalFieldPlan<A> {
    if (this.dropped) return 'skip';
    for (let i = from; i < parts.length; i++) {
      if (get(parts[i]) == null) return 'drop';
    }
    // Field null + not dropped + from > 0 cannot happen (the first batch
    // starts at 0 and either creates the field or drops it), so parts[0]
    // is a carrier whenever it is consulted here.
    const ctor = (this.field?.elementCtor ?? (get(parts[0]) as A).constructor) as new (
      n: number
    ) => A;
    validateLadderFieldDtype(parts, from, (p) => get(p) as A, ctor, label);
    return { ctor };
  }

  /** Apply a {@link plan} decision (create/grow or drop the arena field). */
  apply(plan: OptionalFieldPlan<A>, perItem: number, newTotal: number, isFinal: boolean): void {
    if (plan === 'skip') return;
    if (plan === 'drop') {
      this.field = null;
      this.dropped = true;
      return;
    }
    if (!this.field) {
      this.field = new ArenaField<A>(plan.ctor, perItem, newTotal);
    } else {
      this.field.ensureCapacity(newTotal, isFinal);
    }
  }
}
