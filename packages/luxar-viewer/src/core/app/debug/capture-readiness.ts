/**
 * Readiness verdict over a `__luxarDebug.getState()` snapshot — "does this
 * scene graph carry drawable elements?".
 *
 * WHAT IT DOES **NOT** MEASURE: whether the framebuffer will contain pixels.
 * The {@link DebugState} totals deliberately include HIDDEN nodes (see the
 * `totalLines` / `totalTriangles` field docs there) and sum EVERY level of a
 * substitutive `kind=lod` group, not just the active one. So an all-hidden
 * scene, or one whose only content is an inactive LOD level, reports
 * `ok: true` and the screenshot can still be blank. That is intentional: the
 * verdict mirrors `debug-state.ts`'s aggregate contract rather than
 * re-deciding visibility here, so the two can never disagree. Filter on the
 * per-node `visible` flags in the snapshot if you need the stricter question.
 *
 * WHY THIS IS A MODULE AND NOT THREE LINES INSIDE `page.evaluate`:
 * `tools/capture-hires.ts` used to compute the verdict inside the browser
 * closure, where nothing can unit-test it — and it was WRONG for every scene
 * ever captured. It read the totals from a `state.performance` sub-object that
 * `debug-state.ts` has never produced ({@link DebugState} is FLAT), so `perf`
 * was always `{}`, every total was `undefined`, and `undefined > 0` made `ok`
 * false for points, lines, gsplats and mesh alike. The bug was invisible
 * because `JSON.stringify` drops `undefined` keys, so the printed diagnostics
 * simply omitted the totals rather than showing `undefined`
 * (royerlab/luxar#1579).
 *
 * So the verdict lives OUT here, as a pure function over a plain object: the
 * browser side only has to hand back `getState()` verbatim. Deliberately
 * dependency-free — no THREE, no browser globals — so it is importable both
 * from a vitest suite and from a `tsx`-run Node tool.
 *
 * Every field is read defensively (finite-number / `Array.isArray` guards)
 * because the input crosses a `page.evaluate` serialisation boundary and may
 * come from an older viewer build: a partial snapshot must yield `ok: false`
 * with a `reason`, never a leaked `NaN` or `undefined`.
 *
 * @module core/app/debug/capture-readiness
 */

import type { DebugState } from './debug-state';

/**
 * JSON-serialisable readiness summary — what a capture tool logs and branches
 * on. Reports all FOUR geometry types: a mesh-only or lines-only scene is as
 * legitimately "loaded" as a points one, and naming only points/gsplats (as the
 * old inline read did) makes a fully-populated mesh scene read as empty.
 *
 * `reason` is NOT exclusive to `ok: false`. A scene can be ready AND carry a
 * caveat about the numbers printed alongside it (a total that was present but
 * `Infinity`/`NaN` is counted as 0, so the counts under-state the scene). A
 * count the tool could not read must never print as a bare `0` with nothing
 * said about it — that silent zero is the class of bug #1579 was.
 */
export interface CaptureReadinessSummary {
  /**
   * `true` iff `totalElements` is greater than zero — i.e. the scene graph
   * carries drawable elements. HIDDEN nodes count, and every level of a
   * substitutive `kind=lod` group counts, exactly as in {@link DebugState}; so
   * `ok: true` does not guarantee a non-blank screenshot.
   */
  ok: boolean;
  /**
   * Why the verdict is not ok — or, on an otherwise-ready verdict, a caveat
   * about the reported numbers (currently: a present-but-non-finite total that
   * had to be counted as 0). Absent when the scene is ready and every total
   * read cleanly.
   */
  reason?: string;
  /** Drawn points summed over all point-cloud nodes. */
  totalPoints: number;
  /** Drawn splats summed over all gsplat nodes. */
  totalGSplats: number;
  /** Drawn line segments summed over all line nodes. */
  totalLines: number;
  /** Drawn triangles (current draw range) summed over all mesh nodes. */
  totalTriangles: number;
  /**
   * The readiness number: the maximum of the snapshot's own reported
   * `totalElements` field and the sum of the four per-type totals above.
   *
   * The `max` is a VERSION-SKEW hedge, not arithmetic. The tool talks to
   * whatever viewer build happens to be served at `APP_URL`, so the snapshot
   * may be stale or partial; taking the max means such a snapshot's own
   * `totalElements` can only under-claim relative to itself, never under-claim
   * against the per-type totals it is carrying (which is exactly how #1579
   * printed 1200 triangles next to "nothing loaded"). The current viewer sets
   * the field to precisely that sum (`debug-state.ts`), so on a live snapshot
   * the max is inert and this is simply the sum.
   */
  totalElements: number;
  /** Number of point-cloud nodes in the scene. */
  pointCloudCount: number;
  /** Number of gsplat nodes in the scene. */
  gsplatCount: number;
  /** Number of line nodes in the scene (`lineMeshes`). */
  lineCount: number;
  /** Number of mesh (triangle-surface) nodes in the scene (`meshNodes`). */
  meshNodeCount: number;
}

/**
 * Finite-number coercion, clamped at zero: anything that is not a finite number
 * (undefined, NaN, Infinity, a string) reads as 0, and so does a NEGATIVE one.
 *
 * The clamp is not cosmetic. An element count cannot be negative, and an
 * unclamped one both prints as a nonsense measurement and can CANCEL a real
 * positive in the sum — `{totalPoints: -1200, totalTriangles: 1200}` summed to
 * exactly 0 and reported "nothing loaded" over a populated mesh scene.
 */
function finiteOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? Math.max(0, value) : 0;
}

/**
 * `true` when the field is PRESENT as a number — finite or not. Used to tell a
 * shape mismatch (the total is missing entirely) from a present-but-unusable
 * total (`Infinity`, `NaN`), which deserve different reasons.
 */
function isNumber(value: unknown): value is number {
  return typeof value === 'number';
}

/** Length of an array-valued field, 0 when the field is absent or not an array. */
function lengthOrZero(value: unknown): number {
  return Array.isArray(value) ? value.length : 0;
}

/**
 * Summarize a `getState()` snapshot into a capture-readiness verdict.
 *
 * Reads the totals from the FLAT {@link DebugState} shape — `state.totalPoints`,
 * not `state.performance.totalPoints`. `totalElements` is
 * `max(snapshot total, sum of the four per-type totals)`: a missing total is
 * re-derived from the per-type ones, and a present-but-stale total (the tool
 * captures against whatever viewer build is served, so skew is possible) can
 * only under-claim relative to itself, never contradict them. On a current
 * viewer `debug-state.ts` sets the field to exactly that sum, so the `max` is
 * inert and this is just the sum. (Trusting the
 * snapshot's own field outright let a snapshot carrying
 * `totalTriangles: 1200, totalElements: 0` report a fully-loaded mesh scene as
 * empty — the very symptom of royerlab/luxar#1579.)
 *
 * @param state The object returned by `window.__luxarDebug.getState()`, or
 *   `null`/`undefined` when the debug interface was never installed (the URL
 *   lacked `?debug`).
 * @returns A plain summary; `ok` is false with a `reason` for every
 *   not-ready case (no state, no totals at all, non-finite totals, empty
 *   scene). A READY verdict can carry a `reason` too — as a caveat, when some
 *   total was present but non-finite and therefore counted as 0. Note that
 *   `ok: true` means the scene graph carries elements, not that they are
 *   visible — see the module doc.
 */
export function summarizeCaptureReadiness(
  state: Partial<DebugState> | null | undefined
): CaptureReadinessSummary {
  const empty = {
    totalPoints: 0,
    totalGSplats: 0,
    totalLines: 0,
    totalTriangles: 0,
    totalElements: 0,
    pointCloudCount: 0,
    gsplatCount: 0,
    lineCount: 0,
    meshNodeCount: 0,
  };

  if (!state || typeof state !== 'object') {
    return {
      ok: false,
      reason: 'no debug state (viewer not opened with ?debug, or getState() unavailable)',
      ...empty,
    };
  }

  const totalPoints = finiteOrZero(state.totalPoints);
  const totalGSplats = finiteOrZero(state.totalGSplats);
  const totalLines = finiteOrZero(state.totalLines);
  const totalTriangles = finiteOrZero(state.totalTriangles);
  const summed = totalPoints + totalGSplats + totalLines + totalTriangles;
  const totalElements = Math.max(finiteOrZero(state.totalElements), summed);

  const counts = {
    pointCloudCount: lengthOrZero(state.pointClouds),
    gsplatCount: lengthOrZero(state.gsplatMeshes),
    lineCount: lengthOrZero(state.lineMeshes),
    meshNodeCount: lengthOrZero(state.meshNodes),
  };

  // A snapshot carrying none of the five count fields is a SHAPE mismatch, not
  // an empty scene — the exact failure #1579 hid behind a silent `undefined > 0`.
  // PRESENCE, not usability, is the test: a total that is present but
  // non-finite (`Infinity` / `NaN`) is not a shape mismatch and must not be
  // diagnosed as one — it falls through to the not-ready reason below.
  const totalFields = [
    'totalElements',
    'totalPoints',
    'totalGSplats',
    'totalLines',
    'totalTriangles',
  ] as const;
  const hasAnyTotal = totalFields.some((field) => isNumber(state[field]));
  // Names, not just a boolean: the caveat below has to say WHICH count it could
  // not read, or the caller is back to guessing which of the printed zeros is
  // real. Kept in declaration order so the message is stable.
  const nonFiniteFields = totalFields.filter(
    (field) => isNumber(state[field]) && !Number.isFinite(state[field])
  );
  if (!hasAnyTotal) {
    return {
      ok: false,
      reason: 'debug state carries no element totals (unexpected getState() shape)',
      ...empty,
      ...counts,
    };
  }

  const totals = {
    totalPoints,
    totalGSplats,
    totalLines,
    totalTriangles,
    totalElements,
  };

  if (totalElements > 0) {
    // Ready, but a total we could not read is still worth saying out loud: it is
    // reported as 0 above, and an unannounced 0 is indistinguishable from a
    // genuinely empty geometry type.
    if (nonFiniteFields.length > 0) {
      return {
        ok: true,
        reason:
          `${nonFiniteFields.join(', ')} present but not finite (Infinity/NaN) — ` +
          'counted as zero, so the reported counts under-state the scene',
        ...totals,
        ...counts,
      };
    }
    return { ok: true, ...totals, ...counts };
  }

  return {
    ok: false,
    reason:
      nonFiniteFields.length > 0
        ? `${nonFiniteFields.join(', ')} present but not finite (Infinity/NaN) — ` +
          'counted as zero, nothing usable to capture'
        : 'zero points, gsplats, lines and triangles — nothing loaded',
    ...totals,
    ...counts,
  };
}
