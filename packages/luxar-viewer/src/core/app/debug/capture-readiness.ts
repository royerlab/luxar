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
 */
export interface CaptureReadinessSummary {
  /**
   * `true` iff `totalElements` is greater than zero — i.e. the scene graph
   * carries drawable elements. HIDDEN nodes count, and every level of a
   * substitutive `kind=lod` group counts, exactly as in {@link DebugState}; so
   * `ok: true` does not guarantee a non-blank screenshot.
   */
  ok: boolean;
  /** Why the scene is not ready. Present only when `ok` is false. */
  reason?: string;
  /** Drawn points summed over all point-cloud nodes. */
  totalPoints: number;
  /** Drawn splats summed over all gsplat nodes. */
  totalGSplats: number;
  /** Drawn line segments summed over all line nodes. */
  totalLines: number;
  /** Drawn triangles (current draw range) summed over all mesh nodes. */
  totalTriangles: number;
  /** Sum of the four totals above — the readiness number. */
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

/** Finite-number coercion: anything else (undefined, NaN, a string) reads as 0. */
function finiteOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
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
 * re-derived from the per-type ones, and a present-but-stale total can only
 * under-claim relative to itself, never contradict them. (Trusting the
 * snapshot's own field outright let a snapshot carrying
 * `totalTriangles: 1200, totalElements: 0` report a fully-loaded mesh scene as
 * empty — the very symptom of royerlab/luxar#1579.)
 *
 * @param state The object returned by `window.__luxarDebug.getState()`, or
 *   `null`/`undefined` when the debug interface was never installed (the URL
 *   lacked `?debug`).
 * @returns A plain summary; `ok` is false with a `reason` for every
 *   not-ready case (no state, no totals at all, non-finite totals, empty
 *   scene). Note that `ok: true` means the scene graph carries elements, not
 *   that they are visible — see the module doc.
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
  const rawTotals = [
    state.totalElements,
    state.totalPoints,
    state.totalGSplats,
    state.totalLines,
    state.totalTriangles,
  ];
  const hasAnyTotal = rawTotals.some(isNumber);
  const hasNonFiniteTotal = rawTotals.some((value) => isNumber(value) && !Number.isFinite(value));
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
    return { ok: true, ...totals, ...counts };
  }

  return {
    ok: false,
    reason: hasNonFiniteTotal
      ? 'element totals are present but not finite (Infinity/NaN) — counted as zero, nothing usable to capture'
      : 'zero points, gsplats, lines and triangles — nothing loaded',
    ...totals,
    ...counts,
  };
}
