/**
 * Readiness verdict over a `__luxarDebug.getState()` snapshot — "does this
 * scene graph carry drawable elements, and are they the ones the store
 * describes?".
 *
 * THREE WAYS THE SECOND HALF FAILS, and all three are refusals here. A node can
 * be truncated by the renderer's per-node element-texture clamp
 * (`totalDroppedElements`); progressive refinement can stop scene-wide at the
 * residency byte ceiling (`refinementResidency`), leaving a partial scene whose
 * composition depends on which nodes were offered first; and the GPU buffer pool
 * can go over its VRAM byte budget and shed pooled geometry to get back under it
 * (`gpuPool.byteBudgetEvictions`). In every case the element counts stop being a
 * property of the STORE and become a property of the machine and the run — so a
 * capture is not comparable against another build's, and an A/B is measuring the
 * ceiling rather than the change (royerlab/luxar#2508). The residency stop used
 * to surface as one console warning that nothing downstream read, which made
 * "grep the console before comparing two builds" a convention instead of a gate.
 *
 * BOTH NEW SIGNALS ARE LOADER-SCOPED AND CUMULATIVE WITHIN THAT LIFE, AND THAT
 * IS DELIBERATE: each says "this happened at some point while this scene was
 * loaded", NOT "this is true right now", so a scene that stopped at the ceiling
 * and then refined fully after a view change is still refused. The contract in
 * full — why that is the verdict a capture tool wants, and why an in-page
 * dataset switch clears it without a reload — lives on `RefinementResidencyStop`
 * in `data/scene-loader/progressive/residency-budget.ts`.
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
 * come from an older viewer build. A per-type total that is missing — or
 * present but unusable — reads as 0 AND is named in `reason`, because an
 * unannounced 0 is indistinguishable from a genuinely empty geometry type; a
 * snapshot carrying NO recognisable total at all is reported as a shape
 * mismatch rather than as an empty scene. Either way the summary never leaks a
 * `NaN` or an `undefined`.
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
 * `Infinity`/`NaN`, or absent from the snapshot altogether, is counted as 0, so
 * the counts under-state the scene). A count the tool could not read must never
 * print as a bare `0` with nothing said about it — that silent zero is the class
 * of bug #1579 was.
 */
export interface CaptureReadinessSummary {
  /**
   * `true` iff `totalElements` is greater than zero and nothing made the counts
   * a property of this RUN rather than of the store: no elements dropped by
   * renderer capacity limits, no refinement stop at the residency ceiling, no
   * GPU-pool eviction on the byte budget. HIDDEN nodes count, and every level
   * of a substitutive `kind=lod` group counts, exactly as in {@link DebugState};
   * so `ok: true` does not guarantee a non-blank screenshot. A false verdict
   * can therefore mean "retry later", "loaded but truncated", or "loaded but
   * capped"; callers must use `reason` to distinguish them.
   */
  ok: boolean;
  /**
   * Why the verdict is not ok — dropped elements, a residency stop, a
   * byte-budget pool eviction, an empty scene, an unusable snapshot shape — or,
   * on an otherwise-ready verdict, a caveat about the reported numbers (a total
   * that had to be counted as 0 because it was non-finite, or because the
   * snapshot did not carry it at all). Concurrent causes are reported together,
   * `; `-joined, rather than the first one shadowing the rest. No clause can
   * contain that sequence — the residency clause separates its own parenthetical
   * fields with commas, the unreadable-totals caveat separates its two halves
   * the same way, and every snapshot-supplied string this module echoes has its
   * semicolons flattened first (see `echoable`) — so splitting on `'; '`
   * recovers exactly the causes and nothing else. Absent when the scene is ready
   * and every total read cleanly.
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
  /** Elements omitted by renderer capacity clamps. */
  totalDroppedElements: number;
  /**
   * Distinct node paths progressive refinement declined at the residency byte
   * ceiling. `0` when refinement never stopped — and also `0` when the snapshot
   * carried a stop whose count was unreadable, in which case the verdict is
   * still a refusal and `reason` says the count could not be read (presence of
   * the stop is the signal, not its magnitude).
   */
  refinementDeclinedPathCount: number;
  /**
   * Cumulative GPU buffer-pool evictions forced by the VRAM byte budget. `0`
   * when the pool never evicted on bytes, when the snapshot carries no pool
   * stats, or when the field was unreadable. Deliberately NOT the pool's total
   * `evictions`, which is dominated by routine LRU recycling on any nD scene
   * and would refuse every normal capture.
   */
  gpuByteBudgetEvictions: number;
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
 * An element count cannot be negative, and an unclamped one would both print as
 * a nonsense measurement and be able to CANCEL a real positive in the sum
 * (`{totalPoints: -1200, totalTriangles: 1200}` would sum to exactly 0 and read
 * as "nothing loaded" over a populated mesh scene). No producer emits one; this
 * is a boundary guard, not a fix for something observed.
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

/** Bytes as a one-decimal MiB string — the same spelling the viewer logs. */
function mib(value: number): string {
  return `${(value / (1024 * 1024)).toFixed(1)} MiB`;
}

/**
 * A whole, non-negative count: {@link finiteOrZero} with the fraction dropped.
 *
 * THE SINGLE COERCION for both cap counts, used by the reason strings AND by the
 * reported summary fields, so the two can never disagree. They did while the
 * message floored and the summary did not: `declinedPathCount: 0.5` printed
 * "declined-path count unreadable" beside a reported `0.5`, and
 * `byteBudgetEvictions: 0.5` refused a capture while announcing "0 GPU
 * buffer-pool evictions" — the exact self-contradiction the unreadable wording
 * exists to prevent.
 */
function wholeCount(value: unknown): number {
  return Math.floor(finiteOrZero(value));
}

/**
 * How many paths declined, in words — or an admission that the snapshot did not
 * say. Never a bare `0`: a stop that reports no paths is a contradiction, and
 * printing "0 paths declined" next to a refusal would read as a bug in the
 * verdict rather than as an unreadable field.
 */
function describeDeclinedPaths(count: unknown): string {
  const declined = wholeCount(count);
  if (declined <= 0) return 'declined-path count unreadable';
  return `${declined} node path${declined === 1 ? '' : 's'} declined`;
}

/**
 * A finite, non-negative number, or `undefined` for anything a message must not
 * print. Negatives are refused for the same reason {@link finiteOrZero} clamps
 * them: a byte figure cannot be negative, and printing
 * `resident -0.0 MiB of a -0.0 MiB budget` states a measurement that was never
 * taken. Unreadable is the honest word for it.
 */
function nonNegativeOrUndefined(value: unknown): number | undefined {
  return isNumber(value) && Number.isFinite(value) && value >= 0 ? value : undefined;
}

/** The ceiling the stop was measured against, or an admission that it was unreadable. */
function describeResidencyBytes(residentBytes: unknown, budgetBytes: unknown): string {
  const resident = nonNegativeOrUndefined(residentBytes);
  const budget = nonNegativeOrUndefined(budgetBytes);
  return resident !== undefined && budget !== undefined
    ? `resident ${mib(resident)} of a ${mib(budget)} budget`
    : 'resident and budget bytes unreadable';
}

/**
 * How much of a snapshot-supplied string this module will echo. The stop
 * `reason` is an enum-shaped token — the longest the viewer emits is
 * `next-rung-would-exceed`, 22 characters — so 40 leaves room for a newer
 * build's code while bounding what an arbitrary string can add. A node path gets
 * more, because printing it is only useful while it still identifies a subtree.
 */
const MAX_ECHOED_REASON_CHARS = 40;
const MAX_ECHOED_PATH_CHARS = 120;

/**
 * A snapshot-supplied string, made safe to quote inside a `reason`.
 *
 * The snapshot crosses a `page.evaluate` boundary and is NOT trusted input,
 * while `reason`'s concurrent causes are `'; '`-joined and read back by
 * splitting on that sequence. A value carrying it could therefore FORGE a
 * clause: `reason: 'a; 999 elements were dropped by renderer capacity limits'`
 * echoed verbatim reads to a split-based caller as a real dropped-elements
 * cause that no renderer clamp ever produced. Semicolons and whitespace runs
 * collapse to one space so no echoed value can spell the separator or break a
 * single-line log record, and the result is truncated so a pathological string
 * cannot swamp the message it is embedded in.
 */
function echoable(value: string, maxChars: number): string {
  const flattened = value.replace(/[\s;]+/g, ' ');
  return flattened.length > maxChars ? `${flattened.slice(0, maxChars)}…` : flattened;
}

/**
 * The FIRST refusal's admission verdict, in words.
 *
 * The two codes the reporter can record say genuinely different things —
 * `over-budget` means the scene was already past the ceiling when that rung was
 * offered, `next-rung-would-exceed` that it fitted but one more step would not —
 * and that is the difference between "author a coarser ladder" and "it very
 * nearly fitted". An unrecognised code (a newer viewer build) is quoted rather
 * than guessed at — through {@link echoable}, since quoting an untrusted string
 * verbatim is what lets it forge a clause — and a non-string is named
 * unreadable, same as every other field here.
 */
function describeStopReason(reason: unknown): string {
  if (reason === 'over-budget') return 'already past the ceiling';
  if (reason === 'next-rung-would-exceed') return 'the next rung would have crossed it';
  if (typeof reason === 'string' && reason.length > 0) {
    return `reason "${echoable(reason, MAX_ECHOED_REASON_CHARS)}"`;
  }
  return 'stop reason unreadable';
}

/**
 * Blocking reason for a progressive-refinement stop at the residency ceiling,
 * or `undefined` when the snapshot carries none.
 *
 * PRESENCE OF THE OBJECT IS THE SIGNAL. The viewer only sets the field once a
 * rung has actually been declined (`RefinementResidencyReporter.snapshot`
 * returns `undefined` otherwise), so a malformed one still means the scene
 * stopped short; the message then says which parts it could not read instead of
 * printing zeros or dropping the refusal. A field that is absent, `null`, a
 * primitive, or an array is not a stop at all and yields `undefined` — an older
 * viewer build simply does not carry it, and that must never read as trouble.
 *
 * The parenthetical's own fields are COMMA-separated on purpose. `'; '` is what
 * joins concurrent causes, so spelling it in here would split this one cause
 * into four fragments for any caller that reads `reason` by splitting — which
 * is exactly what it did until #2508's second review round.
 */
function residencyStopReason(stop: unknown): string | undefined {
  if (typeof stop !== 'object' || stop === null || Array.isArray(stop)) return undefined;
  const { reason, declinedPathCount, firstPath, residentBytes, budgetBytes } = stop as Record<
    string,
    unknown
  >;
  const first =
    typeof firstPath === 'string' && firstPath.length > 0
      ? echoable(firstPath, MAX_ECHOED_PATH_CHARS)
      : 'unknown';
  return (
    'progressive refinement stopped at the residency ceiling ' +
    `(${describeStopReason(reason)}, ${describeDeclinedPaths(declinedPathCount)}, ` +
    `first: ${first}, ${describeResidencyBytes(residentBytes, budgetBytes)}) — the counts ` +
    'describe a PARTIAL scene, and which nodes reached full detail is not deterministic'
  );
}

/**
 * Blocking reason for GPU buffer-pool eviction on the byte budget, or
 * `undefined` when none happened.
 *
 * The opposite rule to {@link residencyStopReason}: a pooled build attaches
 * `gpuPool` to every snapshot, so its presence carries no information and only a
 * POSITIVE, readable count refuses. An unreadable count therefore passes rather
 * than blocking every capture on a shape the viewer has never emitted. The two
 * guards legitimately differ on arrays for that same reason: a residency stop
 * refuses on PRESENCE, so it must exclude an array explicitly, whereas here an
 * array's missing `byteBudgetEvictions` already reads as 0 and falls out at the
 * count check.
 *
 * WHAT THE MESSAGE MAY AND MAY NOT CLAIM. The byte pass disposes POOLED
 * (already-released) buffers only; active in-use buffers are never candidates
 * (`gpu-buffer-pool/byte-budget-evictor.ts`). So an eviction here does not by
 * itself prove anything left the screen — it proves the pool went over its VRAM
 * budget and had to shed geometry to get back under, on this machine, in this
 * run. That still matters, and not only as a memory-pressure hint: the LOD
 * registry demotes cold ACTIVE levels to pooled, and this pass then reclaims
 * them, which is how committed detail actually does get thrown away. Either way
 * the run's residency is a property of the machine rather than of the store, and
 * the capture is not comparable.
 *
 * THE REFUSAL IS DELIBERATELY CONSERVATIVE, AND STAYS SO EVEN OVER THE KNOWN
 * BENIGN TRIGGER `PoolStats.byteBudgetEvictions` records (a node growing through
 * a capacity tier, whose superseded smaller buffer the byte pass then reclaims).
 * The asymmetry is the point: a false refusal costs one warning line in
 * `tools/capture-hires.ts`, which takes the screenshot regardless, whereas a
 * false pass is exactly the #2508 defect — a truncated capture believed by every
 * A/B downstream. So the conservative direction is the right one.
 */
function poolEvictionReason(gpuPool: unknown): string | undefined {
  if (typeof gpuPool !== 'object' || gpuPool === null) return undefined;
  const evictions = wholeCount((gpuPool as Record<string, unknown>).byteBudgetEvictions);
  if (evictions <= 0) return undefined;
  const event = evictions === 1 ? 'eviction' : 'evictions';
  return (
    `${evictions} GPU buffer-pool ${event} on the VRAM byte budget — the pool went over ` +
    'its budget and shed pooled geometry to get back under it, including any levels the LOD ' +
    'registry had demoted from active, so what stayed resident depends on this machine and ' +
    'this run (it does not on its own prove rendered geometry was lost)'
  );
}

/**
 * The two run-dependent cap counts, as numbers safe to print.
 *
 * Both go through {@link wholeCount}, which also absorbs a
 * `refinementResidency` / `gpuPool` that arrived as a primitive or `null`
 * (`(5)?.x` is `undefined`, not a throw). An unreadable field therefore reports
 * as 0 HERE while the reason strings above carry the "could not read it" —
 * splitting the two is what lets a stop with a garbled count still refuse.
 */
function capCounts(state: Partial<DebugState>): {
  refinementDeclinedPathCount: number;
  gpuByteBudgetEvictions: number;
} {
  return {
    refinementDeclinedPathCount: wholeCount(state.refinementResidency?.declinedPathCount),
    gpuByteBudgetEvictions: wholeCount(state.gpuPool?.byteBudgetEvictions),
  };
}

/**
 * Join the blocking reasons that apply, in the order given, or `undefined` when
 * none do. Concurrent causes are reported together: each says something
 * different about how the counts are wrong, and dropping all but the first
 * would send a caller chasing one cap while another still bites.
 */
function joinReasons(reasons: readonly (string | undefined)[]): string | undefined {
  const present = reasons.filter((reason): reason is string => !!reason);
  return present.length > 0 ? present.join('; ') : undefined;
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
 *   scene, renderer-truncated scene, a refinement stop at the residency
 *   ceiling, a byte-budget GPU-pool eviction). A READY verdict can carry a `reason` too
 *   — as a caveat, when some per-type total was counted as 0 because it was
 *   non-finite or absent from the snapshot rather than because the scene has
 *   none of that type. Note that `ok: true` means the scene graph carries
 *   elements without renderer truncation, not that they are visible — see the
 *   module doc.
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
    totalDroppedElements: 0,
    refinementDeclinedPathCount: 0,
    gpuByteBudgetEvictions: 0,
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
  const totalDroppedElements = finiteOrZero(state.totalDroppedElements);

  const counts = {
    pointCloudCount: lengthOrZero(state.pointClouds),
    gsplatCount: lengthOrZero(state.gsplatMeshes),
    lineCount: lengthOrZero(state.lineMeshes),
    meshNodeCount: lengthOrZero(state.meshNodes),
  };

  // Reported on every verdict, including the shape-mismatch one below: all
  // THREE run-dependent measurements — dropped elements, the residency stop, the
  // byte-budget eviction — describe the RUN, so they stay meaningful even when
  // the totals do not. Both the numbers AND their reason clauses: a snapshot too
  // skewed to carry totals is exactly the one whose caps a caller cannot
  // otherwise find out about, and reporting the count while dropping the
  // sentence that explains it was the half-measure this comment used to
  // describe. `totalDroppedElements` was the last one left behind on that path.
  const caps = capCounts(state);
  const capReason = joinReasons([
    residencyStopReason(state.refinementResidency),
    poolEvictionReason(state.gpuPool),
  ]);
  const droppedReason =
    totalDroppedElements > 0
      ? `${totalDroppedElements} elements were dropped by renderer capacity limits`
      : undefined;

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
  // A per-type total the snapshot never carried is exactly as unreadable as a
  // non-finite one — it prints as 0 all the same — so it earns the same caveat
  // rather than a silent zero. Only reachable through the version skew the
  // `Math.max` above already hedges against: the current viewer emits all five
  // totals unconditionally, so this list is empty on a live snapshot.
  // `totalElements` is excluded: it is re-derived from the per-type totals, so
  // its absence costs nothing and is not a count anyone lost.
  const absentFields = totalFields.filter(
    (field) => field !== 'totalElements' && !isNumber(state[field])
  );
  if (!hasAnyTotal) {
    return {
      ok: false,
      // The shape mismatch is the VERDICT and leads; the run-dependent causes
      // follow in their documented order (dropped, then the caps), the same
      // order `blockingReason` uses below.
      reason: joinReasons([
        'debug state carries no element totals (unexpected getState() shape)',
        droppedReason,
        capReason,
      ]),
      ...empty,
      totalDroppedElements,
      ...caps,
      ...counts,
    };
  }

  const totals = {
    totalPoints,
    totalGSplats,
    totalLines,
    totalTriangles,
    totalElements,
    totalDroppedElements,
  };

  // Every way the counts can be a property of the run rather than of the store,
  // as ONE string. Dropped elements stay first so a snapshot carrying only that
  // one produces exactly the message it always has — pinned by an exact-order
  // assertion in the test suite, since a reordering is otherwise invisible.
  const blockingReason = joinReasons([droppedReason, capReason]);

  /**
   * The counts this call had to give up on, named: `<fields> present but not
   * finite (Infinity/NaN), <fields> absent from the snapshot`, or `undefined`
   * when every total read cleanly. Each caller appends its own consequence
   * clause, since "counted as zero" means something different either side of
   * the readiness verdict.
   *
   * The two halves are COMMA-joined for the same reason `residencyStopReason`
   * comma-separates its parenthetical: this is ONE caveat clause, and `'; '` is
   * what joins concurrent causes. Spelling the separator here split a snapshot
   * carrying both a non-finite and an absent total into two fragments for any
   * caller reading `reason` by splitting, which is the guarantee documented on
   * {@link CaptureReadinessSummary.reason}.
   */
  const unreadable = (nonFinite: readonly string[]): string | undefined => {
    const parts: string[] = [];
    if (nonFinite.length > 0) {
      parts.push(`${nonFinite.join(', ')} present but not finite (Infinity/NaN)`);
    }
    if (absentFields.length > 0) {
      parts.push(`${absentFields.join(', ')} absent from the snapshot`);
    }
    return parts.length > 0 ? parts.join(', ') : undefined;
  };

  if (totalElements > 0) {
    // Ready, but a total we could not read is still worth saying out loud: it is
    // reported as 0 above, and an unannounced 0 is indistinguishable from a
    // genuinely empty geometry type. Name only the PER-TYPE fields here —
    // `totalElements` is re-derived from them (`Math.max` above), so saying it
    // was "counted as zero" next to a positive printed value would be a lie. In
    // practice the per-type and the aggregate go non-finite together, since the
    // viewer computes the field as the sum.
    const understated = unreadable(nonFiniteFields.filter((field) => field !== 'totalElements'));
    if (understated) {
      return {
        ok: !blockingReason,
        reason: [
          blockingReason,
          `${understated} — counted as zero, so the reported counts under-state the scene`,
        ]
          .filter(Boolean)
          .join('; '),
        ...totals,
        ...caps,
        ...counts,
      };
    }
    if (nonFiniteFields.length > 0) {
      return {
        ok: !blockingReason,
        reason: [
          blockingReason,
          'totalElements present but not finite (Infinity/NaN) — re-derived from the per-type totals',
        ]
          .filter(Boolean)
          .join('; '),
        ...totals,
        ...caps,
        ...counts,
      };
    }
    if (blockingReason) {
      return { ok: false, reason: blockingReason, ...totals, ...caps, ...counts };
    }
    return { ok: true, ...totals, ...caps, ...counts };
  }

  const lost = unreadable(nonFiniteFields);
  return {
    ok: false,
    reason: [
      blockingReason,
      lost
        ? `${lost} — counted as zero, nothing usable to capture`
        : 'zero points, gsplats, lines and triangles — nothing loaded',
    ]
      .filter(Boolean)
      .join('; '),
    ...totals,
    ...caps,
    ...counts,
  };
}
