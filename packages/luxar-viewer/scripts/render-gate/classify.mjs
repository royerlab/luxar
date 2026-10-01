/**
 * Exactness row classification: scores in, status out. Pure (no I/O), so the
 * gate's decision table is unit-testable without a browser; `run-gate.mjs`
 * captures, scores, writes heatmaps and calls `classifyRow` for the verdict.
 */

import { differingBBox, judge, scorePickBuffers } from './exactness.mjs';

/** Whether two drawn-element count maps agree key for key. */
export function sameCounts(a, b) {
  const ka = Object.keys(a).sort();
  const kb = Object.keys(b).sort();
  return ka.length === kb.length && ka.every((k, i) => k === kb[i] && a[k] === b[k]);
}

/** A score without its per-pixel map, for the JSON report. */
export function strip(score) {
  const rest = { ...score };
  delete rest.perPixelUlp;
  return rest;
}

/**
 * Score the A/A pair of pick buffers. Neither capture having one is `null` (the
 * case does not pick); two buffers of different length, or only one buffer, is
 * recorded as a `lengthMismatch` rather than dropped: it is a stronger sign of
 * nondeterminism than differing ids.
 *
 * @param {Float32Array | null} first Pick buffer of the first baseline capture.
 * @param {Float32Array | null} second Pick buffer of the second baseline capture.
 */
export function scoreControlPick(first, second) {
  if (!first && !second) return null;
  if (!first || !second || first.length !== second.length) {
    return { lengthMismatch: true, first: first?.length ?? null, second: second?.length ?? null };
  }
  return scorePickBuffers(first, second);
}

/** Whether an A/A pick score shows the baseline disagreeing with itself. */
function pickDiffers(aaPick) {
  return !!aaPick && (aaPick.lengthMismatch === true || aaPick.mismatches > 0);
}

/** An A/A float score for the report, with the bounding box of its differing pixels. */
function controlScore(score, width, height) {
  return { ...strip(score), bbox: differingBBox(score.perPixelUlp, width, height) };
}

function excluded({ a, a2, aa, aaLdr, aaPick }) {
  const pickNote = !pickDiffers(aaPick)
    ? ''
    : aaPick.lengthMismatch
      ? ', pick buffer sizes differ'
      : `, ${aaPick.mismatches} pick px`;
  return {
    status: 'excluded',
    failures: [
      `nondeterministic baseline (A/A differs on ${aa.differing} HDR px, ${aaLdr.differing} LDR px${pickNote})`,
    ],
    control: {
      hdr: controlScore(aa, a.width, a.height),
      ldr: controlScore(aaLdr, a.width, a.height),
      pick: aaPick,
      counts: { first: a.counts, second: a2.counts },
      camera: { first: a.camera, second: a2.camera },
      stable: { first: a.stable, second: a2.stable },
      bufferStability: { first: a.bufferStability, second: a2.bufferStability },
    },
  };
}

/**
 * Classify one exactness view.
 *
 * @param {object} input
 * @param {object} input.a First baseline shot.
 * @param {object} input.a2 Second baseline shot (the A/A control).
 * @param {object} input.b Candidate shot.
 * @param {object} input.aa A/A HDR score (with `perPixelUlp`).
 * @param {object} input.aaLdr A/A LDR score (with `perPixelUlp`).
 * @param {object | null} input.aaPick A/A pick score from `scoreControlPick`.
 * @param {object} input.hdr Base/candidate HDR score.
 * @param {object} input.ldr Base/candidate LDR score.
 * @param {object | null} input.pick Base/candidate pick score.
 * @param {object} input.blocks Base/candidate block score.
 * @param {string} input.cls Verdict class (`IDENTICAL`, `ULP` or `INTENDED`).
 * @param {boolean} input.requiresPick Whether the case must carry a pick buffer.
 * @returns {{ status: string, failures?: string[], control?: object, counts?: object }}
 */
export function classifyRow(input) {
  const { a, a2, b, aa, aaLdr, aaPick, hdr, ldr, pick, blocks, cls, requiresPick } = input;
  if (!a.settled || !b.settled || !a2.settled) {
    return { status: 'error', failures: ['did not settle'] };
  }
  if (!(hdr.peak > 0)) {
    // An empty frame matches an empty frame and certifies nothing.
    return { status: 'error', failures: ['baseline frame is empty (nothing on screen)'] };
  }
  if (requiresPick && (!pick || pick.hits === 0)) {
    return {
      status: 'error',
      failures: [pick ? 'pick buffer carries no ids' : 'pick buffer not captured'],
    };
  }
  if (!a.stable || !a2.stable || aa.differing > 0 || aaLdr.differing > 0 || pickDiffers(aaPick)) {
    return excluded(input);
  }
  if (!sameCounts(a.counts, b.counts)) {
    return {
      status: cls === 'INTENDED' ? 'changed' : 'fail',
      failures: ['drawn element counts differ between builds'],
      counts: { base: a.counts, cand: b.counts },
    };
  }
  if (cls === 'INTENDED') {
    return { status: hdr.differing > 0 || ldr.differing > 0 ? 'changed' : 'unchanged' };
  }
  const verdict = judge(cls, hdr, ldr, pick, blocks);
  return { status: verdict.pass ? 'pass' : 'fail', failures: verdict.failures };
}
