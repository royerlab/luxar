/**
 * Playback aspiration: during dimension playback, cap each LOD group at the
 * finest level whose measured reload fits the playback period.
 *
 * A playing timelapse bumps the view version every period, so a lazy level
 * that cannot reload within it would be stale on every frame and drop the
 * display to the coarsest fresh level. The finest level that CAN keep up is
 * what a video player's "auto quality" picks. This module owns the
 * measurement and the decision:
 *
 * - **Load timing** ({@link foldLoadTime}): each timed (re)load of a lazy level
 *   is folded into an EWMA (``LODGroupChild.loadEwmaMs``). The first, cold
 *   sample is replaced by the second; only a FULL load or reload is a timing —
 *   the registry starts a ladder refinement step untimed.
 * - **Admission with hysteresis** ({@link PlaybackAspiration.aspire}): a level
 *   finer than the aspiration must fit ``playbackLoadBudgetFraction`` of the
 *   period; the aspiration and the levels below it only need
 *   ``playbackKeepBudgetFraction``.
 * - **Probe**: a capped level is re-measured once per
 *   ``playbackProbeIntervalMs``; the probe's reload REPLACES the stale average.
 *   The replace mark (``playbackProbePending``) is set only when the probe's
 *   reload actually starts ({@link PlaybackAspiration.noteTimedLoadStarted}),
 *   and every mark is forgotten when playback stops.
 *
 * Holds no per-frame tick demand: a playing timeline ticks every period, and a
 * reload in flight keeps the loop alive through the registry's loading walk.
 *
 * @module scene/playback-aspiration
 */

import { config } from '../config';

/** The slice of a LOD-group child playback reads and writes. */
export interface PlaybackChild {
  ensureLoaded?: () => void;
  loading?: boolean;
  failed?: boolean;
  loadStartMs?: number;
  loadEndMs?: number;
  loadEwmaMs?: number;
  loadSamples?: number;
  lastPlaybackProbeMs?: number;
  playbackProbePending?: boolean;
  playbackProbeAdmissionPending?: boolean;
}

/** The slice of a LOD-group entry playback reads. */
export interface PlaybackEntry {
  children: PlaybackChild[];
  activeChildIndex: number;
}

/**
 * Fold a finished (``loading`` cleared) timed load of ``c`` into
 * ``loadEwmaMs``: start to its stamped end, on the registry clock that stamped
 * the start. Without an end stamp (a thunk that never calls ``onLoadSettled``)
 * the load is timed to ``now()`` — the first moment it was observed finished.
 */
export function foldLoadTime(c: PlaybackChild, now: () => number): void {
  if (c.loadStartMs === undefined || c.loading === true) return;
  if (c.failed !== true) {
    const ms = Math.max(0, (c.loadEndMs ?? now()) - c.loadStartMs);
    const samples = c.loadSamples ?? 0;
    // The cold first sample seeds nothing: the second replaces it outright.
    // A playback probe replaces it too: the level has not been reloaded for
    // a second or more, so its average describes a cache state that is gone
    // (cold loads the first loop paid, while the cache has warmed since).
    c.loadEwmaMs =
      c.loadEwmaMs === undefined || samples < 2 || c.playbackProbePending === true
        ? ms
        : c.loadEwmaMs + config.lod.loadEwmaAlpha * (ms - c.loadEwmaMs);
    c.loadSamples = samples + 1;
  }
  c.playbackProbePending = undefined;
  c.loadStartMs = undefined;
  c.loadEndMs = undefined;
}

/** Per-registry playback state: the playing edge and this frame's probe. */
export class PlaybackAspiration {
  private wasPlaying = false;
  /**
   * The level the current entry's probe chose this frame, so the kick that
   * actually starts its reload can mark it (``playbackProbePending``).
   */
  private probeCandidate: PlaybackChild | null = null;

  /**
   * Start of an evaluated frame: on the playback FALLING edge forget every
   * probe mark, so a later unrelated load (a paused refinement, a preload)
   * blends into the average instead of replacing it, and the next playback
   * starts without a stale admission.
   */
  beginFrame(periodMs: number | null, entries: Iterable<PlaybackEntry>): void {
    const playing = periodMs !== null;
    if (this.wasPlaying && !playing) {
      for (const entry of entries) {
        for (const c of entry.children) {
          c.playbackProbePending = undefined;
          c.playbackProbeAdmissionPending = undefined;
        }
      }
    }
    this.wasPlaying = playing;
  }

  /** End of an evaluated frame (or of one entry): no probe is pending a kick. */
  endEntry(): void {
    this.probeCandidate = null;
  }

  /**
   * During playback, the finest level at or below ``desired`` that can reload
   * within the admission budget of ``periodMs``: an eager level (no
   * ``ensureLoaded`` — the per-slice sweep carries it), an unmeasured one, or
   * one whose ``loadEwmaMs`` fits. Once per probe interval, the next finer
   * capped level is probed for a warm-load sample. ``desired`` is unchanged
   * when not playing (``periodMs`` null).
   */
  aspire(entry: PlaybackEntry, desired: number, periodMs: number | null, nowMs: number): number {
    this.probeCandidate = null;
    if (periodMs === null) return desired;
    const affordable = affordablePlaybackLevel(entry, desired, periodMs);
    return this.probedLevel(entry, desired, affordable, nowMs);
  }

  /**
   * A timed (re)load of ``c`` just started: when it is this frame's probe, its
   * timing replaces the level's average.
   */
  noteTimedLoadStarted(c: PlaybackChild): void {
    if (c === this.probeCandidate) c.playbackProbePending = true;
  }

  clear(): void {
    this.wasPlaying = false;
    this.probeCandidate = null;
  }

  /**
   * Periodically re-measure the next finer capped level after a cold load. The
   * probe's reload replaces the level's average (``playbackProbePending``,
   * set when that reload starts): the samples it holds are at least a probe
   * interval old.
   */
  private probedLevel(
    entry: PlaybackEntry,
    desired: number,
    affordable: number,
    nowMs: number
  ): number {
    const next = entry.children[affordable + 1];
    if (affordable >= desired || next?.loadEwmaMs === undefined) return affordable;
    const last = next.lastPlaybackProbeMs ?? Number.NEGATIVE_INFINITY;
    if (nowMs - last < config.lod.playbackProbeIntervalMs) return affordable;
    next.lastPlaybackProbeMs = nowMs;
    // The admission mark guards the keep budget of the level the aspiration
    // just moved to; the replace mark waits for the reload to start.
    next.playbackProbeAdmissionPending = true;
    this.probeCandidate = next;
    return affordable + 1;
  }
}

/**
 * Finest level whose measured reload fits this playback period. A level
 * with only its cold first load measured (``loadSamples`` < 2) counts as
 * unmeasured, so one cold sample cannot demote it; a level whose warm
 * reloads are too slow is demoted after its second load. A level finer than
 * the aspiration must fit ``playbackLoadBudgetFraction`` of the period to be
 * admitted; the aspiration and the levels below it only need
 * ``playbackKeepBudgetFraction`` (hysteresis).
 */
function affordablePlaybackLevel(entry: PlaybackEntry, desired: number, periodMs: number): number {
  const { playbackKeepBudgetFraction, playbackLoadBudgetFraction } = config.lod;
  for (let i = desired; i > 0; i--) {
    const c = entry.children[i];
    const ewma = (c.loadSamples ?? 0) >= 2 ? c.loadEwmaMs : undefined;
    const fraction =
      i <= entry.activeChildIndex && c.playbackProbeAdmissionPending !== true
        ? playbackKeepBudgetFraction
        : playbackLoadBudgetFraction;
    if (!c.ensureLoaded || ewma === undefined || ewma <= fraction * periodMs) {
      if (ewma !== undefined) c.playbackProbeAdmissionPending = undefined;
      return i;
    }
  }
  return 0;
}
