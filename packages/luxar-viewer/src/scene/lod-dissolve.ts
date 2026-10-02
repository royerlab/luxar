/**
 * The LOD level dissolve as a state machine: which groups are dissolving from
 * which level, and how far.
 *
 * When the DISPLAYED level of a group changes, the registry dissolves from the
 * outgoing level to the incoming one over ``config.lod.fadeMs`` — the incoming
 * at ``smoothstep`` of the elapsed fraction, the outgoing at the complement. It
 * is a function of TIME since the change, never of the camera's distance to a
 * threshold, so a parked camera always settles on ONE level at full weight
 * (#2925). This module owns only the bookkeeping; whether a pair CAN dissolve
 * (both ready, the outgoing still fresh, both blendable) is the registry's
 * answer, asked through {@link DissolveHost}, and the opacity writes are
 * ``lod-fade.ts``.
 *
 * Liveness ({@link TickDemand}): a dissolve advances only when the registry
 * evaluates a frame, so the loop must keep ticking while one is in flight.
 *
 * @module scene/lod-dissolve
 */

import { smoothstep } from './lod-blend';
import { FADE_EPSILON } from './lod-fade';
import { NO_TICK, UNTIL_RESOLVED, type TickDemand } from './tick-demand';

/**
 * An in-flight dissolve between two levels of one group. ``progress`` ∈ [0, 1)
 * is the INCOMING level's share of the dissolve; its opacity is
 * ``smoothstep(progress)`` and the outgoing level's the complement. It advances
 * at ``1 / fadeMs`` per millisecond from ``startProgress`` at ``startMs``.
 */
export interface LevelFade {
  /** The outgoing level (drawn at the complement weight). */
  fromIdx: number;
  /** The incoming level — the one displayed. */
  toIdx: number;
  startMs: number;
  startProgress: number;
  progress: number;
}

/** The slice of a LOD-group entry the dissolve reads. */
export interface DissolveEntry {
  path: string;
  /** The level displayed LAST frame (``undefined`` before the first). */
  displayedChildIndex?: number;
}

/** What the dissolve asks of the registry. */
export interface DissolveHost<E extends DissolveEntry> {
  /** Whether ``fromIdx`` can be drawn dissolving against ``toIdx`` this frame. */
  canDissolve(entry: E, fromIdx: number, toIdx: number, version: number | null): boolean;
}

/**
 * The dissolve that starts when the displayed level becomes ``toIdx`` (the
 * previous frame displayed ``prevIdx``), given the dissolve in flight, if any.
 * It starts from what is on screen. With nothing in flight, from ``prevIdx`` at
 * progress 0. Reversing an in-flight dissolve (back to its outgoing level)
 * starts at ``1 − progress``, so each level keeps exactly its current opacity
 * and the reversal only undoes what happened. Retargeting to a THIRD level
 * keeps the more opaque of the two as the outgoing level, at its current
 * opacity (the other one, at most half-weight, drops out).
 */
export function retargetedFade(
  inFlight: LevelFade | null,
  prevIdx: number,
  toIdx: number,
  startMs: number
): LevelFade {
  let fromIdx = prevIdx;
  let start = 0;
  if (inFlight !== null && inFlight.toIdx === prevIdx) {
    const p = inFlight.progress;
    if (toIdx !== inFlight.fromIdx && p < 0.5) {
      fromIdx = inFlight.fromIdx; // still the more opaque one, at 1 − w(p) = w(1 − p)
      start = p;
    } else {
      start = 1 - p; // the incoming-so-far becomes outgoing, at w(p) = 1 − w(1 − p)
    }
  }
  return { fromIdx, toIdx, startMs, startProgress: start, progress: start };
}

/**
 * Every group's in-flight dissolve, by group path, plus the outgoing level of a
 * dissolve dropped before this frame's visibility pass (so that pass still
 * recounts the level leaving the screen).
 */
export class LodDissolves<E extends DissolveEntry> implements TickDemand {
  private readonly fades = new Map<string, LevelFade>();
  private readonly droppedFrom = new Map<string, number>();

  /**
   * Advance (or start) the dissolve of ``entry`` toward ``displayIdx`` at
   * ``nowMs`` and return it, or ``null`` when the group should draw
   * ``displayIdx`` alone — in which case the dissolve is forgotten.
   *
   * A dissolve STARTS when the displayed level differs from the one displayed
   * last frame. It starts from what is on screen: when the previous level was
   * itself still dissolving in (a retarget mid-dissolve), the new one begins at
   * ``1 − progress``, which keeps the previous level at exactly the opacity it
   * had, so reversing a change only has to undo the part that happened. It
   * ENDS once ``progress`` reaches 1, or as soon as the outgoing level can no
   * longer be drawn against the incoming one (released, stale for the current
   * slice, or not blendable).
   */
  advance(
    entry: E,
    displayIdx: number,
    frame: { nowMs: number; fadeMs: number; version: number | null },
    host: DissolveHost<E>
  ): LevelFade | null {
    const fade = this.retarget(entry, displayIdx, frame.nowMs, frame.fadeMs);
    if (fade === null || fade.toIdx !== displayIdx) return this.end(entry.path);
    fade.progress = Math.min(1, fade.startProgress + (frame.nowMs - fade.startMs) / frame.fadeMs);
    // The last few percent end the dissolve: applyLodFade treats a weight within
    // FADE_EPSILON of 1 as the authored opacity, so the complement would be drawn
    // on top of a full-weight level.
    if (
      smoothstep(0, 1, fade.progress) >= 1 - FADE_EPSILON ||
      !host.canDissolve(entry, fade.fromIdx, displayIdx, frame.version)
    ) {
      return this.end(entry.path);
    }
    return fade;
  }

  /** The in-flight dissolve, retargeted when the displayed level changed. */
  private retarget(entry: E, displayIdx: number, nowMs: number, fadeMs: number): LevelFade | null {
    let fade = this.fades.get(entry.path) ?? null;
    const prev = entry.displayedChildIndex;
    if (prev !== undefined && prev !== displayIdx && fadeMs > 0) {
      fade = retargetedFade(fade, prev, displayIdx, nowMs);
      this.fades.set(entry.path, fade);
    }
    return fade;
  }

  /** Forget ``path``'s dissolve without recording it as dropped; ``null``. */
  end(path: string): null {
    this.fades.delete(path);
    return null;
  }

  /**
   * Forget ``path``'s dissolve on a path that returns before the dissolve is
   * advanced, recording its outgoing level so the visibility pass recounts it.
   */
  drop(path: string): void {
    const fade = this.fades.get(path);
    if (fade) this.droppedFrom.set(path, fade.fromIdx);
    this.fades.delete(path);
  }

  /** {@link drop} every dissolve (a frame the registry cannot select for). */
  dropAll(): void {
    for (const [path, fade] of this.fades) this.droppedFrom.set(path, fade.fromIdx);
    this.fades.clear();
  }

  /** The outgoing level of ``path``'s in-flight dissolve, or ``-1``. */
  fadingFrom(path: string): number {
    return this.fades.get(path)?.fromIdx ?? -1;
  }

  /**
   * The outgoing level of ``path``'s dropped dissolve, or ``-1`` — consumed by
   * the visibility pass, which then calls {@link settled}.
   */
  droppedFromIdx(path: string): number {
    return this.droppedFrom.get(path) ?? -1;
  }

  /** The visibility pass for ``path`` ran: its dropped dissolve is accounted for. */
  settled(path: string): void {
    this.droppedFrom.delete(path);
  }

  /** Forget everything about ``path`` (unregistered). */
  forget(path: string): void {
    this.fades.delete(path);
    this.droppedFrom.delete(path);
  }

  clear(): void {
    this.fades.clear();
    this.droppedFrom.clear();
  }

  /** Whether any dissolve is in flight. */
  isAnimating(): boolean {
    return this.fades.size > 0;
  }

  tickUntilMs(): number {
    return this.fades.size > 0 ? UNTIL_RESOLVED : NO_TICK;
  }
}
