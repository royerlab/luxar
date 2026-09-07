/**
 * The offline capture's per-frame loop: orbit, settle, grab, repeat.
 *
 * Extracted from `OfflineCaptureStrategy.runOfflineCaptureLoop` (audit A2-01).
 * With the preflight, overlay, LOD drain and teardown already lifted out, this
 * is what was left of the method that actually captures anything — so giving it
 * its own name is what finally lets `runOfflineCaptureLoop` read as the
 * sequence of stages it always was.
 *
 * The per-frame ORDER is the invariant: schedule the orbit callback, spend one
 * rAF so it runs, remove it (fixing the pose), only THEN drain and grab. See
 * the comments inline — each step is ordered against the next for a reason.
 */

import { Modules, log } from '../../utils/log';
import { showToast } from '../toast';
import type { AnimationController } from '../../scene/animation/animation-controller';
import type { LuxarOrbitControls } from '../../controls/luxar-orbit-controls';
import type { CaptureProgress } from './offline-capture-overlay';
import type { CaptureContext, OfflineCaptureDriver } from './drivers/offline-capture-driver';
import type { LodSettleDrain } from './offline-lod-settle';
import type { RecordingSession } from './session';
import type { TurntablePlan } from './offline-capture-preflight';

/**
 * Consecutive per-frame capture failures after which the run gives up. Three in
 * a row is a systemic problem (an unsupported codec at this resolution), not a
 * hiccup, and continuing would produce hundreds more of the same error.
 */
const MAX_CONSECUTIVE_ERRORS = 3;

export interface FrameLoopDeps {
  plan: TurntablePlan;
  session: RecordingSession;
  driver: OfflineCaptureDriver;
  ctx: CaptureContext;
  progress: CaptureProgress;
  lodSettle: LodSettleDrain;
  animationController: AnimationController;
  /** Per-frame callback slot the orbit step registers into and removes. */
  captureCallbackId: string;
  setFrameCount: (captured: number) => void;
}

export interface FrameLoopResult {
  /** Frames the driver accepted. */
  capturedFrames: number;
  /**
   * Frames the loop actually tried to capture — incremented before
   * `driver.captureFrame`, so it counts the ones that threw too. The LOD
   * settle report's denominator: `LodSettleDrain.settleTimeouts` is also
   * counted before the capture attempt, so measuring it against
   * `capturedFrames` (successes only) could print "2 of 1".
   */
  attemptedFrames: number;
}

/** Advance the turntable one frame and pin the pose the capture will see. */
async function orbitToFrame(
  deps: FrameLoopDeps,
  controls: LuxarOrbitControls,
  i: number
): Promise<void> {
  const { plan, animationController, captureCallbackId } = deps;
  animationController.addPerFrameCallback(captureCallbackId, () => {
    if (i > 0) controls.applyOrbitRotation(plan.anglePerFrame);
    // Called on frame 0 too, deliberately: switching the interactive
    // dolly off leaves the camera wherever the swing had reached, so
    // this is what drives the phase to 0 and puts the capture on the
    // true baseline distance rather than on a leftover offset.
    if (plan.dollyCycles > 0) controls.applyOrbitDolly(plan.dollyPhaseFor(i));
  });

  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

  animationController.removePerFrameCallback(captureCallbackId);
}

/**
 * Capture every frame of the turntable.
 *
 * Returns as soon as the session is stopped or aborted, the driver asks to
 * stop, or three consecutive frames fail — the caller reads the counts and
 * decides whether there is anything to finalize.
 */
export async function runFrameLoop(deps: FrameLoopDeps): Promise<FrameLoopResult> {
  const { plan, session, driver, ctx, progress, lodSettle } = deps;
  const { sessionAbort, controls, totalFrames } = plan;
  let capturedFrames = 0;
  let attemptedFrames = 0;
  let consecutiveErrors = 0;

  for (let i = 0; i < totalFrames; i++) {
    if (!session.isRecording) break;
    if (sessionAbort.signal.aborted) break;
    if (driver.shouldAbort?.()) break;

    await orbitToFrame(deps, controls, i);

    // ── Wait for the LOD selector to settle on this pose (#1695) ──
    // MUST stay after the orbit callback's removal above: the camera pose
    // is fixed from that point on, so the extra frames only let pending
    // loads land. Draining before it would keep advancing the turntable
    // while we wait, smearing the sweep.
    //
    // Cost: one rAF per frame even on a fully settled scene THAT HAS LOD
    // GROUPS. At ~16 ms against a full pipeline render plus an async GPU
    // readback plus an encode for every frame, that is noise — and a scene
    // with no lod_group at all pays nothing, because the predicate answers
    // `null` and the whole wait (that tick included) is skipped.
    //
    // Not force-finest: a capture visits the whole scene, so pinning the
    // finest level across a tiled partition would make peak residency the
    // entire dataset. Waiting costs time, not memory.
    await lodSettle.waitForFrame();

    if (sessionAbort.signal.aborted) break;

    attemptedFrames++;
    try {
      await driver.captureFrame(ctx, capturedFrames, progress);
      capturedFrames++;
      consecutiveErrors = 0;
    } catch (err) {
      consecutiveErrors++;
      log.error(Modules.RECORDING, `Frame ${i + 1} capture failed: ${err}`);
      if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        log.error(
          Modules.RECORDING,
          `${MAX_CONSECUTIVE_ERRORS} consecutive failures — aborting capture. ` +
            'The browser may not support 10-bit encoding at this resolution.'
        );
        showToast('HDR video encoding failed — try EXR sequence instead');
        break;
      }
    }

    deps.setFrameCount(i + 1);
  }

  return { capturedFrames, attemptedFrames };
}
