/**
 * Pure helpers for disposing pmndrs/postprocessing effect objects.
 *
 * Extracted from `rendering/post-processing-manager.ts` so the disposal
 * logic — try/catch around each effect's optional `.dispose()` method,
 * tagged warnings via the project log utility — is testable without
 * standing up a full PostProcessingManager (which requires a renderer,
 * scene, camera, EffectComposer, etc.).
 *
 * @module rendering/post-processing/effect-disposal
 */

import { log, Modules } from '../../utils/log';

type LogModule = (typeof Modules)[keyof typeof Modules];

/**
 * Minimum shape needed to dispose an effect: just an optional `dispose()`.
 * Falsy values and effects without `dispose` are no-ops.
 */
export interface DisposableEffect {
  dispose?: () => void;
}

/**
 * Try to dispose an effect; swallow any thrown error after logging it as a
 * warning under the supplied module tag. Returns true if dispose() ran
 * cleanly or wasn't applicable, false if dispose() threw.
 *
 * The caller decides whether a disposal failure should propagate; this
 * helper guarantees that one failing effect cannot prevent the others in
 * a teardown loop from running.
 *
 * @param effect - The effect to dispose (may be null/undefined or lack dispose).
 * @param effectName - Human-readable name used in the warning message
 *   (e.g. "Bloom rebuild", "DOF teardown"). Should encode both effect
 *   identity and lifecycle context, since this function logs nothing else.
 * @param logModule - Module tag for the warning (default POST_PROCESSING).
 * @returns true on success or no-op, false when dispose threw.
 */
export function safeDisposeEffect(
  effect: DisposableEffect | null | undefined,
  effectName: string,
  logModule: LogModule = Modules.POST_PROCESSING
): boolean {
  if (!effect || typeof effect.dispose !== 'function') return true;
  try {
    effect.dispose();
    return true;
  } catch (error) {
    log.warning(logModule, `Error disposing ${effectName}: ${error}`);
    return false;
  }
}

/**
 * Dispose a list of (effect, name) tuples in order, swallowing per-effect
 * errors so a bad apple doesn't block the rest. Returns the count of
 * successful disposals (effect existed with `dispose` and didn't throw).
 *
 * Convenience for the common pattern where a manager tears down the full
 * effect chain on rebuild or context loss.
 */
export function safeDisposeEffects(
  effects: ReadonlyArray<readonly [DisposableEffect | null | undefined, string]>,
  logModule: LogModule = Modules.POST_PROCESSING
): number {
  let succeeded = 0;
  for (const [effect, name] of effects) {
    if (safeDisposeEffect(effect, name, logModule)) {
      // Only count when there was actually something to dispose.
      if (effect && typeof effect.dispose === 'function') {
        succeeded++;
      }
    }
  }
  return succeeded;
}
