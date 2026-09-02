/**
 * Gallery timelapse settling policy shared by still and orbit capture.
 *
 * `getState().isLoading` clears at the first geometry commit, before a
 * progressive LOD ladder finishes. The loader's broad update lock stays held
 * through that refinement drain, so it is the signal that prevents screenshots
 * from baking in intermediate rungs.
 */

import type { SceneLoaderManager } from '../../data/scene-loader-manager';

const DEFAULT_TIMELAPSE_SETTLE_MS = 8000;

interface TimelapseSettlePage {
  waitForFunction(
    pageFunction: typeof isTimelapseSliceSettled,
    arg: undefined,
    options: { timeout: number }
  ): Promise<unknown>;
}

type TimelapseDebugGlobal = typeof globalThis & {
  __luxarDebug?: { getSceneLoader?: () => SceneLoaderManager | null };
};

export function resolveTimelapseSettleMs(raw: string | undefined): number {
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_TIMELAPSE_SETTLE_MS;
}

/**
 * Whether the default loader has released its full update lock, including the
 * progressive-refinement drain that `getState().isLoading` intentionally omits.
 */
export function isTimelapseSliceSettled(): boolean {
  const debug = (globalThis as TimelapseDebugGlobal).__luxarDebug;
  const loader = debug?.getSceneLoader?.()?.getDefaultLoader();
  return loader?.isUpdateInProgress() === false;
}

/** Wait for a timelapse slice's progressive ladder without aborting the sweep. */
export async function waitForTimelapseSliceSettled(
  page: TimelapseSettlePage,
  demoId: string,
  timeoutMs: number
): Promise<void> {
  await page
    .waitForFunction(isTimelapseSliceSettled, undefined, { timeout: timeoutMs })
    .catch((error: unknown) => {
      console.warn(`[${demoId}] timelapse slice did not settle within ${timeoutMs} ms: ${error}`);
    });
}
