const DEFAULT_TIMELAPSE_SETTLE_MS = 8000;

export function resolveTimelapseSettleMs(raw: string | undefined): number {
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_TIMELAPSE_SETTLE_MS;
}

export function isTimelapseSliceSettled(): boolean {
  const debug = (globalThis as any).__luxarDebug;
  const loader = debug?.getSceneLoader?.()?.getDefaultLoader?.();
  return loader?.isUpdateInProgress?.() === false;
}
