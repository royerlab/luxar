// Headless Linux ANGLE/Vulkan measured WebGL orbit throughput at 17.1–17.2 ms
// for scenes with synced frame costs of 11.5–13.1 ms. Treat a result within
// about 1 ms of a 60 Hz frame as pacing-bound, not a GPU throughput estimate.
const FRAME_60_HZ_MS = 1000 / 60;
const PACING_WINDOW_MS = 1;

export function pacingBound(backend, ms) {
  return (
    backend === 'webgl' && Number.isFinite(ms) && Math.abs(ms - FRAME_60_HZ_MS) <= PACING_WINDOW_MS
  );
}
