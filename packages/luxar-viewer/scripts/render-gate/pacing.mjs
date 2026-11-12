// Headless Linux ANGLE/Vulkan measured WebGL orbit throughput at 17.1–17.46 ms
// for scenes with synced frame costs of 11.5–13.1 ms. Pacing can only add
// time, so allow a narrow margin below 60 Hz and drift above it.
const FRAME_60_HZ_MS = 1000 / 60;
const PACING_EARLY_MS = 0.25;
const PACING_LATE_MS = 1.5;

export function pacingBound(backend, ms) {
  return (
    backend === 'webgl' &&
    Number.isFinite(ms) &&
    ms >= FRAME_60_HZ_MS - PACING_EARLY_MS &&
    ms <= FRAME_60_HZ_MS + PACING_LATE_MS
  );
}
