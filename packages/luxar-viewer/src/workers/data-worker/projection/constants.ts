/**
 * Shared numeric thresholds for the worker projection dispatchers.
 *
 * These constants were previously inlined (and duplicated) across the
 * worker dispatchers and the now-deleted main-thread projection copies.
 * Centralizing them here gives the worker (and the in-process dispatcher
 * the main thread runs as its fallback) a single source of truth, so a
 * threshold can't silently drift between the GSplats and Lines paths.
 *
 * @module workers/data-worker/projection/constants
 */

/**
 * Minimum attenuated amplitude for a GSplat to count as visible. Splats
 * whose `amplitude × attenuation` falls below this are compacted out.
 */
export const MIN_AMPLITUDE = 1e-6;

/**
 * Detection sentinel for an `extend_to_all` dimension. A per-dimension
 * tolerance at or above this value means "visible across every value of
 * this dimension" — the dimension is skipped entirely during slicing.
 * Set below `EXTEND_TO_ALL_TOLERANCE` (1e10) so the flag is detected
 * even after float round-trips, while staying unreachable for real data.
 */
export const EXTEND_TO_ALL_THRESHOLD = 1e9;

/**
 * Default truncation radius (in sigmas) for the GSplat shifted-Gaussian
 * hidden-dimension attenuation. Splats beyond this Mahalanobis distance
 * in the hidden dims attenuate to zero.
 */
export const SHIFTED_GAUSSIAN_DEFAULT_TRUNCATE = 3.0;
