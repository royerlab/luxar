/**
 * Level-of-detail (`kind=lod` group) display policy — configuration.
 *
 * The LOD group registry (`scene/lod-group-registry.ts`) picks one level per
 * group per frame. What this section tunes is how a CHANGE of the displayed
 * level reaches the screen, and the timing and budget rules of the registry's
 * selection, loading and partition gating.
 */
export interface LodConfig {
  /**
   * Duration, in milliseconds, of the dissolve between the outgoing and the
   * incoming level when a blendable (additive / luminous / volumetric) group
   * changes its displayed level. The dissolve is a function of time since
   * the change, so a parked camera always settles on ONE level. A retarget
   * mid-dissolve continues from the current opacities. `0` makes every change
   * a hard swap; `?noLodFade` disables the dissolve for a session.
   */
  fadeMs: number;
  /**
   * Half-width of the PRELOAD band around each level threshold, as a fraction
   * of the smaller adjacent inter-threshold gap (in the group's selector
   * units). While the selector metric of a blendable group is inside the band
   * of a threshold, the level across it is loaded in the background and kept
   * hidden, so crossing the threshold starts the dissolve at once instead of
   * after that level's load. `0` disables it; it applies only while the
   * dissolve is on and never during playback. Strictly below
   * {@link preloadExitBandFraction}, so the two bands keep a hysteresis gap.
   */
  preloadBandFraction: number;
  /**
   * EXIT half-width of the preload band, in the same units as
   * {@link preloadBandFraction}: a visit ends only once the metric leaves this
   * band, which is wider than the entry band so a camera hovering at the entry
   * edge does not start a new visit (and a reload) per wobble. At most `0.5`,
   * the widest band that cannot reach a neighbouring threshold's.
   */
  preloadExitBandFraction: number;
  /**
   * Milliseconds the view-update version must hold steady before the registry
   * reloads a stale fine level (the settle debounce). While the user is
   * actively scrubbing (the version changes every frame) only the cheap coarse
   * level shows; the fine level reloads once they pause this long. In
   * milliseconds, like {@link staleHoldMs}, so it means the same on any display
   * (it was 8 frames: 130 ms at 60 Hz, 48 ms at 165 Hz). Playback does not wait
   * for it (see {@link playbackLoadBudgetFraction}): a playing timelapse bumps
   * the version every period, so it never settles.
   */
  fineReloadSettleMs: number;
  /**
   * During playback the registry aspires to the finest level whose measured
   * load+commit time (`LODGroupChild.loadEwmaMs`) is at most this fraction of
   * the playback period, and reloads it on every timepoint without waiting for
   * the settle debounce. A level the period cannot carry would be stale on
   * every frame and drop the display to the coarsest fresh level; the finest
   * one that CAN keep up is what a video player's "auto quality" would pick.
   * An unmeasured lazy level counts as fitting, so it gets measured.
   */
  playbackLoadBudgetFraction: number;
  /**
   * Hysteresis for {@link playbackLoadBudgetFraction}: the current aspiration,
   * and any coarser level a demotion steps down to, only needs a reload average
   * within this fraction of the period (a finer level is still admitted at the
   * load budget). Without it a level whose reloads straddle the admission
   * budget flipped with each sample (lod_timelapse at 20 fps on obsidian: an
   * average of 40.6 ms against a 40 ms budget sent the mid level to the
   * coarsest for a second), and a demoted finest level skipped a mid level
   * averaging 45 ms of a 50 ms period. At least the load budget.
   */
  playbackKeepBudgetFraction: number;
  /** Weight of a new sample in `LODGroupChild.loadEwmaMs`, in (0, 1]. */
  loadEwmaAlpha: number;
  /**
   * Retry the next capped playback level this often (ms) to measure warm
   * loads. The probe's reload REPLACES the level's average rather than moving
   * it by {@link loadEwmaAlpha}: on lod_timelapse at 20 fps (obsidian) the first
   * loop's cache-miss reloads (80-200 ms) capped the mid level, and blending
   * each warm 5-10 ms probe into that stale average took four to six probes,
   * i.e. seconds at the coarsest level.
   */
  playbackProbeIntervalMs: number;
  /**
   * Milliseconds the registry keeps a STALE previously-displayed level on
   * screen, rather than dropping to a much coarser fresh one, while the
   * aspiration re-commits for a new slice (`staleHoldDisplayIndex`).
   *
   * The slice-aware fallback shows the coarsest FRESH level the instant a
   * scrub invalidates the aspiration. That is right when the aspiration is
   * seconds away, and wrong when it is milliseconds away from a warm cache: on
   * a 151-timepoint gsplat timelapse the coarse level re-commits in ~10 ms and
   * the finest in ~70 ms, so every single step of the Time slider flashed
   * 6,900 splats down to 108 and back — 1.6% of the detail, for four frames. A
   * video player holds the previous frame until the next one decodes; so does
   * this.
   *
   * Bounded, because holding is only better while the wait is short. The
   * budget is spent from when the hold STARTS and is not refreshed by further
   * version bumps, so a continuous drag (a new version every frame, the
   * aspiration never committing) exhausts it once and then shows live coarse
   * geometry. In MILLISECONDS, deliberately: sizing it in frames makes it
   * display-dependent (8 frames worked on a 60 Hz panel and still flashed on a
   * 165 Hz one, where 8 frames is 48 ms and the re-commit needs ~70). 250 ms is
   * comfortably above the ~70 ms a warm re-slice takes on a 1.6 M-splat
   * timelapse and well below the point where a frozen frame reads as a hang.
   */
  staleHoldMs: number;
  /**
   * How much worse the coarse fresh fallback must be, as a fraction of the held
   * level's committed element count, before holding a STALE finer level is
   * worth it. Freshness normally wins: showing the right slice matters more
   * than showing more geometry. The exception is a fallback that is not a
   * slightly coarser view of the new slice but a token of it — the
   * 108-of-6,900-splat drop that made a timelapse step read as a flash. At this
   * fraction of the detail or better the fallback is taken immediately.
   */
  staleHoldMinRatio: number;
  /**
   * Milliseconds a lazy level stays in the `failed` state before the registry
   * retries its deferred load — long enough to avoid per-frame retry storms
   * after a hard failure, short enough that a transient (network blip) failure
   * self-heals. Wall-clock, not a frame count. A one-shot wake at cooldown
   * expiry lets a parked camera retry without per-frame ticking; each
   * consecutive failure doubles the wait (``scene/retry-wakes.ts``).
   */
  failedRetryMs: number;
  /**
   * How long (ms) a deferred partition part's activation request may wait for
   * a loader pass to reach it before the per-frame gate asks again (the resync
   * was rejected, or superseded by a pass targeting other parts). A one-shot
   * wake at expiry asks again; each unanswered request doubles the wait
   * (``scene/retry-wakes.ts``).
   */
  lazyActivationRequestTimeoutMs: number;
  /**
   * Screen-space pad of the partition frustum gate, as a fraction of the
   * viewport half-extent on x and y. Cold parts have no loaded footprint to
   * union, so the pad preloads them before they enter the view and keeps entry
   * and exit symmetric.
   */
  partitionFrustumMargin: number;
  /**
   * Asymmetric hysteresis of the threshold and footprint picks, on the
   * "downgrade to coarser" direction only: the metric must fall this fraction
   * of the gap below the current level's threshold before a coarser one is
   * picked. Mirrored by `HYSTERESIS_RATIO` in `luxar.io.lod_screening`.
   */
  hysteresisRatio: number;
  /**
   * The GSplat footprint pick's median-sigma limit, in logical CSS pixels.
   * GSplats draw to about 3σ, so 1.5 px corresponds to a typical rendered blob
   * about 9 px across. The #2685 sweep retained this policy.
   */
  maxMedianFootprintPx: number;
}
