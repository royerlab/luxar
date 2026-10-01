/**
 * Level-of-detail (`kind=lod` group) display policy — configuration.
 *
 * The LOD group registry (`scene/lod-group-registry.ts`) picks one level per
 * group per frame. What this section tunes is how a CHANGE of the displayed
 * level reaches the screen.
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
   * dissolve is on and never during playback. At most `0.5`, which keeps the
   * bands of adjacent thresholds from overlapping.
   */
  preloadBandFraction: number;
}
