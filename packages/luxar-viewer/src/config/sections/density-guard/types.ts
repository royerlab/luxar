/**
 * Projected-density guard — configuration.
 *
 * The guard measures, per committed data node and per frame, how many
 * elements land on each drawing-buffer pixel of the node's projected
 * bounding sphere. Above the cap the node is over-drawn: on a 1.5 M-point
 * example framed into ~1 600 px the frame cost 42 ms at DPR 1 and 83 ms at
 * DPR 0.5 (lower resolution concentrated the same fragments into fewer
 * tiles), while the same points dollied 4× closer rendered at 120 fps. The
 * fix is fewer elements per pixel, not fewer pixels: a per-node keep
 * fraction on the shaders (blendable modes, brightness-compensated) and a
 * cap on which additive rungs the refinement loop admits.
 */
export interface DensityGuardConfig {
  /** Master switch; the `?noDensityGuard` URL flag clears it for a session. */
  enabled: boolean;
  /**
   * Elements per drawing-buffer pixel above which a node counts as
   * over-drawn. Additive/luminous/volumetric nodes may be thinned down to
   * this density; other modes only stop admitting further rungs.
   */
  capElementsPerPixel: number;
  /**
   * Density cap for the REFINEMENT rung gate on nodes whose blend mode has no
   * linear brightness knob (`max` / `normal` / `opaque`): they cannot be
   * thinned, so their only relief is to stop admitting rungs earlier. One
   * element per pixel already saturates a max projection or an opaque surface.
   * Blendable nodes use `capElementsPerPixel` for the rung gate too.
   */
  nonBlendableCapElementsPerPixel: number;
  /**
   * Smallest keep fraction the shader ladder may reach (the ladder is
   * 1, 1/2, 1/4, … down to this value).
   */
  minKeepFraction: number;
  /**
   * Hysteresis around the cap for the keep-fraction ladder: thin one step
   * further only above `cap × enterRatio`, restore one step only below
   * `cap × leaveRatio`, so a slow zoom does not flicker between steps.
   */
  enterRatio: number;
  leaveRatio: number;
}
