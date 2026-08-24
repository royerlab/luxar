#### Keep the full DESI cosmic web drawable on 4096-class GPUs

The DESI DR1 demo still carries all 9,751,955 galaxies and quasars in both of its
colourings, but no longer puts the finest level into one Points node. Each layer's
finest branch is now a four-part spatial partition whose largest leaf holds about
2.44 million points, below the 5,591,040-point element-texture capacity on a
4096-class GPU. The existing bounded additive ladder remains inside every part,
so no streaming commit exceeds 900,000 points.

The partition-bound screen-area ladder intentionally changes its child switch
fractions from `[0.0, 0.25, 0.5]` to `[0.0, 0.5, 1.0]`.

The partition is storage-neutral for the shared geometry: corresponding position
arrays in the redshift layer remain `array_ref`s to the tracer layer. The
precomputed scene is rebuilt with the new hierarchy, so the fix applies on the
demo's normal fast path rather than only after a local recomputation — once the
rebuilt archive reaches the `cc-by` record the manifest now pins it to.
