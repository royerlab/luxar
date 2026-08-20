#### Reductions remeasure the LOD Q·e stamps they render with (#1789)

A content-changing gsplat rewrite no longer carries the input ladder's
`energy_fraction_cum`, `reference_energy`, `quality` and splat counts onto a
smaller or amplitude-edited artifact. Those values drive the viewer's
progressive brightness compensation and upgrade gate, so the stale metadata was
rendering-visible: a fully loaded reduced level could still be divided by the
input's partial-energy fraction and appear over-bright.

The shared reduction chokepoints now rebuild existing artifact-local stamps from
the rewritten splats. Per-rung counts and cumulative energy fractions are
remeasured, every substitutive level receives the rewritten finest level's
common reference-energy weight, and an authored level quality is measured again
against that finest content. The operation preserves absence — unannotated and
reveal ladders do not acquire energy compensation — and preserves authored
stamps verbatim for content-neutral rewrites and the `at_substitutive` accessor
used by scene authoring. `level_stats.refine_stats` is removed after a reduction
because its MSE values were measured against a source volume the rewriter does
not have; the descriptive refine method remains.
