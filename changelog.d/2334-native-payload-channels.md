#### The Visible Human colours and the CT organ labels now live inside their fits

Both hosted archives carry their per-splat payload as a native gsplat attribute
instead of alongside a positional `.npz`: `vh_head` as `colors`, `ct_atlas` as
the exact categorical `label_ids` channel. Those sidecars had no positions, so
each was only correct while its row order happened to match its store's, and
both fell out of order twice — once sampled before the writer's spatial reorder
(#1670), and again when the archives were restructured underneath them (#2334),
leaving the shipped pairs agreeing on 0.7% and 37% of splats. A native attribute
is permuted in lockstep with the centers, so the two cannot separate and the
`_colors_match_fit` / `_labels_match_fit` guards have nothing left to compare on
that path.

The 37% is the one worth dwelling on. Organs are large and spatially contiguous,
so a misindexed label attach still lands roughly a third of splats in the right
organ and renders plausible-looking anatomy with the rest silently mislabelled —
whereas vh_head's 0.7% looks obviously broken. Neither is caught by looking; both
were caught by the guards. That asymmetry is why the label channel is exact by
contract: an amplitude-style encoder would renumber organs, returning class 51 as
50 or 52 and attributing splats to the wrong anatomy with no error anywhere, so
the round trip is asserted for per-class populations, not just for a range.

Both payloads were attached to the PREVIOUS generation of each fit, before the
first save — the generation their pinned bytes pair with at agreement 1.000. No
re-sampling was involved, which matters because the acceptance gate cannot
validate a re-sample: splats sharing a voxel share an index in any coordinate
frame, so a re-sample taken with the wrong crop box also scores 1.000, and #1670
needed three separate out-of-band checks to trust one. The provenance here is the
same bytes those checks already cleared.

The two-pass sampling on the refit paths stays, for a reason unrelated to
ordering. `sample_colors` and `sample_labels` round to the nearest voxel while the
writer quantizes centers, so a splat within a quantization step of a rounding
boundary changes voxel across a save. Attaching in a single pass mislabels exactly
those — 2 of 6000 on the vh_head fixture, each within 6.2e-05 of a boundary
against a 1.2e-04 quantization step. Invisible on screen, but it drops a fresh
pair from agreement 1.000 to 0.9993, and the thresholds are calibrated against an
exact 1.000 baseline. So both demos save, sample at the stored centers, attach,
and save again.

The sidecars are not gone. Each in-repo Git-LFS payload is a different generation
with no payload channel, and is still what a checkout resolves first, so both
demos prefer the native attribute and fall back to the sidecar for a fit that has
none. They retire together with those payloads (#2354). The hosted sidecars were
re-exported from the new archives' own channels, which is what let each
positional pair move atomically as the manifest generator requires — and makes
them aligned by construction rather than by a measurement that happened to pass.

`ct_atlas` also got smaller on the way through: 10.1 MB to 7.0 MB.
