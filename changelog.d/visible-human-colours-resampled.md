#### The Visible Human demo has a working fast path again (#1670)

The shipped `vh_head_colors.npz` had been sampled in the pre-save splat order, so
it did not correspond to the `vh_head.gsplats.zarr.zip` beside it. The guard
caught that — same-voxel agreement 0.00097 over 1,911,192 splats — and did the
right thing, refusing to render it. The cost was that the DEFAULT path for anyone
who pulled Git LFS became a ~1.1 GB cryosection download plus a multi-million
splat refit, on every run.

The sidecar could not be repaired in place. It carries colours and nothing else:
with no positions in it there is no way to recover which splat each colour
belonged to. Both stores on disk were ruled out as the missing order too — the
cached fit and the shipped fit have byte-identical centers and score the same
0.00097 — so the order it was written in exists nowhere any more.

Re-sampling was therefore the only route, but it needed no refit: the fit itself
was never wrong. The RGB volume was rebuilt from the cryosections and sampled at
the SHIPPED store's own centers. The pair now agrees at 1.0.

That number is necessary and not sufficient, which is worth being precise about
because `load_or_build` deliberately refuses to do this resample automatically
for exactly that reason: splats sharing a voxel share an index in *any*
coordinate frame, so a resample taken in a drifted frame would also score 1.0.
The frame was therefore checked out of band, three ways. The rebuilt volume's
shape, 636 x 451 x 896, matches the stored centers spanning [0, 0, 0]–[635, 450,
895] exactly, so neither the crop box nor the resample factor moved. 99.96% of
those centers land on non-zero (tissue) voxels against a 32.93% tissue fraction
for the volume as a whole — where a 10-voxel shift scores 98.9%, a 25-voxel
shift 90.7%, and a y/x axis swap 22.0%, below the base rate. And the regenerated
colours preserve the previous sidecar's colour distribution (total-variation
distance 0.0065), which pins the source volume and its masking independently of
the ordering.

None of the three suffices alone — the distribution check would survive a small
translation, the tissue-hit rate would survive a subtle resample change, and the
shape check says nothing about content — so all three are recorded in the demo's
module docstring for the next regeneration to reproduce.

With the fast path live, everything that documented the slow one is reverted:
`download_mb` 1100 to 25, `compute` "heavy" to "medium", the module docstring,
the **Requires** paragraph in `demos/README.md`, the row in
`demos/data/README.md`, and the `gsplats_3d_visible_human_head` entry in the
gallery's `UNBUILDABLE_IDS` (now empty). A tripwire test had been left behind to
force exactly this list to be revisited; it fired, and is replaced by the
positive assertion it asked for.

That new gate is the point. The sidecar went out mis-ordered because the write
path sampled the in-memory fit before saving it, at a time when nothing checked
the pair at all; #1673 fixed that write path and added the guard, which
immediately caught this artifact — and then the artifact was parked behind a
tripwire rather than repaired. So what was missing here was never the mechanism,
which is thoroughly tested against synthetic pairs, but a positive assertion
about the REAL shipped bytes. Its sibling `ct_totalsegmentator` has had exactly
that (`test_shipped_pair_is_accepted`) all along, and its artifact stayed
correct.

The Visible Human now has the same coverage, in two halves so that neither
environment escapes it: CI pins the verified fit and colors SHA-256 pair through
`data_manifest.json`, which needs no LFS payload, while a slow test on a
materialized Git-LFS checkout measures their actual same-voxel correspondence.
Changing either asset without re-verifying and updating the pinned pair is a red
test even where LFS payloads were never downloaded. The CT sibling's payload
correspondence check still requires a materialized Git-LFS checkout.
