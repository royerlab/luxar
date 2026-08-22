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
the SHIPPED store's own centers, which is aligned by construction. The volume
reproduced the fit's frame exactly (shape 636 x 451 x 896 against centers
spanning [0, 0, 0] to [635, 450, 895]), confirming the shipped fit came from this
same pipeline. The regenerated sidecar also preserves the previous sidecar's
colour distribution (total-variation distance 0.0065), confirming the source
volume was rebuilt in the intended frame rather than merely sampled consistently
in a wrong one. The pair now agrees at 1.0.

With the fast path live, everything that documented the slow one is reverted:
`download_mb` 1100 to 25, `compute` "heavy" to "medium", the module docstring,
the **Requires** paragraph in `demos/README.md`, the row in
`demos/data/README.md`, and the `gsplats_3d_visible_human_head` entry in the
gallery's `UNBUILDABLE_IDS` (now empty). A tripwire test had been left behind to
force exactly this list to be revisited; it fired, and is replaced by the
positive assertion it asked for.

That new gate is the point. The guard was already covered by tests, but only
against synthetic pairs — the mechanism was tested and the artifact was not,
which is how a mis-ordered sidecar shipped at all. CI now pins the verified fit
and colors SHA-256 pair through `data_manifest.json`, while a slow test on a
materialized Git-LFS checkout checks their actual same-voxel correspondence.
Changing either asset without re-verifying and updating the pair is therefore a
red test even where LFS payloads are not downloaded.
