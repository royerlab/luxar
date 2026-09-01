#### A failed progressive-LOD commit no longer strands the node

A progressive loader appends each streamed LOD level to its ladder as soon as
the data arrives, before the caller has concatenated, projected and committed
it. When that commit threw, the cursor had already moved — so the retry resumed
from the advanced cursor and attempted a *larger* allocation than the one that
had just failed, and on reaching the last rung flipped `hasMoreLODs` to false.
The refinement loop then dropped the node, its 3-strike failure cap was never
reached, no notification fired, and the node froze permanently at its last
committed rung while the data-loading monitor still reported a complete ladder.

The hosted Cosmicflows/Laniakea demo showed exactly this: basins reporting
"LOD 7/7 ~100%" while drawing a coarse prefix, with the recovery retries
themselves driving the tab further out of memory.

Each progressive loader now records its level count at pass start and unwinds
to it when a commit fails, so a retry re-attempts the same prefix instead of
escalating. Failures now reach the cap and are reported rather than passing
silently. All four geometry types carried the identical defect and all four are
fixed.
