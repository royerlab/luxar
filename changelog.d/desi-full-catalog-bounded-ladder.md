#### DESI DR1 ships the whole catalog again, on a ladder that can stream it

The finest level was capped at a 1.25M-row sample (#1812) because it downloaded
all 9.75M rows in one commit and blocked the main thread. The diagnosis named
the wrong culprit. The payload was unstreamable because of the SHAPE of the
ladder under it, not its size: `stream:<c>` doubles until it reaches `n`, so its
last increment is always `n/2` — whatever base the sibling-aware rule picks.
That is 4,875,971 points at full density, and it was still 625,000 at the cap.
Capping the catalog shrank the final commit without fixing its geometry, and
gave up 87% of the survey to do it.

`streaming_breakpoints` fixes the geometry instead. It keeps the geometric ramp,
which is what makes first paint cheap — the first rung is still 2,000 points,
one range request — but stops doubling at a commit ceiling and finishes in equal
steps of that size. The largest single commit is therefore that ceiling
(900,000) at any `n`. On the rebuilt scene the finest level is 9,751,955 points
over 20 rungs whose largest is 9.2% of the level, against five rungs whose
largest was 50%. `SCENE_MAX_POINTS` is gone; `None` means every row.

No library change was needed. `_validate_counts` already clamps an explicit
cumulative list to each level's own `n` and stops there, so one list serves the
whole ladder: the 152K level takes just the geometric head (8 rungs), the 1.22M
middle level avoids a majority-sized final rung, and the finest takes the whole
thing (20). That also keeps the small levels laddered,
which a sibling-aware `stream:` base at this scale would not have — its base
scales with `n`, so at 9.75M the first chunk alone would have been 609K.
Reaching the coarser sibling's size, the point where swapping this level in is
worth it, costs about 14% of the finest payload, so the upgrade does not "wait
until fully loaded".

`warn_if_scene_is_stale` now measures the largest RUNG rather than the level
total. A scene can carry five rungs, pass a total-based check, and still hand
the main thread millions of points at once — which is exactly how the real
defect survived the cap. The check reads `additive_<i>` increments and names the
ceiling it breached.

Both ladders are unchanged in kind: three substitutive levels, each with an
additive ladder beneath it. Only the additive rung schedule and the row count
moved. The shipped Git-LFS scene is rebuilt from the full catalog (10 MB → 73 MB,
`data_manifest.json` checksum re-pinned, `download_mb` updated); the Zenodo copy
still needs reuploading to match the new checksum.
