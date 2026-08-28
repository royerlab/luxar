#### The bounded additive ladder schedule is now shared, not one demo's private function

`stream:<c>` doubles all the way to `n`, so its final increment is `n` minus the
largest doubling below it — a quantity that grows with `n` and approaches `n/2`
at worst, whatever the first chunk is. Shrinking `c` does not help the tail. At
`c=2000` a 6,248,730-element leaf commits 2,152,730 points in one go, twice the
1,000,000 at which `scripts/check_demo_ladders.py` fails a level.

`desi_galaxies` already solved this (#1812) with a schedule that keeps the
geometric ramp, which is what makes first paint cheap, but stops doubling before
the next increment would exceed a ceiling and finishes in equal steps of that
size — largest commit is the ceiling at any `n`. That schedule lived in the demo
module, so the next demo with a multi-million-element leaf had nothing to reach
for while `luxar.utils.lod_breakpoints.stream_cuts` still offered only the
doubling-to-`n` shape the demo's own comment calls unstreamable.

It is now `luxar.utils.lod_breakpoints.capped_stream_cuts(n, chunk, max_commit)`,
beside `stream_cuts`, with the two numbers as module constants
(`DEFAULT_CAPPED_FIRST_CHUNK` 2,000 — small enough for one range request;
`DEFAULT_MAX_ADDITIVE_COMMIT` 900,000 — under the gate's 1,000,000 with margin).
`demo_desi_galaxies.streaming_breakpoints` delegates and keeps its two pinned
constants, so the published archive's ladder is unchanged; its 47 tests pass
unmodified, which is what proves that.

It returns a cumulative list rather than adding a `"capped-stream:<c>:<max>"`
spec string. The string form's only advantage would be per-node adaptation, and
`_validate_counts` already gives that: it clamps a cumulative list to each
node's own `n` and stops there, so one list serves every level, part and leaf —
a small coarse level takes just the geometric head while the finest takes the
whole schedule.
