#### Give a sliced node's additive ladder a share of the frame, not a byte budget

`stream_ladder` derived its first rung from a download budget — 39,062 elements
for 200 ms at 25 Mbps — and applied it to the node's TOTAL element count. But the
viewer draws one hidden-axis coordinate at a time, so on a sliced nD node that
rung arrives divided by the number of stops. Measured per resident slice:
human_multiome 6,510 of 1,041,455 (0.63%), zebrahub 5,580 of 640,830 (0.87%),
cellxgene 1.30%, mouse and ESM3 3.39%.

Above one slice the first rung is now floored at `n / SLICED_LADDER_MAX_DEPTH`,
giving every sliced node **12.5% of its resident slice** at first paint:

| demo | slices | before | after |
|---|---|---|---|
| human_multiome | 6 | 0.63% | 12.50% |
| zebrahub | 7 | 0.87% | 12.50% |
| cellxgene_census | 3 | 1.30% | 12.50% |
| mouse_multiome | 6 | 3.39% | 12.50% |
| esm3_protein | 2 | 3.39% | 12.50% |

A share rather than a byte budget because a share is what predicts whether the
opening frame is recognisable — measured against observed playback, 0.03% renders
blank, 2.4% has lost its structure, 12.5% is soft but usable, and 54% is fine on
only 1,735 absolute elements. The latency cost is accepted rather than hidden:
first paint moves from ~200 ms to ~123-666 ms, part of which is repaid in
requests, since hosted first paint is dominated by request count and fewer, fatter
rungs mean fewer nodes to fetch.

Two properties worth knowing. The floor needs no slice term — requiring
`rung0/S >= share * (n/S)` cancels to `rung0 >= share * n` — so it is
slice-invariant and a demo that gains a dimension cannot silently regress. And it
is a `max()`, so a node whose budget rung already clears the share keeps its finer
ladder, and an unsliced node is untouched entirely.

New `hidden_axis_stops(positions, hidden_dims)` counts distinct **occurring**
coordinate combinations, not the product of per-axis cardinalities: `taxon x
period` on biodiversity is 126 of 140 populated, and drosophila declares 500 stops
with data at 499, so a product overstates the divisor. Call sites pass
`dims.non_displayed` rather than literal indices, because the hidden axis is
column 0 in the multiome demos and last in the timelapse demos.

Authoring half of #2374; the viewer-side playback fix is separate. No published
store changes — the five were measured to be on the `refine` path, which already
streams every cache-resident level, so their bytes needed no rebuild.
