#### The nuclear pore gains a streaming ladder; biodiversity loses three partitions

Two demos, opposite directions, same measurement.

**`nuclear_pore_complex` keeps its partition and gains a ladder.** Its 32-leaf BSP
is doubly load-bearing — the subunits are concave and interpenetrate, so the split
planes are the only valid draw order, camera-inside-the-channel included — but
`partition=` is **eager**: every part is fetched and the GPU only frustum-culls at
draw time. So first paint was the whole 4,937,064-atom resident state across 32
parts, with no rungs anywhere. It now carries an additive ladder, resolved per
part by `_validate_counts` clamping the shared spec to each part's own ~308,569
atoms.

Its `MAX_ELEMENTS_PER_PART` note is corrected while we are here. It claimed a
single unpartitioned node "would silently drop ~4.3M" atoms of the 9,874,128
stored. It would not: `state` is a hidden axis, so the viewer slices to one state
and only **4,937,064** are ever allocated — under the 5,591,040 Points cap, not
over it. That is exactly the node-total-versus-resident-slice error the policy
module now documents. What is true is that 4,937,064 leaves 12% headroom, which is
no margin; and the correctness argument stands on its own regardless.

**`biodiversity_planetary_scale` drops all three partitions.** Both `taxon` and
`period` are hidden, so every layer is sliced to one marginal: 45,000 points for
the occurrence cloud, 106,026 and 95,418 vertices for the two track layers,
against caps of 5,591,040 and 2,793,472 — two to three orders of magnitude of
headroom. A partition exists so the viewer can skip off-screen geometry; there is
nothing here worth skipping, and each part costs a request. `Migration highways`
never even wrote a wrapper (105,662 segments against a 2,500,000 cap resolved to
one part), so that one was already a no-op.

`Migrations by slice` gains a ladder in exchange, since 318,078 vertices is above
the 200,000 floor `scripts/check_demo_ladders.py` enforces. That is only possible
because indexed lines can now be laddered when verified: `chain_segment_indices`
emits consecutive pairs within each ragged chain and never across, so every
component is an ascending simple path — including the length-0 and length-1 chains
that consume a vertex slot without a segment and become isolated single-vertex
components.

Its module docstring is corrected too. It still described a `kind=partition` of
per-tile `kind=lod` ladders holding ~0.2M of 15M, which the shipped scene stopped
building when the dense aggregate layer was removed (`add_lod_tiles` is that
layer's now-uncalled builder, left in place).

**One trap found and pinned.** `stream_ladder` now takes `geometry=`, because on
`add_lines` an explicit `counts` **list** is in POLYLINES while `"stream:<c>"` is
in VERTICES. A vertex-sized list therefore clamps to the polyline count and writes
**no rungs at all** — no error, no warning. Measured on 4,000 polylines x 27
vertices: `counts=[39062, 78124, 108000]` produced 0 rungs; `"stream:39062"`
produced 3, of 39,069 / 39,069 / 29,862 vertices. Only `check_demo_ladders.py`
catches the silent case, and only above 200,000.
