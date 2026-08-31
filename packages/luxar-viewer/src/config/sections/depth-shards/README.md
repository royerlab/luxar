# Depth-Shards Config Section

Cross-node depth ordering — splitting an order-dependent node's draw into
several contiguous depth ranges so ranges of DIFFERENT overlapping nodes can
interleave (`docs/guides/specs/CROSS_NODE_DEPTH_ORDERING_SPEC.md` §4).

Depth sorting is exact WITHIN a node, but a draw call is atomic and three.js
offers one `renderOrder` integer per object, so two overlapping order-dependent
nodes composite one entirely before the other — and for two concave
interpenetrating objects no whole-object order is correct from every viewpoint.
A node's committed permutation is already back-to-front, so any contiguous range
of it is a depth interval; drawing each range separately makes the interleaving
expressible.

| Knob                  | Default | Meaning                                                                                                            |
| --------------------- | ------- | ------------------------------------------------------------------------------------------------------------------ |
| `enabled`             | `false` | Master switch. `false` draws every node as one call, exactly as before. URL: `?depthShards=N` enables, `=0` disables. |
| `shardsPerNode`       | `16`    | Ranges a qualifying node is split into. Never more than the node has elements.                                      |
| `maxInterleavedDraws` | `512`   | Soft ceiling on total interleaved draws; exceeding it scales every qualifying node down proportionally (min 2).      |
| `minElements`         | `4096`  | Nodes smaller than this are never split.                                                                            |

## Why it is off by default

Enabling it changes the composited image of any scene with two overlapping
order-dependent nodes — for the better, which is the point, but a change — and it
costs measured GPU time on exactly those scenes. Both argue for opting in until
it has run against real datasets rather than the synthetic ones the cost was
measured on.

## Why `shardsPerNode` is generous and the budget is the backstop

The per-draw cost was measured, not estimated (spec §7), and it is **sublinear in
draw count and saturating**: at 2M splats, interleaving cost **+5.5 ms at 128**
interleaved draws and **+6.4 ms at 512** — only ~1 ms apart despite 4× the draws
— while interleaving *at all* cost the first +5.5 ms. The dominant term follows
node ALTERNATION rather than draw count and is GPU-side, most plausibly
element-texture locality plus pipeline state.

So the expensive decision is **which nodes interleave**, which is what the
overlap gate in `rendering/depth-sort-coordinator/shard-policy.ts` decides.
Tuning `shardsPerNode` down forfeits ordering accuracy without recovering the
cost, which is why the default is generous. `maxInterleavedDraws` bounds the
small term that *is* linear in draw count; that term dominates only at low
element counts (at 20k elements, 512 draws made a 0.3–0.5 ms frame 5–6× more
expensive — still only ~2 ms absolute, which such a scene can afford).

## What qualifies a node

Only a node whose world-space bounds overlap a **foreign** order-dependent
node's. Members of one order group — a spatial partition's parts — are already
ordered exactly against each other by BSP painter rank, so their mutual overlap
does not qualify; that is the spec's partition rule falling out of group identity
rather than needing a special case.

The gate is a pure function of world bounds: no camera, no frame state. Whether
two nodes overlap changes only when a node commits or the tracked set changes,
never as the camera moves, so re-evaluating it is cheap and applying it is stable.
