#### The data-loading monitor now sees mesh

On `ocean_currents_earth` the monitor's headline card read `VISIBLE LINES 4.1M`
and the scene-graph header said `48 lines, 3 mesh` — the panel knew the meshes
were there and reported nothing about them anywhere else. A sweep of the whole
panel found mesh missing from every surface except the scene-graph tree, and the
reason it stayed invisible for so long is worth recording: the type *vocabulary*
was already four-way (`GEOMETRY_TYPES`, `SceneGraphState.visibleByType.mesh`, the
visible-counts walk), so the counts were being aggregated every update cycle —
and then dropped, because the display layer's named per-type fields
(`GlobalStats.visiblePoints` / `visibleSegments` / `visibleSplats`) stopped at
three. Nothing was broken enough to report itself.

**Headline counts.** Overview hero cards, their per-tick patcher, and the
collapsed compact badge each kept their own hard-coded list of three types, with
their own presence test and their own label strings. They now read one shared
table (`ui/data-loading-monitor/headline-counts.ts`) that is a
`Record<GeometryTypeName, …>`, so a fifth geometry type is a compile error in one
place instead of a silent omission in three. Mesh reports `VISIBLE TRIANGLES` —
the drawn-primitive convention the rest of the monitor already used for mesh, and
the same choice `lines` makes in counting segments rather than vertices. A
mesh-only scene previously showed a permanent `LOADING …` card, since all three
presence tests came back false.

The hero grid is now `repeat(auto-fit, minmax(150px, 1fr))`: one row while the
cards fit, wrapping to 2×2 in a narrow panel. That also fixes a quieter bug — the
fixed grid classes only ever defined `--cols-1` and `--cols-2`, so a three-type
scene emitted a `--cols-3` with no CSS rule behind it and its cards silently
stacked in a single column.

**Loader telemetry.** `MeshWholeNodeLoader` and `MeshProgressiveLoader` now
implement the `LoaderMonitor` surface (`mesh-whole-node`), so mesh joins the
loader list, the loader-memory and throughput totals, the load-rate and bandwidth
windows. Whole-node mesh latency is deliberately excluded from the advisor's
chunk-size slow-load recommendation. The four-method shape is
load-bearing rather than cosmetic: `connectLoaderToMonitor` duck-types the
complete set and skipped every mesh node **without logging** — the panel simply
had no mesh row. What is reported is a whole-node loader's honest telemetry:
`loads` / `bytesLoaded` (decoded) / `avgLoadTime` for the one fetch per level,
`elementsLoaded` in triangles, `memoryUsed` for the resident payload plus the
per-node projection scratch, and `queries` / `avgQueryTime` / `spatialIndex`
deliberately zero or absent — there is no spatial index here and a view change
re-serves the resident mesh, so a query sample would put a ~0 ms entry for work
that never touched the store into the panel's QUERY SPEED average. For the same
reason mesh counts in `totalLoaders` but not in `activeSpatialLoaders`, which is
what drives the badge's "spatial-index streaming" label.

`visibleElements` is pushed IN from `commit-mesh-geometry.ts` rather than read
out, because for mesh the count is produced downstream of the loader: the mesh is
resident in full and projection decides which faces reach the index buffer. On a
reveal ladder the wrapper overrides rather than sums it — the committed surface is
the revealed prefix's concatenation, a node-level fact, not a quantity each level
owns a share of.

**Performance tab.** Mesh was not a recognized node type, so its per-node
sessions fell through the aggregator's `Unknown` pass-through: one row per mesh
layer, while points / lines / gsplats each collapsed into a single row with a
summed element count. Mesh now aggregates the same way, and its count rides a
typed `TimingMetadata.triangles` (in `SUMMED_METADATA_KEYS`) instead of a
free-text `info: "N faces"` string — which was last-write on merge, so an
aggregated row over three mesh layers reported only whichever finished last. The
four per-type accumulator literals and the four-armed condition chain behind that
row are now one `NODE_TYPE_COUNTERS` table.

**Scene-graph tree.** A mesh node's badge tooltip gains the
`(M visible after slicing)` suffix its three siblings already had; the walk was
already stamping the count per path and `syncVisibleCountsIntoTree` was skipping
mesh. For a mesh the suffix means "how much of this surface the current nD slab
indexes" rather than a streaming residency — the node is resident either way.

**Fixed in passing:** `aggregateLoaderMetrics` hard-coded
`'point-spatial-index'` as the type of an empty aggregate, so a disposed lines,
gsplats or mesh ladder (dispose clears its level loaders) reported itself as a
points loader. The type is now supplied by the caller.

Mesh remains deliberately absent from the Memory tab's GPU-pool and accumulator
tables: those are keyed by `POOLED_GEOMETRY_TYPES`, the instanced-quad
element-texture path, and a mesh uploads its own `BufferGeometry` and keeps no
per-slice working set. Its resident bytes show up in the loader totals instead.

Verified on the reported demo: three `mesh-whole-node` loader rows
(`/earth/part_0`, `/earth/part_1`, `/earth clouds`) reporting 131,072 + 131,072 +
36,864 = 299,008 triangles, a `VISIBLE TRIANGLES 299.0K` card beside
`VISIBLE LINES`, and a single `Mesh · 299.0K tris · 3 nodes` row in the
Performance tab.
