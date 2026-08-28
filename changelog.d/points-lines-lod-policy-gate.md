#### The LOD policy now gates Points and Lines too, not just gsplat fits

`test_lod_policy.py` has held gsplat demos to a deliberate topology choice for a
while, but its universe is `_FIT_CALLS` — so it never saw a Points or Lines node,
and that is exactly where the machinery had accumulated: six embedding demos with
three or four coarse replacement levels each, for nodes whose *resident* slice ran
five to twenty-nine times under its cap.

`test_points_lines_lod_policy.py` closes that. Any module requesting
`substitutive_lod=` or `partition=`, or hand-building an
`add_partition_group` / `add_lod_group`, must appear in `JUSTIFIED` with a reason.
An additive ladder needs no justification — it is the cheap default the rule
prefers. Seven entries today, each carrying its measurement: desi (9,751,955
resident, no hidden axis, genuinely past the cap, and its coarse levels are what
bound residency because `partition=` alone is eager), the nuclear pore (concave
interpenetrating subunits with no valid whole-object draw order), ocean currents
(547,000 resident whole-globe against 9.9M zoomed), gaia, dmri (2.1 GB vs ~200 MB
measured), zebrahub streamlines (nothing shipped) and biodiversity (an uncalled
helper).

Three properties keep it from rotting into a rubber stamp. It is a **shrinking**
allowlist: an entry whose module stopped using the feature fails the suite, so the
list cannot silently re-permit something. Every reason must cite a number, which
caught two of my own entries that argued from adjectives. And it is **fires-proof**
per entry — dropping any one of the seven from `JUSTIFIED` is asserted to turn the
gate red against the real corpus, so a gate that can never fail is distinguishable
from one that always passes.

`_lod_policy` gains the Points/Lines half of the rule alongside `stream_ladder`:
the per-geometry caps (Lines 2,793,472 / Points 5,591,040 / GSplats 4,194,304 —
there is no single ceiling, and comparing a Points node against the gsplat number
over-flags it by 1.33x), and the two ways a partitioned node's resident count gets
mismeasured. Measuring one `part_N` under-reports by the part count — that read
`nuclear_pore_complex` as 164,633 when it is 4,937,064, a 30x error that turned a
load-bearing partition into an apparently obvious removal. Summing each part's
largest slice over-reports, because different parts peak on different hidden
coordinates (5,708,398 for the same node, which crosses the cap and argues the
opposite way). Group globally by hidden coordinate first, then take the max.

Separately, `demo_gsplats_2d_cmu1_pathology`'s `create_luxar_scene` docstring is
corrected. It said the hosted archives "are still flat leaves"; in fact **two
generations exist under the same file names**. `data_manifest.json` pins both —
`sha256`/`bytes` for the in-repo LFS payload (38.2, 39.2, 36.4 MB) and
`hosted_sha256`/`hosted_bytes` for what the record serves (84.5, 93.2, 100.3 MB) —
and `data_fetch` resolves the in-repo one first, so a checkout with a stale LFS
object keeps building from the older generation. The in-repo copy is a flat
laddered leaf; the hosted one carries a partition. This demo therefore builds a
different scene shape per host, with nothing visible in a diff, and a locally
measured topology says nothing about what a fresh machine produces. The
discriminator is to hash the file and see which digest it matches; byte size is
only suggestive.
