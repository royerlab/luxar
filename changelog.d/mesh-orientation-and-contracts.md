#### `add_mesh` warns when `dim_order` reverses face handedness (#2141)

`dim_order` renumbers a mesh's vertex columns, and an orientation-reversing
permutation reflects space: since `cross(Ra, Rb) = det(R)·R·cross(a, b)`, a
triangle's geometric normal is negated while its stored corner order is
untouched. Faces wound counter-clockwise in the caller's own column order come
out clockwise in the scene's, and with `double_sided=False` the surface renders
inside-out — an open surface vanishes. With stored normals the mismatch also
flips the shading gradient through `gl_FrontFacing`, even when both sides draw.

Whether that is a *defect* depends on which frame the caller wound in, and the
store cannot tell. `normal_dims` names SCENE dimension indices — the layout after
`dim_order` — so §3.2's winding frame, `sorted(normal_dims)`, is a scene triple:
a caller who read the contract literally wound against the scene frame and is
already correct. Flipping their faces to help the other caller would corrupt
them. So the writer warns, names the consequence, and gives the remedy
(`faces[:, [0, 2, 1]]`), in the same warn-only posture as the unwelded-vertices
lint. It stays quiet when there are no `normals` (no declared frame, and the
viewer already renders such a mesh double-sided).

The gap this closes is documentation as much as behaviour: the spec mentioned
`dim_order` only in a done-list line, and §3.4 warns at length about normals
stored against the wrong axes without covering the one operation that renumbers
axes. §3.7 now specifies the whole interaction — including, explicitly, that
`normals` needs **no** companion transform, because its components are already in
the destination frame. That is where mesh differs from gsplats, whose Cholesky
factors *are* authored in the source frame and must be carried through the map;
the asymmetry is in the contract, not in the adders. An `apply_dim_order_normals`
would be a double transform, and would break precisely the callers who got it
right — `demo_lsystem_forest` among them, which passes `normal_dims=[2, 3, 4]`
against four authored columns.

#### Mesh winding parity is decided when the stored-normal shader is active (#2142)

`resolveWinding` returned early for a double-sided mesh, on the grounds that
"both orientations draw, so parity is unobservable". That holds for
rasterization coverage and not for `gl_FrontFacing`, which the stored-normal
fragment variant reads to flip the interpolated normal toward the camera. In an
odd-parity epoch every projected winding is reversed, so the flip lands on the
wrong side and the surface shades with an inverted gradient collapsing toward
`uAmbient` — the exact artifact the flip exists to remove. §5.4 already said as
much, in the sentence ruling out the alternative `side`-swap implementation.

`double_sided` defaults true, so the exemption covered the default
configuration. `reverse` is now decided whenever winding is observable at all —
when the material culls, or when the stored-normal variant is active — while
`side` still follows the node's own request and the undecidable-frame notice
stays gated on it, so a double-sided node gains no new log line.

This is latent rather than user-visible: `displayDims` is built ascending at both
construction sites and nothing reorders it at runtime, so the reversal branch is
currently unreachable. It is fixed because the machinery is a defense against a
reordering the viewer does not yet produce, and one arm of that defense was
wired backwards — in the default configuration — so it would have failed on
arrival. The test that pinned the old reasoning now pins the corrected version.

#### The mesh slab tolerance is authorable (#2143)

Mesh draws a triangle only when all three of its vertices fall inside the nD
slice slab, so on a *continuous* hidden dimension it shows a slab of finite
thickness rather than a cross-section. The spec calls the slab thickness "the
only control a mesh has" — and nothing set it. `ToleranceOptions.meshSlabTolerance`
was read by exactly one function, supplied by exactly one unit test, and reached
from no production caller: `processMeshData` called `computeTolerance('mesh', …)`
with no options at all.

It is now an authored, mesh-only node attr, `slab_tolerance`, measured in cells
of the hidden dimension's own step and defaulting to one cell. Non-positive
values are refused, and that refusal is load-bearing rather than tidy: a zero
slab reduces whole-triangle membership to exact float equality with the slice
plane and the node renders nothing, which is the same trap that stops mesh
reusing the Lines tolerance arm.

No Layers-panel slider. Changing this re-culls rather than updating a uniform,
unlike the five appearance attrs it sits beside, and that is the honest boundary.

#### Authoring refuses a mesh the viewer provably cannot load (#2145)

`validate_vertices_for_writing` states the principle — "a bound that exists only
on the read side is not a bound" — and mirrors the 2²⁷ vertex cap that the pick
vote key needs. But the bound that actually rejects real meshes is the viewer's
512 MiB per-node decode budget, which bites around 4.6M vertices: roughly 29×
tighter, and with no write-time counterpart at all, not even a warning. A 10M-vertex
surface wrote cleanly and then failed to load, in a browser, far from the
`add_mesh` call that caused it.

`validate_mesh_decode_budget` closes that, and is deliberately weaker than the
loader's gate. A hard error that over-counted would refuse a store the viewer
would happily accept, which is a worse bug than the one it fixes, so it charges
only each array's decoded size — dtype-independent, with a broadcast colour
charged at its real `n_vertices` expansion — and omits the stored bytes, the
largest per-chunk allocation, and the label CSRs. Every omission is a term the
loader adds, so it is a strict lower bound: anything refused here is certainly
refused there. The cost of that soundness is completeness — a mesh between about
half the ceiling and the ceiling still writes — and the error message says so
outright, since a user who splits to just under the reported figure and is
refused again would fairly call the first message a lie.

One shortfall in that accounting was *multiplicative* rather than a bounded
constant, and is closed rather than documented: the viewer concatenates a reveal
ladder's `additive_<i>` levels into one node's buffers and keeps all of them
resident, so it charges their SUM against a single budget. The write side
otherwise never saw the sum — the flat check runs once on the authored mesh and
each level re-checks only itself — and a shell ladder duplicates every boundary
vertex, so a six-level radial ladder runs several times the flat surface. A
comfortably under-budget mesh could therefore still write a ladder the viewer
refuses. `validate_mesh_ladder_decode_budget` charges the levels together, once,
from the payloads the writer is about to emit, and shares its per-mesh accounting
with the flat check so the two cannot drift.

The viewer-side constant's comment claimed there was no Python twin because the
write side cannot know what a tab survives. That is true of this gate's *primary*
job, policing hostile `?src=` stores, and the two purposes turned out to be
separable; the comment now records the partial twin and the direction of its
weakness.

#### Mesh documentation: the whole-triangle cull is finally user-facing (#2144)

§5.3 of the spec says the ragged-boundary limitation "must be documented in the
user guide, not glossed". It never was — the mesh section of the format guide
described the on-disk layout and said nothing about slicing, `VIEWER_GUIDE.md`
mentioned mesh once in a list, and the nD navigation tutorial not at all. There
was no §8 checklist item for the obligation, so it was never ticked and never
missed.

The format guide now has an *nD slicing: whole-triangle cull* section covering
the rule, the ragged edge, the thick slab, `slab_tolerance`, why mesh is the only
type whose slab is tunable, and when to reach for Gaussian splats instead;
`VIEWER_GUIDE.md`'s nD Navigation section cross-references it, since every other
rule there describes per-element visibility. §6.4's snapshot inventory, which
read 37 pairs against a tree holding 39, is corrected.
