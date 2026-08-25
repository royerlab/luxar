#### `dim_order` now carries a mesh's normals and face winding (#2141)

`add_mesh(dim_order=[...])` renumbers the vertex columns into the scene's
dimension order. It left the two arrays that describe a *direction* in those
columns untouched, so the resulting store was silently wrong: it passed every
validator, the viewer accepted it, and it rendered with the wrong shading and —
for an orientation-reversing permutation — inside-out geometry.

Measured on one triangle whose normal lay along authored axis 0,
`dim_order=["z","y","x"]` produced a stored normal exactly **orthogonal** to its
own face, and a face whose cross product had flipped sign. Both follow from the
same omission. `normals` components are positionally bound to `normal_dims`, so
renumbering the columns without remapping the label leaves the component
describing a different axis. And because `cross(Ra, Rb) = det(R)·R·cross(a, b)`,
a reversing permutation negates the geometric normal while the stored winding
stays as authored — which inverts the shader's `gl_FrontFacing` normal flip and,
on a `double_sided: false` mesh, turns an open surface inside out.

`dim_order=["z","y","x"]` — a `(Z,Y,X)` volume into an `(x,y,z)` scene — is the
standard microscopy call, and it is one of the reversing ones.

The fix is the companion transform the type was missing. GSplats has carried its
orientation data through `dim_order` since it gained the kwarg
(`apply_dim_order_cholesky`, applied right after the positions); Points and Lines
carry no orientation data and need nothing. Mesh has two such arrays and had one
call, and `apply_dim_order_orientation` is now its peer: it remaps `normal_dims`
through the same forward map, permutes the components so the triple stays
ascending (the viewer uses stored normals only when `normal_dims` equals
`displayDims` *in order*, so leaving it unsorted would have swapped one silent
failure for another), and flips winding when the map restricted to
`sorted(normal_dims)` is odd. It runs once above the four structural branches, so
the flat, `partition=`, `substitutive_lod=` and `additive_lod=` routes all
inherit it; `split_mesh_by_faces` preserves corner order and the decimators
re-derive normals from the stored winding, so coarse levels follow for free.

Two carve-outs are deliberate and pinned by tests. A mesh with no `normals` is
left alone, because `sorted(normal_dims)` is the only declared winding frame
there is — flipping on a guess would be worse than not flipping, and the viewer
already renders such a mesh double-sided. And a `normal_dims` entry that indexes
no authored column is now refused rather than range-checked against the scene
width, where it would pass while binding a component to an axis the caller never
supplied.

The interaction was unspecified: `MESH_NODE_SPEC.md` mentioned `dim_order` only
in a done-list line, and §3.4 warns at length about normals stored against the
wrong axes without covering the one operation that renumbers axes. §3.7 now
specifies it.

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
