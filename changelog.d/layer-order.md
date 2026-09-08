#### Authored cross-layer draw order (`layer_order`)

Overlapping order-dependent layers no longer have to accept an inferred draw
order. A layer may now state one: `layer_order`, an integer where **higher draws
nearer the camera** — the CSS `z-index` / Illustrator convention — composed
nearest-setter-wins like `blending_mode`, editable per layer in the Layers panel,
and surfaced in the data monitor's draw-order chip. Any sign, and bounded to the
JS safe-integer range — the attr is read by the viewer as a JS number, and past
2^53 - 1 two orders you separated would silently collapse into one band, so the
writer refuses that magnitude rather than recording an order the viewer cannot
represent.

Until now the order came from geometry. Groups sorted by mean view-z, and a
bounding-sphere containment rule then forced a container to draw before its
contents. That is a sound default and it is what keeps the multichannel
bioimaging demos stable today — but it is a property of where the splats landed
rather than of what the author meant, and it can rest on very little: on the
Acto3D heart, the vasculature/cardiac-tissue containment edge clears its test by
2.2 units out of a 1130-unit radius, 0.19%, so a refit that nudged either
channel's extent would silently flip the pair.

Layers on different levels never interleave, whatever the camera does. Unset
changes nothing anywhere — every group lands in one band, the comparator falls
through to today's, and the emitted order is bit-identical, which is why absence
is never stamped on disk. That is load-bearing rather than tidy: an authored
`0` orders exactly like an unset value, but the panel's blank `auto` state and
authored-band diagnostics must still distinguish the two.

It applies to every blending mode. A level on an `additive` layer is inert
against other commutative layers but decides where it sits relative to a
`volumetric` or `normal` one, which was previously not expressible at all —
commutative layers were pinned to draw first, always.

Refused where it would lie: a level authored strictly inside a `kind=partition`
or `kind=lod` group raises, through both the adder and a post-hoc
`node.attrs[...]` assignment. A partition's parts are ordered exactly against
each other from their stored BSP planes, and splitting the wrapper across bands
would destroy that; a LOD level would simply be inert. Set it on the wrapper —
the wrapper is the layer, and a level there moves the whole block while the
part order travels with it intact.

Three diagnostics say when a level is not doing what it looks like it is doing:
a band that splits a containment relation (the one way a level can wash out an
embedded layer), an authored opaque level at or above a transparent one
(`renderOrder` cannot cross that bucket split), and an order group whose members
disagree on their band.

Two demos now state their order — the Acto3D heart and the 2-channel neuromast.
The heart returns to `volumetric` with orders 3/2/1 for nuclei, vasculature, and
cardiac tissue. The neuromast uses low-absorption `volumetric` blending and
states an order that deliberately differs from the inferred one: the membrane
shell encloses the nuclei, but this fit gives the nuclei channel the marginally
larger bounding sphere (555.5 vs 546.4), so containment had it backwards.

What this does not do is make interpenetrating layers _correct_. For two concave
volumes no single order is right from every viewpoint, so a level pins such a
pair to one stated answer instead of a camera-dependent one — better, and
diagnosable, but not a substitute for per-element compositing. Design and
measurements: `docs/guides/specs/LAYER_ORDER_SPEC.md`.
