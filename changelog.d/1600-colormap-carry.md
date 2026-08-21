#### An ancestor-authored `colormap` now reaches the leaf

A `colormap` set on a Group (or on a scene / `.gsplats.zarr` root) never
rendered. Two independent causes, each of which made the other invisible: the
writer manufactured `colormap="gray"` on every colorless gsplats leaf, which
sits nearer the leaf than the authored value, and the viewer never composed the
attr at all — it survived only as a raw per-node pass-through, so every consumer
read the leaf's own value.

`apply_gsplat_group_attrs` now stamps the gray default only when nothing above
authored a palette. The two write paths answer that question differently and
both are covered by one rule: the scene compiler walks the store (a Group's
attrs are written before any child leaf exists), while the standalone
`.gsplats.zarr` tree writer — which emits a wrapper's attrs *after* its children
— threads the value down the recursion the way `coverage_fraction` already
does. On the viewer side `colormap` joins `blending_mode` / `join` as a
nearest-setter-wins composed attribute, with a `'custom'` palette's LUT bytes
travelling alongside the name so an inherited custom palette actually renders
(and so a name and a LUT can never be paired from two different nodes).

With the shadow gone, `colormap` also joins `AUTHORED_APPEARANCE_ATTRS`, so a
structure-only rebuild (`gsplat lod` and the rest of the rewriting family)
carries the source root's palette onto the result instead of dropping it. The
one exception is the `'custom'` sentinel: it names a sibling `colormap_lut`
array that the attrs-only carry cannot reach, so it is dropped with a message
pointing at `gsplat convert --colormap` rather than written as a dangling
reference. `amplitude_data_range` / `scalar_data_range` stay excluded — each
leaf re-derives them from its own post-reduction values, so composing them
across a `kind=lod` boundary needs a basis gate first.

A Group is now a legitimate place to author a palette, so
`LuxarZarrCompiler.write_group` resolves an ndarray LUT or a
matplotlib/colorcet name into a sibling `colormap_lut` array there too, exactly
as the leaf writers do.

An inherited palette applies where the geometry can use one: Points / Lines /
Mesh still require `has_scalars`, while a GSplats leaf is always
colormap-capable — its amplitude *is* the scalar — so an ancestor's palette
overrides even per-splat colours there, matching what the Layers panel's
colormap dropdown already does when fanned out over a group.
