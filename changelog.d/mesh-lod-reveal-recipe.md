#### `luxar mesh lod --recipe reveal`: the mesh reveal ladder is authorable from the CLI

The reveal ladder shipped complete — format, writer, viewer, end-to-end guard — and
unreachable to anyone not importing `luxar` in a script. `add_mesh(additive_lod=…)` was
the only entry point, and `-m/--add-method` had been *reserved* for this by the August
flag rename and never filled in. It is filled in now.

`--recipe` selects between the two ladders rather than the knobs composing, because a
mesh has no coarse prefix — a prefix of an arbitrary index buffer is a surface with
holes, not a simpler surface — and `add_mesh` refuses the two together. `levels` is the
default, so every existing invocation is untouched. A knob aimed at the other recipe is
refused by name, and the message names the flag that does the same job under the recipe
you chose, rather than dropping the flag and answering a different question.

`-m` moves with it. It was held by the hidden legacy option while it was reserved; it now
names the additive ordering, as on `gsplat lod`. The migration survives the move: `-m
cluster` is recognised as a decimation method and pointed at `--subst-method cluster`,
value carried.

`--reveal-center` / `--spatial-dims` are the same flags with the same parser as
`gsplat lod`, extracted to `cli/reveal_options.py` so the two commands cannot drift on
what a centre — or a `--spatial-dims` ORDER, which pairs with the centre's coordinates —
means.

Also fixed, in passing: the Points, Lines and Mesh ladder writers popped the private
`_skip_scene_bounds` flag *below* `group.attrs.update(attrs)`, so every sub-LOD of every
ladder carried it on disk. An internal flag in the on-disk format, and one that
round-trips — a tool reading a level's attrs and re-writing them handed it back as a
caller attribute.
