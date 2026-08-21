#### A demo's on-scene credit can no longer contradict its own citation

A demo credits its data twice: structurally in `DEMO_META["citation"]`, and
visibly in the `scene.add_text(...)` footer painted onto the scene — which is
the credit a gallery tile actually shows. Nothing kept the two in step, and the
drift was not hypothetical: three footers named the wrong paper and were fixed
by hand in the previous pass, found only because someone happened to read them
side by side.

`test_demo_meta.py` now sweeps every demo that declares a real credit and fails
when an overlay string makes a claim its own `citation["short"]` does not back.
Extraction is static (`ast`, never an import, like the rest of the registry) and
covers plain strings, module-level constants, concatenation, and f-strings —
for an f-string only the literal segments are read, with each `{...}` hole left
in place as an explicit break so two fragments that are never adjacent on screen
cannot be spliced into a credit nobody wrote.

The comparison is deliberately narrow, because a credit footer states only two
things a machine can check: a year, and the leading name of the group in front
of it. Both must appear in the declared `short`, compared case- and
accent-insensitively (`Leike & Enßlin 2020` matches `Leike et al. 2020`). Only
the *leading* name of a group is required, never every name in it, and a bare
separator (`•`, `—`) ends the group — the rule that keeps `3D UMAP • Kim et al.
2024` from reading "UMAP" as an author. The corpus shapes that a naive rule gets
wrong are pinned as their own tests rather than merely observed to pass.

#### The gallery manifest carries each tile's credit

`scripts/gallery/manifest.json` gains an optional `citation` string, copied
verbatim from the demo's `DEMO_META["citation"]["short"]` for the 25 gallery
demos that declare one. A tile's attribution is now reviewable in a diff instead
of only on the rendered page.

The manifest cross-check pins it in both directions: a demo with a credit whose
manifest entry has none is a tile that silently dropped its attribution, and a
manifest entry with a credit its demo does not declare is an unsourced claim in
front of readers. A demo that declares no credit therefore carries no key at all
— absence means "unknown or not yet recorded", which is deliberately not the
same statement as `None`'s "procedurally generated, nothing to credit". Nothing
reads the field at capture time; it exists to be reviewed and to be pinned.
