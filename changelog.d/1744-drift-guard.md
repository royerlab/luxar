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
covers plain strings, module-level constants, `+` concatenation, the two branches
of a conditional, and f-strings — for an f-string only the literal segments are
read, with each `{...}` hole left in place as an explicit break so two fragments
that are never adjacent on screen cannot be spliced into a credit nobody wrote.
An overlay is not only an `add_text` call in the demo's own file: a `credit=`
argument to any call is read too, because the six `gsplats_interop` demos paint
their footer through a shared helper and a demo-file-only reader would have
reported green on them having read nothing.

The comparison is deliberately narrow: a credit footer states two things a
machine can check, a year and the names in front of it. The year must appear in
the declared `short`, and so must at least one of those names, matched on
unicode token boundaries (so `Kim` is not satisfied by `Kimura`, nor `Tan` by
`Tanø`) and case- and accent-insensitively — `_fold` casefolds first, so `Enßlin`
and `Ensslin` are the same name, then strips combining marks so `Muñoz` and
`Munoz` are too. Any name in the group will do, because a footer routinely leads
with a title or a dataset name — `Dip-C, Tan et al. 2018` — and requiring the
leading one turns a correct credit red as soon as an edit deletes a lowercase
word from it (the real dMRI footer is one such word away). The cost is stated in
the test: five of the credited demos have a `short` carrying more than one
name-shaped token, and for those a wrong author at the right year still passes.
A year with no name in front of it is a number, not an attribution, so
`2048³ voxels` is left alone.

What it does not catch, stated plainly: a footer that names the wrong dataset
without giving a year; a credit assembled at runtime (an f-string hole, a list
joined later — the historical `esm3_protein_landscape` case); a demo that
declares no credit at all, which has no declared fact to contradict (the
historical `mouse_multiome_peak_umap` case, where the hazard was an uncredited
demo painting a credit); and a footer that names a dataset the `short` form also
names while attributing it to the wrong author. Of the 26 demos that declare a
credit, 8 paint a year-bearing one today — those 8 are judged and the rest are
visited (every count here describes the corpus at the time of writing; nothing
derives them). The corpus shapes a naive rule gets wrong are pinned as their own
tests, and so is the set of credits the real corpus is known to yield, so the
sweep cannot quietly stop reading anything.

#### The gallery manifest carries each tile's credit

`scripts/gallery/manifest.json` gains an optional `citation` string, copied
verbatim from the demo's `DEMO_META["citation"]["short"]` for the 26 gallery
demos that declare one. A tile's attribution is now reviewable in a diff instead
of only on the rendered page.

The manifest cross-check pins it in both directions: a demo with a credit whose
manifest entry has none is a tile that silently dropped its attribution, and a
manifest entry with a credit its demo does not declare is an unsourced claim in
front of readers. A demo that declares no credit therefore carries no key at all
— absence means "unknown or not yet recorded", which is deliberately not the
same statement as `None`'s "procedurally generated, nothing to credit", and a
written-out `null` is rejected rather than read as either. Nothing reads the
field at capture time; it exists to be reviewed and to be pinned.

A `script: null` entry no longer escapes the cross-check either. Such an entry is
resolved by `id` against the demo key, and once it resolves the manifest and the
demo are two descriptions of the same thing, so every comparison runs on it —
credit and structure alike. All three of today's scriptless entries resolve, and
both halves found something: the TotalSegmentator README tile had lost the credit
its demo declares, and the cryo-EM virus capsid tile was filed under `microscopy`
against the `structural` its demo (and both its structural-biology peers)
declares. The manifest is corrected to `structural`.
