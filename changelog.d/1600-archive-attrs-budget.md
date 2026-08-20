#### A large archived partition no longer peeks as "no appearance authored"

The authored-appearance carry reads the input's root attributes, and for a
`.gsplats.zarr.zip` / `.tar.gz` it does so by *peeking* — one member out of the
archive, no extraction. That peek refused any metadata member over 4 MiB, on the
stated grounds that "a zarr group's attrs are a small JSON object". True of a
format-2 `.zattrs`, which literally *is* the attributes mapping; false since
Luxar moved to zarr format 3, where the root document is `zarr.json` and carries
the node's structural fields plus — at a consolidated root, which every Luxar
store is — the entire consolidated index of the tree.

That document therefore grows with the *shape* of the store while its
`attributes` stay a handful of scalars, and it grows per **node**, not per part.
Measured on real partitions, as bytes of root `zarr.json` per part: about 8 KB
for a bare-leaf part (a real `gsplat partition`: 9 KB), about 48 KB when each
part carries a 6-step `stream` ladder, about 100 KB for an `adaptive`-shaped part
(3 levels x 4 sub-LODs). Since the recommended recipes multiply nodes per part by
6-12x, the old cap was crossed at roughly 500 bare-leaf parts but at only ~87
laddered ones and ~42 `adaptive` ones — so this was reachable by a fairly modest
partition, not just by a huge `batch-fit merge`.

Past that point the peek returned `{}`, which is indistinguishable from "this
dataset authored no appearance" — so every rewriting command wrote its own
defaults over the authored look of a zipped or tarred partition, silently, while
the *same tree as a directory store* answered correctly. Two input shapes
disagreeing is the part that made this hard to see.

The single budget is now two, chosen by the document's name: a `.zattrs` keeps
the small 4 MiB one (nothing about format 2 changed), and a format-3 `zarr.json`
gets 128 MiB — roughly 1,300-2,800 parts of headroom on the laddered shapes
people actually publish, ~16,000 on bare leaves, and no more than that because
the document is parsed whole, so the transient parse rather than the byte count
is the real ceiling. Raising the document budget does not raise what the peek can
hand back: a document read under it has the *attributes* it unwraps to re-checked
against the small budget (a `.zattrs`, already bounded by that same number as a
member, is not measured a second time — the re-check measures a re-serialization,
which for non-ASCII attrs is 2-3x the bytes on disk and would refuse something
the member budget just admitted). The tar path also now bounds its read to the
budget instead of trusting the header size, which is what the zip path already
did.

The remaining headroom is finite, so a refusal is no longer silent: every size
refusal now warns, naming the archive, the member, the measured size and the
budget it crossed. An invisible `{}` was the whole reason this took so long to
find.
