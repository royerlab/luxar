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
Measured twice independently on real partitions, as bytes of root `zarr.json` per
part: about 8-10 KB for a bare-leaf part (a real `gsplat partition`: 9 KB), about
48-54 KB when each part carries a 6-step `stream` ladder, about 100-110 KB for an
`adaptive`-shaped part (3 levels x 4 sub-LODs). Taking the largest-per-part end,
and since the recommended recipes multiply nodes per part by 6-12x, the old cap
was crossed at roughly 400 bare-leaf parts but at only ~78 laddered ones and ~37
`adaptive` ones — so this was reachable by a fairly modest partition, not just by
a huge `batch-fit merge`.

Past that point the peek returned `{}`, which is indistinguishable from "this
dataset authored no appearance" — so every rewriting command wrote its own
defaults over the authored look of a zipped or tarred partition, silently, while
the *same tree as a directory store* answered correctly. Two input shapes
disagreeing is the part that made this hard to see.

The single budget is now two, chosen by the document's name: a `.zattrs` keeps
the small 4 MiB one (nothing about format 2 changed), and a format-3 `zarr.json`
gets 128 MiB — roughly 1,200-2,500 parts of headroom on the laddered shapes
people actually publish, ~13,000 on bare leaves, and no more than that because
the document is parsed whole, so the transient parse rather than the byte count
is the real ceiling.

Raising the document budget does not raise what the peek can hand back: the
*attributes* a member unwraps to are re-checked against the small budget. That
re-check is triggered by the member's own **bytes** — it runs exactly when
`len(raw) > 4 MiB`, i.e. exactly when the raised budget was actually used — and
not by the document's name, because the raised budget is the only thing it exists
to bound. Three consequences, one of them arithmetic rather than a convention: a
`.zattrs` can never be re-checked at all (its member budget *is* 4 MiB, so the
member gates refuse first); a `zarr.json` inside 4 MiB is not re-checked either,
so nothing the old single budget admitted is newly refused; and a `zarr.json`
over 4 MiB is, which is the whole point. That last distinction is load-bearing
rather than tidy: `json.loads` is not round-trip-length-preserving (`{"a":1e10}`
is 10 bytes and comes back 19), so a 2.61 MiB document of exponent literals
measures 4.24 MiB re-serialized and a name-keyed re-check would refuse it for
nothing. The re-serialization is also compact and `ensure_ascii=False`, so
non-ASCII attributes are not inflated 3x by escaping the way a default
`json.dumps` inflates them — which is what makes the re-check that *does* run
safe. The budget itself, and the word a refusal uses for it, stay keyed on the
document's *name*, never on comparing budget values. The tar path also now bounds
its read to the budget instead of trusting the header size, which is what the zip
path already did.

The remaining headroom is finite, so a refusal is no longer silent: every size
refusal now warns, naming the archive, the member, the measured size and the
budget it crossed. An invisible `{}` was the whole reason this took so long to
find.
