# luxar.core.group.gsplats_pipeline

High-level write path for Gaussian splats. These free functions back the
public `Group.add_gsplats_from_data`, `add_gsplats_from_file`, and
`add_gsplats_from_volume` methods: the orchestrator in
[`core/group/group.py`](../group.py) keeps the public signatures and
docstrings, then delegates the body to the `*_impl` functions here.

## Overview

A fitted (or loaded) [`GSplatData`](../../../gsplats/gsplat_data.py) object
can carry up to two orthogonal LOD axes — a **substitutive** hierarchy
(coarser levels *replace* finer ones) and an **additive** ladder (later
sublods *add to* earlier ones). This package resolves those two axes and
routes the result to one of three write shapes:

| Resolved shape | Write path | On-disk result |
|----------------|------------|----------------|
| multi-substitutive | `add_gsplats_as_lod_group_impl` | a `kind=lod` `Group`, one gsplats child per level |
| single-substitutive + multi-additive | `add_gsplats_multi_lod_impl` | one gsplats leaf with `additive_<i>/` subgroups |
| single-substitutive + single-additive | `Group.add_gsplats` | a flat single-leaf gsplats node |

The substitutive axis is resolved first (it can produce a multi-level
result), then the additive axis (uniform across levels).

## File Structure

```
gsplats_pipeline/
├── __init__.py        # package docstring only (no re-exports)
├── amplitude_norm.py  # one insertion-time amplitude scale per structure
├── from_data.py       # add_gsplats_from_data_impl — top-level dispatch
├── from_io.py         # add_gsplats_from_file_impl / add_gsplats_from_volume_impl
└── lod_dispatch.py    # add_gsplats_as_lod_group_impl / add_gsplats_multi_lod_impl
```

## Entry points

### `amplitude_norm.py` — insertion-time amplitude normalization

The public data, file, and volume doors normalize fitted raw-unit amplitudes
before writing a scene. ``True`` / ``"auto"`` maps a robust p99.9 reference to
1.0 only when it exceeds 1.0, ``False`` opts out, and a positive number sets an
explicit target. A child inserted directly into a ``kind=lod`` or
``kind=partition`` group defaults to no normalization because its exposure must
stay shared with its siblings; whole structures still use one pooled factor.
Partition parts share one pooled reference; substitutive LOD groups contribute
only their finest child to that reference, while the resolved factor still
scales every level and additive rung. Applied factors are stamped as
``amplitude_normalization_factor``.

### `from_data.py` — `add_gsplats_from_data_impl(group, *, name, result, ...)`

The dispatch hub. Validates that `result` is a `GSplatData`, propagates
`truncation_radius` from the data into `attrs` (unless the caller overrode
it), then resolves the two LOD axes via
[`core/group/lod/gsplats.py`](../lod/gsplats.py):

```python
result, explicit_coverage_fractions = \
    resolve_substitutive_axis_gsplats(result, lod_group)
result = resolve_additive_axis_gsplats(result, additive_lod)
```

Branch selection then keys off `result.n_substitutive` and
`result.n_additive_sublods`. Passing `coverage_fraction` while the resolved
result is multi-substitutive raises `ValueError` — thresholds are derived
per-child instead (or set via `lod_group=dict(coverage_fractions=[...])`).

#### An explicit `None` means absent (#1496)

Everything that is not a named parameter of `add_gsplats_from_data_impl` arrives
in `**attrs` and is forwarded verbatim to children, where `validate_render_attrs`
rejects an unknown key by NAME and never looks at its value. So a
present-but-`None` key is not the same thing as an absent one, and the idiomatic
`partition=maybe_partition` call refused from inside `child_0` with the
`kind=lod` wrapper already on disk. Two entry-point calls, run in this order at
the top of `add_gsplats_from_data_impl` — and at the top of
`add_gsplats_from_file_impl` and of `graft_gsplat_node` in `from_io.py` — settle
it:

1. `strip_absent_attr_kwargs(attrs, ABSENT_WHEN_NONE_ATTRS)` (the shared helper
   in `core/group/compositing.py`, which `from_data.py` and `from_io.py` each
   import from there directly; only the `ABSENT_WHEN_NONE_ATTRS` tuple is
   defined here and imported by `from_io.py`) deletes every
   `ABSENT_WHEN_NONE_ATTRS` key valued `None`:
   - `labels`, `image_labels`, `partition`, `colors` — named params of the leaf
     `Group.add_gsplats` defaulting to `None`, so `None` already means "absent"
     one level down;
   - `truncation_radius` — which this module *injects* from
     `result.truncation_radius`, so an explicit `None` must mean "no override"
     rather than clobbering the data's own value;
   - `colormap` and `coverage_fraction` — the two render attrs whose `None` no
     value validator catches, so it reached disk and wrote something **wrong**
     rather than refusing. `validate_render_attrs` guards its colormap check on
     `is not None`, and `compositing.sync_custom_colormap_attr` then rewrites the
     `None` to `'custom'` with no `colormap_lut`, which the viewer answers by
     warning and falling back to viridis — measured, `colormap=None` wrote
     `'custom'` where the omitted key writes `'gray'`. `coverage_fraction=None`
     wrote a literal `coverage_fraction: null` selector threshold.
     Every *other* render attr is deliberately left refusing a `None` (`opacity`
     / `gamma` / `intensity` / `absorption` "must be convertible to float, got
     NoneType", `blending_mode` "must be a string", `layer` / `visible` "must be
     a boolean", `scalars` an unknown attribute) — those are loud, so reading
     their `None` as "absent" would only mask typos.
2. `reject_data_owned_channels(name, attrs)` refuses a `colors` key **with a
   value**, or a `centers` / `amplitudes` / `cholesky_factors` key **at all**
   (asked in that order), with a `ValueError` naming the collision — this adder
   passes all four positionally from the `GSplatData`, and on a split route they
   cannot be sliced per child anyway. Previously the flat route leaked Python's
   raw `TypeError` ("got multiple values for keyword argument 'colors'") and the
   other three answered the misleading `Unknown node attribute 'colors'. Did you
   mean 'colormap'?`.

   The asymmetry is real and worth stating precisely: this step tests `kwarg in
   attrs` with no value check, so `amplitudes=None` and
   `cholesky_factors=None` are refused as well — those three are required
   positional params of the leaf adder with no `None` default, so no "absent"
   reading exists for them and `amplitudes=maybe_amps` is *not* a safe call form.
   `colors` looks like an exception only because step 1 already deleted a `None`
   one. That also makes `colors` the one key here whose refusal message is not
   literally true of every value ("cannot be passed as a keyword here", yet
   `colors=None` is passed and accepted), so its message carries an extra clause
   saying so.

`ABSENT_WHEN_NONE_ATTRS` is *derived* from two sources, so no part of it can
drift. From `GATE_FORWARDED_LEAF_PARAMS`, the node-attrs gate's own exclusion
tuple: the two answer different questions, and the gate's set is "leaf named
params this adder forwards onward STRUCTURALLY" — `colors`, `truncation_radius`,
`colormap` and `coverage_fraction` are deliberately **not** in it, because a
non-`None` value of each must still be judged (a collision, a legitimate
override, a colormap name to validate, a real threshold). And from
`compositing.ABSENT_WHEN_NONE_RENDER_ATTRS`, for the last two: their `None` is a
render fault every adder shares, not this module's property, so since #1574 the
four leaf adders strip the same tuple at their own entries and this set adds to
it rather than restating it.

Both calls sit at the adder ENTRY, so they outrank everything below —
`_reject_before_wrapper`'s `dim_order` spec validators and rank guard, and the
`coverage_fraction` refusal. That is deliberate: a channel collision means the
adder cannot build the call it is about to make. In `from_io.py` they run once at
the top of `add_gsplats_from_file_impl` rather than being left to its two
branches, because the graft branch reaches its own copy only *below* that method's
`dim_order` refusal and stored-column-count check — so with the pair left to the
branches, one public method gave two different verdicts for the same mistake
depending on whether the file happened to be matrix-shaped.

Excluding a key from the gate does not excuse its VALUE, and `partition` is the
one excluded key with a value worth judging. It gets an explicit
`partition.resolve_partition_spec` call — the same function the leaf adder calls,
here for its verdict alone — at each of the two doors that build a wrapper before
the first leaf write (#1550). Without it an invalid spec (`partition="nonsense"`,
`{"max_elements": 0}`) rode the exclusion straight down into `child_0` /
`part_0` and was refused only after the `kind=lod` / `kind=partition` wrapper
existed, which then survived `finalize()`. Note this one is NOT an entry-level
call like the pair above: on the `lod_group=` door it sits INSIDE
`_reject_before_wrapper`, in the slot directly below the node-attrs gate and so
below the `dim_order` spec, rank and colours checks — because that is the flat
path's own order (the leaf validates node attrs at its entry and resolves the
spec inside its partition branch, further down), and precedence parity with the
flat path is the whole point. On the graft door (`from_io`'s
`_reject_a_bad_partition_spec_on_a_graft`) there is nothing above it to be below,
so it joins the entry pair. Two values are skipped at both doors, because the
leaf never judges them either: `False` (the explicit no-partition bypass, which
every adder normalises away before resolving) and a sub-2-D width (where
`warn_if_partition_needs_more_dims` DROPS the request with a warning).

A VALID spec has one conflict of its own, at BOTH doors: `partition=` cannot ride
a node that also carries an additive ladder, because the multi-LOD writer has no
`partition` parameter at all — so the key reached `validate_render_attrs` as an
unknown node attribute, on the multi-substitutive route from inside `child_0`
with the wrapper already written, and on the graft door from inside `part_0` with
the `kind=partition` already written. One reason template
(`partition_beside_a_ladder_reason`) with a structure and a remedy per door, the
same convention `labels_on_wrapper_reason` keeps: the `additive_lod=` door says
"drop one of the two", the file/graft door says `gsplat flatten`, because there
is no `additive_lod=` in an `add_gsplats_from_file` call to drop.
`resolve_partition_beside_an_additive_ladder` holds the data door's half, above
the route branch so both routes answer alike; `_reject_a_partition_beside_a_stored_ladder`
holds the file door's, above the spec-shape gate at both of its call sites so the
two doors rank the two faults the same way. Which remedy the file door names is
decided per call rather than fixed, and by the OFFENDING leaf's store alone: a
leaf whose store carries a ladder gets `gsplat flatten` (that ladder survives
dropping the kwarg, so "drop one of the two" would only buy a second refusal),
and one laddered solely by the call's own `additive_lod=` gets "drop one of the
two". "Offending", not "anywhere in the tree": on a mixed store where the kwarg
COLLAPSES the laddered leaf and LADDERS the flat one, the blocking leaf is the
flat one, so the message names `additive_lod=` while a stored ladder sits
untouched next door. That is the right answer — `gsplat flatten` would rewrite a
ladder that is not in the way, and dropping either half of "drop one of the two"
really does resolve it.

`partition=False` is not a request and is never refused — but it stranded the
same wrappers all the same, arriving at the multi-LOD writer as an unknown node
attribute. It is DELETED instead, and only when that writer is the destination:
`False` is the `resolve_auto_partition` bypass everywhere a leaf adder resolves
it, so stripping it earlier would silently re-enable a compiler-level
`auto_partition_max_elements`.

The file door's half runs before any data exists, so it has to ask what the
ladder WILL be rather than what the store holds — and `additive_lod=` moves that
answer in both directions (#1632). It can take a ladder away: `False`, or any
dict resolving to a single rung, flattens every level before the data door asks
the same question, so refusing on the stored ladder would refuse a call that
partitions perfectly well. And it can add one the store has not got:
`additive_lod=` is not a parameter of `graft_gsplat_node` at all, it rides in
`**attrs` down to each part's own `add_gsplats_from_data_impl`, which BUILDS the
ladder there — one level below the `kind=partition` wrapper the graft has
already written, which is exactly the strand this gate exists to prevent.

So each leaf is asked through `lod.gsplats.resolve_additive_rungs`, which states
`resolve_additive_axis_gsplats`'s vocabulary a second time as an ordering-free
rung COUNT (`None`/`True` → stored, `False` → 1, `dict` → stored unless the
resolver would compute, in which case `gsplats.lod.additive.additive_rung_count`
over the same `n_lods=4` default). Presence of the kwarg is not the question:
`{"n_lods": 1}` is one rung and legitimate, `{"method": "radial"}` carries no
`n_lods` to read and takes the `n_lods=4` default — four rungs on a leaf of at
least 4 splats, fewer on a smaller one, where equal-count cuts clamp to `n`.
`additive_rung_count` shares `make_additive_lod`'s own cut resolver, which is
what makes equal-count / `stream:` / explicit counts exact rather than
re-derived.

An explicit `counts:` list is judged in the resolver's ORDER — strict
`validate_counts_breakpoints` first, `clamp_counts_breakpoints` only after —
because a single grafted leaf IS the `substitutive_levels[0]` that validator
measures against, while the clamp exists for the coarser levels of a pyramid
whose N the caller cannot know. Clamping first answered a confident wrong count
where the resolver raises: `{"recompute": True, "breakpoints": [100]}` on a
laddered 12-splat leaf clamped to `[12]`, counted one rung, and stranded the
wrapper. A rejected list is UNKNOWN instead, which the fallback below turns into
a clean refusal.

Energy-fraction breakpoints and any spec the resolver would reject answer
UNKNOWN, and an unknown count FALLS BACK TO THE STORE: `None` says the gate
cannot read the kwarg, not that the store's ladder went away. Skipping the leaf
instead re-stranded the very wrapper this gate closes — measured on a laddered
`kind=partition`, `{"recompute": True, "breakpoints": [0.3, 1.0]}`,
`{"recompute": True, "n_lods": 0}` and a junk `"stream"` all answered from inside
`part_0` and left `g` childless on disk. The fallback restores the pre-#1632
store-only verdict for those calls exactly, at the cost of over-refusing a
`{"recompute": True, "breakpoints": [1.0]}` that would in fact collapse to one
rung — which pre-#1632 also refused, and which is the conservative direction.
"A query must not pre-empt the builder's own fault report" still holds where it
belongs. An INVALID call — `additive_lod=True`
there, or a malformed spec, counts 1 rung, skips, and is reported by the builder
one level down with its own message. Both fail identically without
`partition=`, so the fault is the call's and the verdict is the builder's; the
outer transaction removes any partial graft. A
VALID energy-fraction spec — `partition={"max_elements": 4},
additive_lod={"breakpoints": [0.5, 1.0]}` on an unladdered store counts UNKNOWN,
falls back to a stored 1, skips — and then hits THIS conflict from inside
`part_0`. The outer graft transaction removes `g` and every descendant before
re-raising that builder verdict. The spec is well-formed and
the builder really does make 2 rungs from it; the count is unknowable only
because energy cuts need the ordering and the energy curve, the expensive half
this query exists to avoid. Refusing on UNKNOWN instead was rejected
deliberately: `{"breakpoints": [1.0]}` resolves to a single rung and partitions
fine, so a gate refusing every uncountable spec would refuse what the flat path
accepts — its own regression.

Two consequences worth stating plainly. The reason BODY of the kwarg-built
refusal is byte-identical to the data door's (they share
`partition_beside_a_ladder_reason`), and the whole message matches only on the
MATRIX-shaped branch, where the data door names the caller's node anyway — on the
GRAFT branch it says `'g'` where the data door would have said `part_0`, which is
the point of hoisting the question. Even the body matches only for a call whose
ONLY fault is this conflict: this gate runs above the data door's `lod_group=`
resolution, so
`add_gsplats_from_file(..., lod_group="bogus", partition={…}, additive_lod={"n_lods": 2})`
answers the conflict where `add_gsplats_from_data` with the same effective
arguments answers `TypeError: lod_group must be …`. Both refuse and write
nothing, so it is naming only — the same sanctioned divergence this package
documents for "a NaN position, an unknown attr". And a fault the COUNT cannot
see (a bad `method`, a stray `substitutive_level` key) is now masked by the
refusal rather than reported by the builder: the same trade, the conflict being
the more fundamental fault.

### `from_io.py` — load/fit then delegate

Both functions produce a `GSplatData` and hand it to
`add_gsplats_from_data_impl`:

- `add_gsplats_from_file_impl` — loads a `.gsplats.zarr` via
  `luxar.gsplats.io.load_gsplats` (raises `FileNotFoundError` if the path
  is missing). A matrix-shaped tree goes down the normal data path; a
  genuinely nested one (kind=partition root, or a lod with non-leaf children)
  is GRAFTED node-for-node by `graft_gsplat_node`, which routes attrs exactly
  like `lod_dispatch` does: [`COMPOSITING_ATTRS`](../compositing.py) — including
  `blending_mode` — land on the wrapper **only**, everything else rides onto
  each child. A `kind=partition` wrapper carries its stored `bsp_tree` unchanged;
  when legacy input has none, the graft reconstructs one from every child
  subtree's center bounds and defensively verifies exact separation before
  stamping it. Empty, overlapping, or interlocking parts keep the attribute
  absent and therefore retain the viewer's centroid-order fallback.
  `blending_mode` must NOT be duplicated onto the parts: it is
  nearest-setter-wins, so a part's copy shadows the wrapper and the layer's
  Blend control goes inert. Per-child `coverage_fraction` thresholds ride from
  each node's own `meta`; the **fallback** for a meta-less lod group is
  partition-binding-aware in three ways — the recursion's `_under_partition` flag,
  a `kind=partition` child of the ladder itself (the `overview` shape, the common
  reachable case), and `is_partition_bound(parent or group)` on the SCENE side.
  Note the scope: only a *non*-matrix-shaped subtree is grafted at all, so a
  per-part `add_gsplats_from_file` of an ordinary ladder store never reaches here
  — it is matrix-shaped and anchored by `lod_dispatch`. The scene-side term is a
  defensive fallback for a hand-built / nested lod-of-lods tree, which no library
  producer writes today. Before it grafts, it checks the STORED tree's column count
  against the scene (#1446), below the `dim_order` / `fill` / `fill_sigma` refusal
  the graft path already raises: the chain is built from the on-disk tree before the
  first leaf is added, so without that check a mismatched store refused from inside
  `part_0` / `child_0` and left a childless wrapper behind. One leaf answers for the
  whole subtree — a graft applies no `dim_order`, and both container node types
  reject mixed-`ndim` children at construction. Directly below that (it is the
  first statement of `graft_gsplat_node`, so both pre-graft checks outrank it),
  and for the same fail-before-the-wrapper reason, `_reject_labels_on_a_grafted_wrapper`
  refuses `labels=` / `image_labels=` (#1471) unless the grafted subtree is
  **exactly one leaf with exactly one additive sub-LOD** — the one shape that can
  actually carry them. Leaf COUNT, not node type: a one-part `kind=partition` of
  a flat leaf still has an exact per-element correspondence, so it keeps
  labelling normally **when the list is the right length**; a wrong length is
  now refused right there too, by the same validators the flat writer runs, so a
  mismatched `labels=`/`image_labels=` no longer refuses one level down from
  inside `part_0` with the wrapper already on disk (#1505). But a one-part
  partition of a LADDERED leaf is refused too, for a different reason and with
  a different message
  (`labels_on_a_laddered_leaf_reason`): `write_gsplat_leaf_subtree` has no labels
  channel at all, so exempting it would push the refusal down into `part_0` with
  the wrapper already on disk — and `--recipe tiles` carries a stream ladder by
  DEFAULT, so that is an ordinary call, not a corner case. Multi-leaf
  cannot be sliced here at all — a stored `GSplatPartition` carries no per-part
  index arrays, unlike `add_gsplats(partition=…)`, which slices the BSP `parts`
  it just computed — so the only definable mapping would be implicit
  leaf-concatenation order, and `gsplat flatten` (which emits exactly that order)
  is the remedy the message names. Both doors of the gate share one message
  template, `from_data.labels_on_wrapper_reason(kwarg, structure, remedy)`, with
  the two `*_STRUCTURE` / `*_REMEDY` constants beside it, so the wording cannot
  drift apart. `compositing.strip_absent_attr_kwargs` runs first on both and
  DELETES a present-but-`None` key: the leaf adders bind both as named params
  defaulting to `None`, but inside `**attrs` a `None`-valued key is still an
  unknown attr to `validate_render_attrs` (which matches by name, never by
  value), so the idiomatic `labels=maybe_labels` used to strand a wrapper.
  Since #1496 the graft entry also runs the same two `**attrs` calls the
  `from_data` entry does — see *An explicit `None` means absent* above — so
  `graft_gsplat_node(..., colors=None)` behaves identically to the other doors
  and a non-`None` `colors=` is refused before any wrapper exists.
- `add_gsplats_from_volume_impl` — fits in one step. With
  `progressive=True` it calls `fit_progressive_gaussian_splats`
  (honoring `max_splats_per_pass`, `psnr_patience`, `max_passes`);
  otherwise `fit_gaussian_splats`. `opacity`, `absorption`, and
  `blending_mode`, when set, are forwarded as scene attrs.

### `lod_dispatch.py` — the two multi-level write paths

**`add_gsplats_as_lod_group_impl`** builds a `kind=lod` `Group` with one
gsplats child per substitutive level, written coarsest→finest and named
`child_<i>`. Substitutive index convention is index 0 = finest,
`n-1` = coarsest, so the level loop iterates in reverse. Per-level splat
counts (summed across each level's additive ladder) feed both logging and the
per-child `coverage_fraction` thresholds. Those thresholds and the group-level
`selector` naming their units come as ONE decision from
[`lod/group.resolve_lod_ladder`](../lod/group.py): an explicit
`coverage_fractions=[...]` list is used verbatim and stamped
`LEGACY_LOD_SELECTOR`, otherwise the ladder is auto-derived through
`derive_coverage_fractions` and stamped `DERIVED_LOD_SELECTOR`. That derivation
chokepoint picks the anchor from the insertion point: `coverage_fractions`
(screen-occupancy halving, finest `WHOLE_OBJECT_FINEST_ANCHOR` = `0.5`) normally,
or `partitioned_coverage_fractions` (the same ladder ×2 in area units, finest
`PARTITION_FINEST_AREA` = `1.0`) when `is_partition_bound(parent or group)` —
i.e. the ladder is going inside a hand-built `kind=partition` wrapper, where it
switches on one tile rather than the whole object.

Attribute routing splits on
[`COMPOSITING_ATTRS`](../compositing.py): compositing attrs (opacity,
absorption, gamma, intensity, offset, blending_mode, transform, layer,
visible, nd_transform) land on the wrapper `Group`; everything else (including
`colormap`, deliberately) rides into each child so the user's intent is
not shadowed by the writer's per-leaf `colormap="gray"` default. Each
child is added through `lod_group_node.add_gsplats_from_data(...)` with
both LOD axes explicitly `None`, so the resolvers no-op and the
multi-substitutive branch is never re-entered.

**`add_gsplats_multi_lod_impl`** writes a multi-additive-LOD gsplats leaf.
It builds one `(centers, amplitudes, cholesky_factors, colors)` tuple per
additive sublod — applying `dim_order` per LOD through
[`apply_dim_order_positions` / `apply_dim_order_cholesky`](../dim_order.py)
and validating dimensions against the scene — then calls the shared walker
`write_gsplat_leaf_subtree` (which routes through
`io/_compiler/gsplat_tree.write_gsplat_node`). Specifying both `colors` and
`colormap` raises `ValueError`; a missing colormap defaults to `"gray"` only
when the data has no colors.

## Usage

These are implementation functions; reach them through the public Group
API:

```python
from luxar import LuxarZarrCompiler, Dimensions

with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())

    # Fit a volume and add in one step
    scene.add_gsplats_from_volume("fitted", volume, seeds=8000, n_iters=1000)

    # Load a pre-fitted .gsplats.zarr (v3.4 node tree grafted into the scene)
    scene.add_gsplats_from_file("loaded", "path/to/fitted.gsplats.zarr")
```

## Dependencies

**Internal:**
- `luxar.gsplats` — `GSplatData`, `fit_gaussian_splats`,
  `fit_progressive_gaussian_splats`, `io.load_gsplats`
- `core/group/lod/gsplats.py` — substitutive/additive axis resolvers
- `core/group/lod/group.py` — `derive_coverage_fractions` (and the
  `coverage_fractions` / `partitioned_coverage_fractions` /
  `is_partition_bound` it dispatches between)
- `core/group/compositing.py` — `COMPOSITING_ATTRS`
- `core/group/dim_order.py` — per-LOD dim_order application
- `core/gsplats.py` — the `GSplats` node returned for multi-additive writes

**External:**
- `numpy` — array handling
- `arbol` — structured logging (`aprint`)

## See Also

- [core/README.md](../../README.md) — scene graph and `add_gsplats*` overview
- [core/group/group.py](../group.py) — public method signatures that delegate here
- [gsplats/README.md](../../../gsplats/README.md) — fitting and `GSplatData`
- `docs/specs/GSPLATS_ZARR_FORMAT.md` — the v3.4 node-tree format (leaf / kind=lod / kind=partition)
