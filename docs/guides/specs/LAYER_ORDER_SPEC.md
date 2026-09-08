# Layer Order — Authored Cross-Layer Draw Order

> **Status**: IMPLEMENTED on branch `feat/layer-order`. This document specifies an authored
> per-layer draw order (`layer_order`), the 2D-layer-stack idea — CSS `z-index`
> / Illustrator layer depth — applied to Luxar's overlapping 3D layers. It is
> the cheap, authorially honest alternative to per-element cross-node
> interleaving or one globally merged draw. It adds no new per-element
> machinery: one integer per layer feeds the `renderOrder` assignment that
> already exists.

---

## 1. What exists today — and why overlapping layers already look stable

`render-order.ts` puts every visible **sorted-mode** mesh on one global integer
scale, `renderOrder` 1..M, farthest first, by three rules (full prose at
`assignGlobalRenderOrder`):

1. Slots group by partition wrapper; a single leaf is a group of one.
2. Groups order by the **mean view-z** of their members' content centroids.
3. **Containment overrides depth**: when one group's bounding sphere strictly
   contains another's, the container is forced to draw FIRST (#843), because
   no single per-mesh integer is correct for an embedded node — the container's
   centroid sorts nearer for ~half of all camera orientations, and an
   order-dependent mode drawn container-last multiplies the embedded node's
   pixels by the container's whole transmittance, ≈ erasing it.

Rule 3 is why the two multichannel bioimaging demos do **not** pop today.
Measured 2026-09-01 on the hosted build, `volumetric` forced on every layer,
8 camera angles spanning 300°:

| demo | fixed order | radius | contains the next? |
| --- | --- | --- | --- |
| acto3d heart | `vasculature` @1 | 1130.74 | yes — margin **2.2** |
| | `tnni3` @2 | 1126.79 | yes — margin 49.7 |
| | `nuclei` @3 | 1064.67 | — |
| neuromast | `nuclei` @1 | 555.54 | yes — margin **9.1** |
| | `membranes` @2 | 546.39 | — |

`DISTINCT_ORDERINGS = 1` for both, and the order is exactly by **decreasing
radius** — the signature of a containment DAG, whose edges always point large →
small. These layers are fits of the same specimen, so they are near-concentric
and the largest sphere swallows the rest.

**Two problems, both visible in that table.**

- **The stability is accidental.** The heart's `vasculature ⊃ tnni3` edge has
  2.2 units of margin on a 1130-unit radius — **0.19%**. A refit that nudges
  either channel's extent breaks the edge and the popping returns, with no
  authoring change and no signal. (The `CONTAINMENT_EPS = 1e-3` slack does not
  rescue this: the edge clears the strict test on its own, by 0.19%.)
- **The inferred order is not necessarily the intended one.** Container-first is
  damage limitation — "under-attenuating a marker is the lesser error vs.
  blinking it out entirely on camera orbit". Nothing lets an author say
  *vasculature should read through the cardiac tissue*.

And a third gap, one rule 3 never addressed: **commutative-mode layers are
pinned at `renderOrder` 0** and therefore draw *before* the farthest sorted
mesh, always. `additive` additionally carries `depthTest: false` and paints
through everything. So `additive`-vs-`volumetric` layering is not merely
approximate, it is not expressible at all.

---

## 2. The mechanism: depth bands

One authored integer per layer, `layer_order`. Higher = nearer the camera =
drawn later = composites on top (§3 D1).

`assignGlobalRenderOrder` gains the level as its **primary key**:

```
groups := today's grouping (partition wrapper | single leaf)
bands  := groups partitioned by effective layer_order
for each band, ascending by level:          # lower level = farther = drawn first
    order the band exactly as today         # mean view-z, then containment hoist
    emit renderOrder = nextRank++ per member as today
```

Two properties fall out, and both are load-bearing:

- **Unset ≡ level 0.** With nothing authored anywhere, every group lands in a
  single band, the band loop degenerates to one iteration, and the algorithm is
  today's algorithm *verbatim* — not a special case, not a guarded fast path.
  §3 D2's "unset is byte-identical to today" is therefore structural, which is
  what makes the change safe to land dark and what makes the golden test in §10
  meaningful.
- **Bands are hard.** Two layers in different bands never interleave, whatever
  the camera does. That is the entire point: it converts today's accidental
  stability into a stated contract.

The output is still the same `renderOrder` 1..M integer sequence, so nothing
downstream changes — no scene-graph change, no new attribute, no per-element
work, no shader change.

**On three.js `groupOrder`, which outranks `renderOrder`.** Both painter
comparators (`three.module.js:8112` `painterSortStable`, `:8142`
`reversePainterSortStable`) compare `groupOrder` → `renderOrder` → `z` → `id`, and
`groupOrder` is derived from the *innermost* `Group` ancestor's `renderOrder`
during `projectObject`. In the standalone app that is 0 everywhere; in the
**embed** path `LuxarLayer` deliberately stamps its configured `renderOrder`
(default 10) onto **every owned `THREE.Group`** (`src/core/layer/luxar-layer.ts:737-741`),
a documented contract to host applications (`LUXAR_LAYER_SPEC.md`, "Draw order and
visibility"). Either way the value is *uniform across Luxar's own content*, so
per-mesh `renderOrder` remains the discriminator among Luxar layers and bands work
unchanged.

What follows is a **scope limitation** rather than a problem: `layer_order` orders
layers *within* the Luxar subtree only. Ordering Luxar content against a host
application's own transparent geometry is the `LuxarLayer` `renderOrder` option's
job — a different knob at a different scope. The two compose (host groups pick the
band block's position; levels order inside it) and must stay documented as
separate. This also means the uniformity above is an invariant worth keeping: if
Luxar ever gave two of its own Groups different `renderOrder`, `groupOrder` would
start splitting bands underneath us.

---

## 3. Pinned decisions

**D1 — Higher = nearer the camera.** `layer_order = 30` draws after (on top of)
`layer_order = 10`. This matches CSS `z-index` and Illustrator's bring-to-front,
i.e. every 2D layer tool an author has met, and is *inverted* from the
renderer's internal "farthest first, lowest `renderOrder` drawn first"
direction. The inversion lives in one comparator and is worth it; the
alternative optimises for the reader of `render-order.ts` over the author of a
scene.

**D2 — Unset ≡ 0, and unset MUST stay distinguishable from an authored 0.**
`layer_order` gets **no writer-stamped default** — it is absent from both
`WRITER_STAMPED_APPEARANCE_DEFAULTS` and `IDENTITY_COMPOSITING_ATTRS`, joining
`blending_mode` / `visible` / `join` / `nd_transform` whose "absence on disk is
genuine silence". `opacity` is the cautionary precedent — its stamped identity
"cannot be distinguished from a deliberate authored identity".

Be precise about *why*, because the tempting justification is wrong. A stamped
`0` would **not** break the ordering: every group would land in band 0, that is
one band, and containment would go on operating exactly as it does for an
unauthored scene (D3 — it is a band *difference* that overrides containment, not
explicitness). What a stamped default would destroy is the ability to tell the
two states APART, and three things depend on that:

- the bucket-straddle diagnostic is gated on the order being authored, so it
  would start firing on scenes that authored nothing;
- the Layers panel would show `0` in every field instead of a blank `auto`, so
  no one could see which layers actually state an order;
- any future rule that wants to treat "the author chose 0" differently from
  "nobody chose" would have no way to, and the distinction cannot be recovered
  after the fact — every store written in the meantime would already claim it.

**D3 — A DIFFERENCE in layer order wins over containment; containment operates
only within a band.** Because bands are hard partitions, a containment edge
between two different bands cannot be represented at all — the relation is
simply dropped. The viewer therefore **warns once per node pair** when a band
split breaks a containment relation, naming both paths, so "my embedded marker
vanished" is diagnosable rather than mysterious. Containment continues to
operate unchanged between groups in the *same* band (including two groups whose
orders are both unset, which is every scene today).

Note the precise wording, because the obvious paraphrase is wrong: it is the
*difference* that overrides containment, not the mere act of authoring. Two
layers that share a band are still ordered by containment, and **an authored
order equal to a neighbour's has no ordering effect** — including the easy-to-hit
case of authoring `0` on one layer while its neighbour states nothing, since
unset also resolves to band 0. Stating an order between two layers therefore
means giving them *distinct* values. `levelExplicit` is tracked per group but
deliberately does not enter the ordering: it gates only the bucket-straddle
diagnostic, because a warning about a half-honoured order should not fire for a
scene that authored no order at all.

**D4 — Applies to every blending mode, commutative included.** A level on an
`additive` layer is a no-op against other `additive` layers (addition
commutes) but is meaningful against `normal`/`volumetric` ones, which is the
§1 third gap. Safe by construction: a layer with no authored level keeps
`renderOrder` 0 and today's behaviour exactly.

> Assumption flagged for correction: this scope was inferred rather than chosen.
> Asked whether commutative modes should be in scope, the answer given was
> "**of course, the default, if no depth-level is provided should be the same as
> what we have today**" — a constraint (D2) that all three scope options
> satisfy. Full scope is taken because it is the only option that both honours
> D2 *and* helps the seven demos #1964 moved to `additive`: with an authored
> order they could return to `volumetric`, which is what they wanted in the
> first place (#880). If the intent was the narrower scope, D4 is the decision
> to revisit and §7's collect-loop change is the only code that changes.

**D5 — It is a per-LAYER property: a compositing attr, nearest-setter-wins.**
`layer_order` joins `COMPOSITING_ATTRS` — the set that "rides on a wrapper Group
(where the user thinks of the wrapper as *their layer*) rather than getting
copied onto each internal child" — and `AUTHORED_APPEARANCE_ATTRS`, so a
structure-only rebuild (`gsplat lod`, `gsplat additive`, …) carries it from
source root to output root instead of dropping it (#1600's class of bug). It
composes root→leaf **nearest-setter-wins**, like `blending_mode` / `join` /
`colormap`, not multiplicatively.

**D6 — A level authored strictly inside ANY specialized group is an ERROR.**
That is `kind=partition` *or* `kind=lod`, at any depth, which makes the rule
statable in one line: **`layer_order` may participate only when authored on a
node that is a layer** — a plain group or a top-level leaf — never on the
internals of a specialized group. The scene root may carry the attr, but like
every rendering attr there it is excluded from composition and ignored. For a
partition the reason is severe (§5: it
would split the wrapper across bands and destroy the exact Fuchs–Kedem–Naylor
part order); for a `kind=lod` group it is that a level is an *alternative*, only
one of which renders, so a level on one would be inert — and an attr that writes
cleanly and silently does nothing is this codebase's most expensive failure mode.
One ancestry check covers both, and covers nested cases (a lod group inside a
partition part) with no extra rule.

Refused at authoring through *both* doors (the adder kwarg and a post-hoc
`node.attrs["layer_order"] = …`), mirroring `reject_lines_only_join` +
`reject_lines_only_join_assignment`. The viewer, which must render whatever it
is handed, instead warns once and uses one level for the whole order group,
keeping the first collected member's — strict write, tolerant read.

**D7 — Layers-panel edits are session-only.** Matching every other control in
that panel: "Edits made in the panel are viewer-only and not persisted back to
the zarr store."

---

## 4. Why this is cheap

| | Route B (archived) | Route C (§8.2) | Layer order |
| --- | --- | --- | --- |
| Per-frame cost | +5.5–6.4 ms at 2M (measured) | one frame of sort lag | **none** |
| New per-element machinery | shard meshes, per-shard AABBs, k-way merge | global element storage, per-element uniform indirection | **none** |
| Subsystems gaining a permanent invariant | 9 | ~all of them | **0** |
| Ordering quality | 84–93% of the error removed | exact | *stability, not correctness* (§5) |
| Expressible authorial intent | none | none | **the whole point** |

The last two rows are the trade, stated plainly: this does not make
interpenetrating layers *correct*. It makes their order **stated, stable and
overridable**, which is what the two demos actually need and what neither
archived route offered.

---

## 5. Interaction with BSP part order

*(The question that shaped this design.)*

**They live at different levels of the hierarchy and do not compete.** A
`kind=partition` wrapper IS the layer (D5), so the level decides where the
*whole partition* sits relative to other layers, while the stored BSP tree
decides the internal order of that partition's parts — the exact
Fuchs–Kedem–Naylor order, valid for any camera pose including inside the volume
(the #565 guarantee). Parts inherit the wrapper's level by nearest-setter-wins,
so **a partition is never split across bands**, and the two mechanisms never
meet.

**That is exactly why a per-part level must be refused (D6, which extends the
same refusal to `kind=lod` internals) rather than ignored.** If parts could carry their own levels, the wrapper would split across
bands and its parts would interleave *by band* instead of *by the tree* —
silently destroying the one ordering guarantee in the system that is exact. An
author writing a per-part level has expressed something the renderer cannot
honour without discarding a stronger guarantee, so the honest response is a
refusal at write time, not a best-effort reinterpretation.

A second, quieter reason: `assignGlobalRenderOrder` chooses BSP-rank-vs-view-z
**once per group** (`everyMemberRanked`), and that all-or-nothing choice is
load-bearing — mixing the two keys makes the comparator non-transitive, and
`Array.sort` may then return an order violating both keys. Bands do not disturb
this, *because* groups stay whole within a band. A per-part level would reopen
it.

Two neighbouring structures, for completeness:

- **`kind=lod` wrappers** are unaffected: levels are alternatives and only one
  renders, so a level on the wrapper composes to whichever level is live.
- **Nested wrappers** (a partition of LOD groups) are covered by D6's refusal,
  which closes the whole class rather than requiring a rule about which of two
  nested levels wins.

### 5.1 The full hierarchy of orders

Seven nested mechanisms decide what is drawn when. The level slots in at #3 —
above every view-dependent rule, below the two that are not ours to move.

| # | Mechanism | Scope | Camera-dependent? | Authorable? |
| --- | --- | --- | --- | --- |
| 1 | three.js bucket (opaque → transmissive → transparent) | whole scene | no | **no** — §8.1 |
| 2 | `groupOrder` (an ancestor `Group`'s `renderOrder`) | subtree | no | unused (always 0) |
| 3 | **`layer_order` band** | layer | **no** | **yes — this spec** |
| 4 | Containment DAG hoist | within a band | no | indirectly (bounds) |
| 5 | Mean view-z of groups | within a band | **yes** | no |
| 6 | BSP part rank (exact) or member view-z | within a group | **yes** | no |
| 7 | `aSortedIndex` per-element permutation (exact) | within a mesh | **yes** | no |

Reading it top-down is the design in one line: **an authored level is the
outermost thing we control, and everything view-dependent below it keeps working
unchanged, carried along inside whatever band it lands in.**

### 5.2 What a level on a group means

Setting a level on a group node moves that group's whole subtree **relative to
other bands, as a rigid block — and the view-dynamic orders inside it travel
with it intact.** A partition wrapper keeps its exact per-frame BSP part order
(#6); every mesh keeps its per-element permutation (#7). The band changes only
*where the block sits*, never *how the block is internally ordered*. That is
the intended behaviour and the reason bands sit above rules 4–7 rather than
among them.

Three consequences that are easy to assume wrongly, and are not obvious from
the sentence above:

- **A band is not an atomic unit.** A level makes a group a block relative to
  *other bands*, not relative to *other layers sharing its band*. Two leaves in
  one group at level 20 and an unrelated layer also at level 20 all order among
  themselves by rules 4–6, so the unrelated layer can interleave *between* the
  group's leaves. To make a group genuinely indivisible, give it a band of its
  own. (A plain `group` node is not an order group — `render-order.ts` groups by
  *partition wrapper or single leaf*, so a plain group's leaves are independent
  groups that merely happen to share a level.)
- **A leaf can escape its group's band.** Nearest-setter-wins (D5) means a level
  authored on a leaf beats the one on its group ancestor, so that leaf leaves
  its siblings behind. This is deliberate and consistent with every other
  compositing attr, and it is *safe* here in a way D6's partition case is not:
  a plain group carries no exactness guarantee to destroy. But it is the
  opposite of "the group moves as a unit", so it is a decision (§12.5), not an
  accident.
- **An authored order cannot override the opaque/transparent bucket split.**
  If an opaque group has an equal or higher order than a transparent group,
  THREE still draws it first. The renderer warns once for the involved order
  groups, including the case where one band contains both bucket types.

Two smaller cases, checked and benign:

- **Negative levels vs. untracked meshes.** `renderOrder` 0 stays reserved for
  meshes the coordinator does not track; collected slots always receive 1..M
  regardless of band, so a negative band cannot alias 0 and untracked geometry
  still draws first. Unchanged from today.
- **A level on the scene root** is ignored, like every other rendering attr
  there. The scene root is a carrier and is excluded from the composition chain.

And one free benefit worth naming: because a band is camera-independent, it adds
**no re-sort trigger and no per-frame work** — and it makes the LOD case stable
that is unstable today, where switching level changes a node's bounds and can
therefore flip a containment edge mid-orbit.

**The honest limit.** A level is the right tool when the author knows the
answer, and the wrong tool when there is not one to know. For two *concave
interpenetrating* layers no valid whole-object order exists from every
viewpoint — a level pins such a pair to one stated wrong answer instead of a
camera-dependent wrong answer. Stable and diagnosable beats flickering, which
is why this is worth shipping; it should never be documented as a geometric
correctness fix.

---

## 6. Authoring surface (Python)

```python
scene.add_gsplats("vasculature", ..., blending_mode="volumetric", layer_order=10)
scene.add_gsplats("tissue",      ..., blending_mode="volumetric", layer_order=20)
scene.add_gsplats("nuclei",      ..., blending_mode="volumetric", layer_order=30)
```

- A `layer_order` keyword accepted by `add_points` / `add_lines` /
  `add_gsplats` / `add_mesh` / `add_group`, plus post-hoc assignment through
  `node.attrs["layer_order"]`.
- Validation (`validation/types.py::validate_layer_order`): a finite Python
  `int` (a `bool` is refused — `isinstance(True, int)` is the classic hole);
  any sign. Bounded to the JS safe-integer range (`JS_SAFE_INTEGER_MAX`,
  2^53 - 1) — not an arbitrary clamp but the representable domain: the attr
  crosses to the viewer as a JS `number`, and past that magnitude two orders
  the author separated can collapse into one band, handing the choice back to
  the inference this attribute exists to override.
- Registered in `COMPOSITING_ATTRS` and `AUTHORED_APPEARANCE_ATTRS`; **absent**
  from `WRITER_STAMPED_APPEARANCE_DEFAULTS` and `IDENTITY_COMPOSITING_ATTRS`
  (D2). A present-but-`None` value is rejected rather than treated as absent.
- D6's refusal: `reject_layer_order_inside_specialized_group` in
  `core/group/compositing.py` plus its assignment-door twin in
  `core/node/node.py::_WriteThroughAttrs`.

---

## 7. Viewer surface

| File | Change |
| --- | --- |
| `data/attrs-composer.ts` | `ComposableAttrs.layer_order?: number`; `EffectiveAttrs.layer_order?: number`; nearest-setter-wins, beside `blending_mode`. Header prose lists the new rule. |
| `rendering/node-factory/*` | Stamp the composed level onto `mesh.userData.layerOrder` at node creation, alongside the other composed appearance values. |
| `rendering/depth-sort-coordinator/render-order.ts` | `OrderSlot` / `OrderGroup` gain `level` + `levelExplicit`; `assignGlobalRenderOrder` sorts once by `(level, meanZ)`, and `orderGroupsWithContainment` restricts edges to groups in the same band so the existing Kahn pass drains bands in order while reporting harmful dropped containment edges (D3's warning). |
| `rendering/depth-sort-coordinator.ts` | The collect loop (`~:1815`) now collects a commutative node **when it carries an explicit level**; other non-order-dependent nodes still reset to `renderOrder = 0` and skip collection. This is the only code D4 touches. |
| `ui/layers/layer-controls.ts`, `layer-state.ts`, `layer-apply.ts` | A **Layer order** control per layer row (a small stepper, blank = unset). Session-only (D7). |
| `ui/data-loading-monitor/templates/scene-graph.ts` | The live draw-order chip already prints `#renderOrder` + bucket; add the band so an author can see *why* a layer sits where it does. |

Deliberately unchanged: the sort worker, the WASM/TS kernels, `element-storage`,
picking, LOD, and the commit pipeline. This feature never touches per-element
data.

---

## 8. What this does NOT fix

1. **The opaque/transparent bucket split.** three.js renders opaque, then
   transmissive, then transparent, and `renderOrder` only sorts *within* a
   bucket. `opaque` is the one blending mode with `transparent: false`
   (`blending-state.ts:292`), so **no `layer_order` can place an `opaque` mesh
   in front of a transparent layer.** A level spanning the two buckets is
   silently partially honoured, which is why the renderer warns once per layer
   when an authored order conflicts with that bucket order
   (`warnBucketOrderConflict`, §12.2) rather than leaving it to be discovered.
   Opaque layers with authored orders also receive positive `renderOrder`
   values, which can reduce front-to-back early-Z efficiency relative to
   untracked opaque meshes; this is a performance tradeoff, not a visual one.
2. **Interpenetrating concave layers** — §5's honest limit.
3. **Per-pixel ordering.** Out of reach of any per-element scheme
   (StopThePop's class of artifact); unchanged deferral.
4. **Within-node ordering**, which is already exact and untouched.

---

## 9. Phases

- **Phase 1 — viewer-only, dark.** The composer field, the band partition, the
  containment-drop warning, the collect-loop change. Nothing authors it yet, so
  §10's golden test is the whole gate: every scene must produce byte-identical
  `renderOrder` integers.
- **Phase 2 — authoring.** The Python kwarg, validation, the two refusals, the
  compositing-set registrations, format-spec documentation.
- **Phase 3 — UI.** The Layers-panel control and the monitor band readout.
- **Phase 4 — the demos.** Return the #1964 demos to `volumetric` with authored
  levels, and A/B them against their additive versions. This phase is what
  decides whether the feature earned its place, and it is a judgement on a
  render, not on a metric.

Phases 1–3 are independently landable; Phase 4 is the one that can say no.

---

## 10. Testing and ship gates

| Gate | Shape |
| --- | --- |
| **Unset is byte-identical** | Golden: for every existing test scene and the E2E fixtures, the assigned `renderOrder` integers with no level authored anywhere must equal today's exactly. This is the invariant that makes the change safe; it should fail loudly if the band loop is ever not a no-op on one band. |
| Band ordering | `fast-check` property over random (level, view-z, radius) sets: never a member of a lower band after a member of a higher band; within a band, today's order reproduced. |
| BSP exactness preserved | The #843 / #565 containment and BSP fixtures must pass unchanged with levels unset, AND with a level authored on the wrapper (which must not perturb internal part order at all). |
| D6 refusals | Both doors, both directions: the adder kwarg on a partition part, and `node.attrs["layer_order"] = …` post-hoc. Plus the viewer's warn-and-ignore on a hand-built store carrying an inner level. |
| D3 warning fires | A fixture where a containment relation is broken by a band split must emit exactly one warning naming both paths — a "fires-proof" test, not just a no-crash one. |
| Commutative scope (D4) | An `additive` layer with an explicit level must receive a positive `renderOrder`; without one it must stay at 0. |
| E2E, non-vacuous | A two-layer fixture where compositing order alone decides the dominant channel at the projected overlap: author level A>B, assert the pixel; swap to B>A, assert it inverted. **Fail-first verified** by pinning both levels equal. |
| Docs gate | New file must be listed in a `docs/index.rst` toctree or `make check-docs` goes red. |

---

## 11. Risks

1. **D2 is the whole safety argument.** If anything ever stamps a default
   `layer_order`, rule 3 silently switches off for every store written after
   that point. The `IDENTITY_COMPOSITING_ATTRS` registration is the thing to
   guard with a test that asserts *absence*, not presence.
2. **D4 widens the collected set.** Commutative nodes have never received a
   positive `renderOrder`; an `additive` layer moving out of the "draws first"
   position can change appearance. Phase 4 converted the neuromast's two layers
   to depth-sorted `volumetric` blending, so it no longer exercises D4's
   commutative path. The risk remains for future scenes that mix a levelled
   commutative layer with untracked `renderOrder = 0` content. Phase 1 is
   behaviour-preserving only for unauthored scenes.
3. **A level is a promise the renderer cannot always keep** (§8.1's bucket
   split, §5's interpenetration limit). Documenting where it is partially
   honoured matters more than the mechanism, because a silently-half-applied
   ordering is worse than none.
4. **The Layers-panel control invites reordering by dragging**, which this
   design does not provide (a stepper, not a drag handle). If drag-to-reorder is
   wanted, it needs a defined mapping from list position to integer levels —
   §12.

---

## 12. Open questions

1. **Drag-to-reorder in the Layers panel?** A stepper is the minimum. Dragging
   rows is the familiar 2D-tool gesture, but needs a rule for turning list
   position into levels (renumber all layers? sparse 10/20/30 with insertion
   between?) and interacts with D7's session-only edits.
2. ~~**Should an authored order conflicting with the opaque/transparent bucket
   boundary warn?**~~ **DECIDED — yes, and implemented.**
   `warnBucketOrderConflict` compares authored opaque and transparent groups
   and warns once for the involved groups when the requested ordering cannot be
   honoured. See §8.1 for why the limit exists at all.
3. ~~**Should the demo authoring audit report authored orders?**~~
   **DECIDED — the audit rule was restated rather than extended.**
   `test_overlapping_gsplat_layers_declare_order.py` replaces #1964's
   all-additive rule with: an overlapping-layer demo must be *either* entirely
   commutative *or* depth-sorted with an explicit `layer_order` on every layer.
   Relaxed in one direction (the old rule would now forbid the correct thing),
   tightened in another (a depth-sorted layer must now carry an order, which
   the old rule had no way to require). `luxar info` is untouched and remains a
   genuinely open, separate question.
4. **URL override** (`?layerOrders=path:level,…`) for A/B measurement without
   re-authoring. Not specified above.
5. **Should a leaf be allowed to escape its group's band?** (§5.2.)
   Nearest-setter-wins says yes and every other compositing attr agrees, but
   "set the group's level" then does not guarantee the group stays together. The
   alternative — a group's level *pins* its whole subtree, refusing or ignoring
   inner levels — would make groups atomic but breaks the composition rule
   uniformity that makes these attrs predictable.
6. **Is an atomic-block mode wanted at all?** If "this group must never be
   interleaved" is a real authoring need (§5.2, first bullet), it is a
   *different* feature from a level — closer to "render this subtree to its own
   pass" — and should not be smuggled into this one.

---

## 13. References

- `rendering/depth-sort-coordinator/render-order.ts` — the three rules, the
  containment test, the per-group all-or-nothing rank choice.
- `core/group/compositing.py` — `COMPOSITING_ATTRS`,
  `AUTHORED_APPEARANCE_ATTRS`, `WRITER_STAMPED_APPEARANCE_DEFAULTS`,
  `IDENTITY_COMPOSITING_ATTRS`.
- `data/attrs-composer.ts` — the composition rules.
- `ui/layers/README.md` — the layer row model and session-only edits.
- PR #843 (containment), #1235 (tightened bound), #880 (demos → volumetric),
  #1964 (overlapping layers → additive), #1600 (appearance carried through
  rebuilds).
