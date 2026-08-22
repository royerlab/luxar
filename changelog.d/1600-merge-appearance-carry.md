#### `gsplat merge` keeps the appearance its inputs agree on

Every gsplat command that reads a dataset and writes one back carries the source
root's authored appearance across, so a structure-only rewrite does not silently
reset what was tuned in the Layers panel. `gsplat merge` was the exception: it
wrote no `root_attrs` at all, so all eleven `AUTHORED_APPEARANCE_ATTRS` keys were
lost on every one of its three modes — the merged root came back with
`colormap="gray"`, the multiplicative attrs at their identity, and
`blending_mode` / `visible` / `nd_transform` simply absent. It had been left
unaudited because "carry the appearance" is ambiguous with N inputs.

The rule is now unanimity, matching `agreed_normalization_stats`: a key is
carried onto the merged root only when every input that *has* an opinion on it
agrees, and an input with no opinion casts no vote — a dataset nobody ever tuned
must not veto a sibling's authored value. On a disagreement the key is dropped
and the command says so, naming the key, the differing values (each with the
input it came from, so a dissenter among N is identifiable) and what lands on
disk instead. That last part is a deliberate divergence from the normalization
rule, which drops silently: normalization stats are machine-recorded, while
appearance is hand-authored, and someone who tuned two datasets and merged them
should not have to discover the loss by looking at the render.

Having no opinion is broader than not carrying the key, which is what makes the
rule usable rather than pedantic. The writer STAMPS the identity values
(`opacity=1.0`, `absorption=1.0`, `gamma=1.0`, `intensity=1.0`, `offset=0.0`,
`layer=true`, and `colormap="gray"` on a colorless store), so a value equal to
the writer's own manufactured default counts as silence exactly like an absent
key — single-sourced as `WRITER_STAMPED_APPEARANCE_DEFAULTS`, read by both the
stamp sites and the vote so the two cannot drift. Without that, the commonest
merge of all — a tuned dataset plus a freshly fitted one — disagreed on *seven*
keys, dropped all seven, and the writer stamped its defaults back, which is the
untouched input's value: the same result as no carry at all, plus seven warning
lines. The cost is stated openly in the docstring: a deliberately authored
identity is indistinguishable on disk from an untouched store and loses to a
sibling's value.

`visible` is the one key where ABSENCE is itself a vote. The viewer reads a
missing `visible` as visible, so an input without the key is positively saying
"shown", and `visible=false` rides along only when every input hides — otherwise
one hidden input opened the whole merged dataset hidden.

`colormap` is dropped even under perfect agreement whenever the merged output
carries per-splat RGB that an input did not: `--channel-colors` bakes one, and
`GSplatData.concatenate` white-fills a colorless input to match a colored
sibling, so a plain merge and `--as-dimension` manufacture colors too. The
viewer makes an ancestor palette override per-splat RGB unconditionally, so a
carried palette would render the colored input through a scalar ramp; the
exclusion is therefore computed from the merged result rather than from the
flag. It is also refused when any input declares the `"custom"` sentinel —
demoting such an input to "no opinion" let the merged root adopt a *sibling's*
palette and repaint the custom-LUT splats with it — and that warning is now
emitted once for N inputs instead of once per input.

`--as-dimension` does NOT invalidate `nd_transform`, contrary to the first cut
of this change. `combine_as_new_dimension` appends the new axis LAST and an
`nd_transform` is keyed by dimension name, so no existing entry's name or index
moves and the new axis simply has no entry — the correct identity default,
verified against both a scene's real dimensions and the positional names the
viewer synthesizes for a detached root.

The audit table that missed this now has a completeness guard: every registered
`gsplat` command must appear either as a rewriter row or in an explicitly
reasoned exemption list, so a newly added command has to opt in rather than
inheriting a silent pass.
