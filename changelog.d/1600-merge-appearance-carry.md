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
agrees, and an input that does not carry the key casts no vote — a dataset
nobody ever tuned must not veto a sibling's authored value. On a disagreement
the key is dropped and the command says so, naming the key and the differing
values. That last part is a deliberate divergence from the normalization rule,
which drops silently: normalization stats are machine-recorded, while appearance
is hand-authored, and someone who tuned two datasets and merged them should not
have to discover the loss by looking at the render.

Two keys are dropped because the merge *mode* invalidates them rather than
because the inputs differ, so they warn even under perfect agreement:
`nd_transform` under `--as-dimension`, which adds a dimension the inputs'
per-dimension affines do not describe, and `colormap` under `--channel-colors`,
which bakes per-splat RGB after which no palette describes what is rendered.

The audit table that missed this now has a completeness guard: every registered
`gsplat` command must appear either as a rewriter row or in an explicitly
reasoned exemption list, so a newly added command has to opt in rather than
inheriting a silent pass.
