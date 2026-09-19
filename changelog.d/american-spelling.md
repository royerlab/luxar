#### Public names settle on American spelling before the first release

A spelling becomes an API the moment the package is on PyPI, and Luxar's public
surface carried both conventions: `luxar optimise` and `luxar.io.optimise`
(with `optimise_store`, `OptimisePlan`, `plan_optimisation`, `summarise_plan`,
`summarise_chunk_layout`) beside `--center`, `center_bounds` and every other
American-spelled name. This is a hard cut with no aliases — pre-release there
is nobody to keep the old spellings for. The command is now `luxar optimize`,
the module `luxar.io.optimize` (`optimize_store`, `OptimizePlan`,
`plan_optimization`, `summarize_plan`, `summarize_chunk_layout`, and
`luxar.cli.optimize_command` / `register_optimize_command`), and the reveal
ladder's knob is `--reveal-center` on `gsplat lod` and `mesh lod`, with the
matching `additive_lod={"reveal_center": …}` spec key and helpers
(`resolve_reveal_center`, `preflight_reveal_center`,
`wants_reveal_center_preflight`, `parse_reveal_center`). The CLI help strings
that named a "centre" or an "optimiser" were reworded to match.

A guard test (`cli/tests/test_american_spelling_guard.py`) walks the live Typer
app the way Click dispatches it and every `__all__` in the package, and fails on
an `optimis`/`centre` fragment in any command name, option spelling, help string
or exported name, so the British forms cannot creep back. It also pins the hard
cut: `luxar.io.optimise` no longer imports.
