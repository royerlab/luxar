#### Two tests that could only fail on macOS

Both were invisible to CI — Linux passes them — so they only ever cost a
developer running the suite locally.

`test_a_dangling_attr_naming_a_metadata_document_is_skipped` now names
`.ZMetadata`, which is absent from a subgroup in both zarr formats. That keeps
the missing-payload branch covered on case-insensitive filesystems. Its sibling
`test_a_payload_named_like_a_metadata_document_is_refused` still uses
`Zarr.json` and skips there because the fixture needs real bytes under a name
that resolves to the group's own `zarr.json`.

`test_small_system_solver_matches_numpy_and_rejects_rank_deficiency` was not a
platform failure at all — it was flaky everywhere and macOS happened to draw the
losing seed. `_solve_system` takes two routes: above 3x3 it delegates to
`np.linalg.solve`, but the 3x3 case uses a closed-form adjugate inverse to avoid
a LAPACK call on a tiny matrix, and an adjugate loses accuracy faster than LU.
Measured over 400 random draws per rung, its relative disagreement with LAPACK is
4e-16 at kappa=1 and 7.4e-12 at 1e4, but median 3.3e-7 and p99 6.5e-6 at kappa=1e8
— against a flat `rtol=1e-6`. So a large share of seeds fail that rung; Linux and
macOS differ only because their LAPACKs build a different `qr` basis from the same
seed.

The tolerance now follows the conditioning (`kappa**1.5 * eps`, floored at 1e-9),
which is the adjugate's actual error growth. Over 2000 seeds per rung that is 0
failures with 17x headroom at kappa=1e8 — while making the well-conditioned rungs
substantially tighter than the constant it replaces. The delegating arm is now
held to exact equality, which also records that it compares `np.linalg.solve`
against itself and therefore pins the delegation rather than any arithmetic.
