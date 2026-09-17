#### A compatibility and deprecation policy, and the helpers to honour it

`docs/guides/user/COMPATIBILITY_POLICY.md` states what counts as public (the
PyPI package's exported names and documented modules, CLI commands and flags,
the `@luxar/viewer` barrel plus `StorageKeys`, URL parameters and embedder
events, and the on-disk formats by `format_version`), how CalVer maps onto the
npm semver form, the single format-version rule readers apply, the deprecation
window — two releases or six months, whichever is longer — and the mechanism
for each surface, that the inputs to `content_hash` are frozen, how the
browser storage-format versions behave when bumped, and the `three` peer-pin
cadence. CONTRIBUTING gains a "Deprecations and renames" section and
`changelog.d` fragments announcing a compatibility event start with
`Deprecated:`, `Removed:` or `Breaking:`.

The helpers exist before they are needed: `luxar.utils.deprecation`
(`warn_deprecated`, `deprecated_kwarg_alias`) and `luxar.cli.utils.deprecated_option`
give the first post-release rename a tested pattern, while the pre-release
renames in this batch stay hard cuts. `luxar.mesh` re-exports its decimation
API, as the docs already claimed, and `luxar.validation` resolves to the
package rather than one submodule. The hosted-demo count in the README and the
docs index is now generated from the gallery manifest and gated, and two more
"current gsplats format version" statements in the docs are pinned to the
contract so they cannot drift again.
