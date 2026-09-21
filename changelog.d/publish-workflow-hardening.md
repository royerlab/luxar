#### The publish workflows no longer ship an uninstallable sdist, and npm stages

`hatch build` produced a wheel *and* an sdist, and `path: dist/` uploaded both.
The sdist cannot be installed: the wheel build hook raises when
`packages/luxar-viewer/dist/index.html` is absent, and the sdist does not carry
it, so `python -m build --wheel` on the unpacked sdist fails outright. The note
in `pyproject.toml` claimed a source install merely produced a "degraded
package" missing the viewer — optimistic in the wrong direction. The build is
now `hatch build -t wheel`, with a following step that fails if any tarball is
staged for upload. The sdist target stays configured on purpose: hatchling
builds an sdist by default, so deleting that section would produce a larger
one, not none.

Both publish workflows pinned `dtolnay/rust-toolchain` by SHA but passed no
`toolchain:`, leaving the channel to implicit resolution while `ci.yml` pins
1.92.0 explicitly and says in a comment why it must. Both now pin it too.

`publish.yml` compared the tag against `__version__` in hand-rolled shell —
one of the four version representations. It now calls
`check_version_consistency.py --expect-tag`, so `package.json` and
`CITATION.cff` are covered as well; nothing validated the citation against the
tag before.

`check_wheel.py` enforced PyPI's 100 MB limit per member, but PyPI applies it
to the uploaded file. The realistic way this project breaches it is thousands
of small test payloads and demo assets with nothing individually large, which
the per-member check cannot see. The wheel's own size is now checked and
reported on every run.

npm changed its default on 2026-09-03: trusted publishers created after that
date may stage, while direct `npm publish` is opt-in. Luxar has never published
to npm, so its publisher will land under the new default and `npm publish`
would have failed on permissions. `publish-npm.yml` now runs `npm stage
publish` and prints the approval commands; a maintainer promotes the staged
version with 2FA. The npm CLI is pinned (11.15.0, the staged-publishing floor)
instead of floating on `@latest`, and the job fails early if that floor is not
met rather than at publish time. `--provenance`/`--access` are dropped from the
command line, since `publishConfig` already sets both.
