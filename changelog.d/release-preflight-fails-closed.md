#### The release preflight no longer passes when it cannot check

`scripts/release.sh` §5 read branch protection with `|| true` and both CI
payloads with `|| echo '[]'`. The realistic trigger is not an outage: the
`protection/required_status_checks` endpoint needs admin, so a token without it
403s as a matter of course. That left the required-context list empty, which
dropped the script into a fallback that only inspected already-completed
check-runs — and if the check-runs read had failed too, the fallback examined an
empty list and reported success. Measured against a commit that does not exist
on the repository at all, the old script printed "all required checks green" and
went on to the publish plan. It now dies on any unreadable read, and dies when
branch protection lists no required contexts, because a green answer there would
carry no information. The check-runs read also gained `--paginate`: the default
page holds 30 and this repo exceeds it, so a required check could fall off the
end (`npm_variable_at` in the same file already paginated). A queued required
check now reports PENDING and fails the gate instead of being filtered out as
"not completed". When several check-runs carry the same required name, every
one of them must be successful before that requirement is green.

`release.sh` also never mentioned the changelog, so it would tag with every
fragment still unfolded — 588 of them at the time of writing — shipping a
release whose `CHANGELOG.md` does not describe it. A new step checks both halves
separately, since they fail independently: `changelog.d/` must hold no
fragments, and `CHANGELOG.md` must name the version. Each names the make target
that fixes it.

`check_version_consistency.py` gained `--expect-tag`, and the preflight passes
the tag through it. The tag is what triggers publishing, so it is the one
representation that has to agree with the other three — and nothing compared
`CITATION.cff` to it before.

`set_version.py` wrote `__init__.py` before parsing `package.json`, so a
malformed `package.json` left the tree half-stamped: Python bumped, citation
not, and a raw traceback to work it out from. It now parses and validates all
three files before writing any of them. Its note on formats claimed the
zero-padded form was the Python/PyPI one; PEP 440 normalises `2026.10.01` to
`2026.10.1` exactly as npm does, so only the files and the git tag keep the
padding. Its "next steps" now name `make changelog` and `make changelog-release`.
