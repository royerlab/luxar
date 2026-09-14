# Changelog fragments (`changelog.d/`)

Each pull request that deserves a changelog entry adds **one file here** instead of
editing `CHANGELOG.md` directly. That is the whole point: `CHANGELOG.md` used to be
edited by roughly a third of all PRs, so under a moving `main` every rebase
re-conflicted on it. A per-PR fragment file never conflicts — your PR only ever
touches its own new file — and `CHANGELOG.md` changes exactly once, at release.

## How to add an entry

Create `changelog.d/<PR-or-issue-number>.md` containing the entry **exactly as it
should read in `CHANGELOG.md`**: a `#### Title` line followed by one or more prose
paragraphs. This is the house style — a short, self-contained explanation of *what*
changed and *why*, not a terse bullet. For example, `changelog.d/1490.md`:

```markdown
#### Serialized merge train stops the rebase cancel-loop

Under a strict, moving `main`, a PR had to be simultaneously up-to-date *and*
CI-green to merge; rebasing to become up-to-date restarted CI, and a merge landing
meanwhile cancelled it. The merge daemon now brings exactly one PR up to date at a
time, so each rebased head gets an uninterrupted CI window.
```

Name the file after the PR number when you know it (e.g. `1490.md`); an issue number
or a short slug also works. Numeric names are folded in ascending order.

## Assembling a release

At release-prep, fold every fragment into `CHANGELOG.md` and delete the fragments:

```bash
python3 scripts/changelog_build.py --check  # validate every fragment; no git history needed
make changelog-draft          # preview what would be written; changes nothing
make changelog                # fold fragments under ## [Unreleased] / ### <Month> and remove them
make changelog MONTH="August 2026"   # pin the month heading explicitly
```

Commit that in the normal version-bump PR, then follow **Cutting a release** below
before tagging the release (`make release`).

## Notes

- Fragments are **Markdown** and touch no source domain, so a fragment-only PR runs
  no code test suite (only the fast docs gate). That gate runs the git-free
  fragment validator and reports every malformed entry in one pass — see the
  `changes` job in `.github/workflows/ci.yml`.
- `CHANGELOG.md` also carries a `merge=union` attribute (`.gitattributes`) so any
  direct edit during the transition still auto-resolves on rebase.
- Not every PR needs a fragment (pure refactors, test-only changes, trivial fixes
  often don't). Use judgement — the same bar as adding a `CHANGELOG.md` entry before.

## Which month an entry lands under

At release-prep, `make changelog` files each fragment under the month it was
**written** — the date of the commit that added the file — not the month the
fold happens to run in. Those are rarely the same: when this rule was
introduced, 440 pending fragments spanned two months and every one of them
would have been filed under the current one, mis-dating 408 entries into a
single unnavigable heading.

Two consequences worth knowing:

- **Commit your fragment.** An uncommitted file has no date git can read, and
  the fold refuses rather than guessing. Committing it is the fix.
- `make changelog MONTH="August 2026"` still pins every entry to one heading,
  which is the escape hatch for a fold that has to run outside a git checkout.

## Cutting a release

`make changelog-release` renames `## [Unreleased]` to `## [<version>] - <date>`
and opens a fresh empty `## [Unreleased]` above it, so the file accumulates
release history instead of one ever-growing section.

Order matters, and it is the reverse of what you might expect:

1. `make changelog` — fold the fragments.
2. `make set-version DATE=YYYY.MM.DD` — the version **is** the release date, so
   it is set last.
3. `make changelog-release` — cut the section, named from `__version__`.

The cut refuses while any fragment is still pending, because folding after the
cut would ship those entries under the *next* version.
