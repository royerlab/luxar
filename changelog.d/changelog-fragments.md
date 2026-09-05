#### Changelog fragments end CHANGELOG.md merge conflicts

`CHANGELOG.md` was edited by roughly a third of all pull requests, so under a strict,
moving `main` it was the single most frequent rebase conflict — every landed entry
collided with every open branch's entry. Entries now live as one file per PR under
`changelog.d/` (a `#### Title` + prose block in the existing house style), and
`make changelog` folds them into `CHANGELOG.md` under the month each entry was written
at release-prep, deleting the fragments. A PR now only ever touches its own new file, so
it never conflicts; `CHANGELOG.md` itself changes exactly once, at release. As a
belt-and-suspenders for direct edits during the transition, `CHANGELOG.md` also gets a
`merge=union` attribute so a rebase keeps both sides' lines instead of stopping.
