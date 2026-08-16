#### The demo records cannot be published by accident, and cannot mis-state a license

A Zenodo record carries one license field, so the demo datasets are grouped
into records by license family while each keeps its own true license alongside.
That model was documented and correct but unenforced, and the failure it admits
is quiet: a ShareAlike dataset filed under the CC-BY record would be offered
under terms its source does not allow, and nothing would say so until someone
downloaded it.

A gate now checks that every hosted dataset names a real record, that its
license is one the record's family can actually offer (public-domain
dedications qualify — a record's license is a floor, not a claim), that nothing
marked non-redistributable or carrying NonCommercial terms or an unresolved
license is hosted at all, and that every hosted dataset credits its source.
`gsplats_tribolium`, whose license is recorded as an outright authority
conflict, and `milky_way_gaia_3m`, which is CC BY-NC, are the live cases this
keeps out.

It also pins `published: false` on all three records. Whether the drafts go
public is the maintainer's decision, taken by hand on Zenodo; a change that
flipped it here would now land in a test first.
