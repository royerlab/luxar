#### Point the README at the Zenodo demo-data records

The demo data is archived on Zenodo in four records, published 2026-09-02, but
the README never said so — it credited the upstream datasets and left the
derived artifacts the demos actually download unaccounted for. A new
"Where the demo data lives" section names the four records with their concept
DOIs, distinguishes lossy splat fits from derived analysis tables, and explains
the checksum guarantees and the temporary dual-pin caveat. The Citation section
now points at them too.

The gallery now credits its data in place. Each of the 21 tiles built on an
outside dataset carries a short credit under its title — `Blin et al. 2019`,
`Allen Institute for Cell Science`, `CAIDA, UC San Diego` — so a reader
scrolling the images sees whose work they are looking at without scrolling to
Acknowledgments, where the full citation, licence and link still live. The eight
synthetic tiles are left blank, with one line above the tables explaining that a
blank credit means Luxar generated the data itself.

Each tile title is also a link to that exact scene on the live demo site, whose
stable `/d/<demo-key>` routes make this possible; the images keep linking to
their full-resolution video. All 29 routes were checked over HTTP.
