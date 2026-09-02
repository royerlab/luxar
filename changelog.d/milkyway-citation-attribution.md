#### The Milky Way dust demo credits all three of its authors

The interstellar-dust reconstruction is Leike, Glatzle & Enßlin (2020), *Resolving
nearby dust clouds*, A&A 639, A138 — three authors. Six strings in the demo cited
it as "Leike & Enßlin 2020", dropping the middle author, including the scene title
and the on-screen caption that ships inside every built scene. The demo's own
citation block already had the correct `Leike et al. 2020`, so the two forms
disagreed with each other.

The caption was worse than a wording slip. `add_demo_caption` appends the
citation, so the rendered credit read "Leike & Enßlin 2020 • 3D dust density • ~1
pc/voxel • Leike et al. 2020" — the work named twice, once wrongly. The caption no
longer names the reference at all and lets the helper supply it, which is the
convention the rest of the corpus follows and what
`test_corpus_captions_do_not_repeat_their_compact_reference` exists to enforce.
That test passed before only because the duplicate was hidden behind the
misattribution.

Found by resolving every cited DOI in the demo corpus against Crossref, DataCite
and Zenodo and comparing the author count to the citation form. All 52 other
demos check out, as do the journal, volume, page and year of this one.
