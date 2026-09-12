#### Zenodo records published; demos now fetch from them

All four demo-data Zenodo records were published on 2026-09-02, so `published` is now true
for each and the fetch leg is live: `ensure_dataset` builds real download URLs
and verifies every file against its recorded SHA-256. Concept DOIs, which Zenodo
mints only at publication, are recorded alongside the version DOIs — those are
the identifiers to cite, since they follow the latest version:

    cc-by            10.5281/zenodo.21912279
    cc-by-sa         10.5281/zenodo.21912281
    h2afva           10.5281/zenodo.21912283
    droso-timelapse  10.5281/zenodo.22118694

Also corrects the Drosophila credit. That recording was made in Philipp J.
Keller's lab at HHMI Janelia Research Campus, where L. A. Royer was then a
postdoctoral fellow — before his own lab existed. The manifest and three demo
docstrings had credited it to the "Royer & Keller labs", which misplaces the
acquisition; the splat fits were computed later at CZ Biohub SF.
