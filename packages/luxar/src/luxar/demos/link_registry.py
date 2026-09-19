"""Canonical demo links and their opt-in external destination audits.

The static demo-link guard consumes the canonical templates while
``scripts/check_demo_links.py`` derives its public-page probes from the same
templates. Keep this registry aligned with the sibling external-reference
checks: Sphinx ``linkcheck`` (``make check-docs-external-links``) and the Zenodo
record pins in ``scripts/gen_data_manifest.py`` added in #1734.
"""

from __future__ import annotations

from typing import Any

CANONICAL_LINKS_BY_HOST = {
    "bgp.he.net": frozenset({"https://bgp.he.net/AS{hover_key}"}),
    "codex.flywire.ai": frozenset(
        {"https://codex.flywire.ai/app/cell_details?root_id={hover_key}"}
    ),
    "doi.org": frozenset({"https://doi.org/{hover_key}"}),
    "earthquake.usgs.gov": frozenset(
        {"https://earthquake.usgs.gov/earthquakes/eventpage/{hover_key}"}
    ),
    # Both placeholders intentionally drive the same Wikipedia search endpoint.
    "en.wikipedia.org": frozenset(
        {
            "https://en.wikipedia.org/wiki/Special:Search?search={hover_label}",
            "https://en.wikipedia.org/wiki/Special:Search?search={hover_key}",
        }
    ),
    # Keep this literal aligned with demo_dipc_3d_genome.GENOME_ASSEMBLY.
    "genome.ucsc.edu": frozenset(
        {"https://genome.ucsc.edu/cgi-bin/hgTracks?db=hg19&position={hover_key}"}
    ),
    # These named-star links are deliberately placeholder-free.
    "simbad.cds.unistra.fr": frozenset(
        {
            "https://simbad.cds.unistra.fr/simbad/sim-basic?Ident=Betelgeuse",
            "https://simbad.cds.unistra.fr/simbad/sim-basic?Ident=Rigel",
        }
    ),
    "science.nasa.gov": frozenset({"https://science.nasa.gov/sun/"}),
    "ssd.jpl.nasa.gov": frozenset(
        {"https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr={hover_key}"}
    ),
    # OLS supports either the key or the display label as its search query.
    "www.ebi.ac.uk": frozenset(
        {
            "https://www.ebi.ac.uk/ols4/search?q={hover_key}",
            "https://www.ebi.ac.uk/ols4/search?q={hover_label}",
        }
    ),
    "www.genecards.org": frozenset({"https://www.genecards.org/card/{hover_key}"}),
    "www.proteinatlas.org": frozenset(
        {"https://www.proteinatlas.org/search/{hover_key}"}
    ),
    # NED resolves "PGC<number>" by name; cosmicflows keys its galaxies on the
    # Principal Galaxies Catalogue number because its hover LABEL is the basin
    # of attraction, which tens of thousands of galaxies share and which
    # therefore resolves nothing.
    "ned.ipac.caltech.edu": frozenset(
        {"https://ned.ipac.caltech.edu/byname?objname={hover_key}"}
    ),
    # Decorated-label search, for the arXiv/bioRxiv/medRxiv corpus specifically. The
    # per-paper DOI (doi.org above) is the real destination; this is the
    # fallback for a cached bundle that predates stored ids, where the only
    # per-point identity is the hover label. Scholar rather than arXiv search
    # because that corpus spans all three preprint servers.
    "scholar.google.com": frozenset(
        {"https://scholar.google.com/scholar?q={hover_label}"}
    ),
    # The /uniprotkb/<accession>/entry form is the canonical deep link and is
    # what both protein demos use when real accessions are available. The
    # ?query= search form is the reviewed fallback for an ESM3 cache that
    # supplies no accessions. Its query is the clean protein-name key, not the
    # decorated hover label that UniProt's parser rejects.
    "www.uniprot.org": frozenset(
        {
            "https://www.uniprot.org/uniprotkb/{hover_key}/entry",
            "https://www.uniprot.org/uniprotkb?query={hover_key}",
            "https://www.uniprot.org/uniref?query={hover_key}",
        }
    ),
    "www.youtube.com": frozenset(
        {"https://www.youtube.com/results?search_query={hover_key}"}
    ),
}
CANONICAL_LINKS = frozenset(
    template for templates in CANONICAL_LINKS_BY_HOST.values() for template in templates
)

# API-backed probes are used only where the public page cannot discriminate a
# valid identifier. The checker still derives and requests the canonical page
# route from CANONICAL_LINKS_BY_HOST before it accepts the API result.
DEMO_LINK_AUDITS_BY_HOST: dict[str, dict[str, Any]] = {
    "bgp.he.net": {
        "mode": "body-marker",
        "good": "15169",
        "bad": "4294967295",
        # The ASN is echoed from the URL; the organization name proves a record match.
        "good_marker": "AS15169 Google LLC",
    },
    "codex.flywire.ai": {
        "mode": "human",
        "reason": "the app shell does not expose cell validity to a plain request",
        "verified_in": "#2091",
        "last_checked": "2026-08-24",
    },
    "doi.org": {
        "mode": "status",
        "good": "10.48550/arXiv.2101.12345",
        "bad": "10.48550/arXiv.0000.00000",
    },
    "earthquake.usgs.gov": {
        "mode": "status",
        # The client-routed public shell answers 200 for arbitrary paths, so this
        # request confirms availability but cannot prove the route is unchanged.
        "url_template": (
            "https://earthquake.usgs.gov/fdsnws/event/1/query"
            "?format=geojson&eventid={value}"
        ),
        "good": "us7000dflf",
        "bad": "not-an-event",
    },
    "en.wikipedia.org": {
        "mode": "redirect",
        "good": "TP53",
        "bad": "LUXAR_NO_SUCH_ARTICLE_2089",
        "good_final_marker": "/wiki/TP53",
    },
    "genome.ucsc.edu": {
        "mode": "human",
        "reason": "Cloudflare Turnstile returns the same interstitial for both loci",
        "verified_in": "#2091",
        "last_checked": "2026-08-24",
    },
    "simbad.cds.unistra.fr": {
        "mode": "body-marker",
        "good": ("Betelgeuse", "Rigel"),
        "bad": "LUXAR_NO_SUCH_STAR_2089",
        "good_marker": ("<h1>Betelgeuse", "<h1>Rigel"),
    },
    "science.nasa.gov": {
        "mode": "status",
        "good": ("sun",),
        "bad": "luxar-no-such-sun-2089",
    },
    "ssd.jpl.nasa.gov": {
        "mode": "human",
        "reason": "the identifier is URL-fragment state and is never sent in HTTP",
        "verified_in": "#2091",
        "last_checked": "2026-08-24",
    },
    "www.ebi.ac.uk": {
        "mode": "json-count",
        "url_template": "https://www.ebi.ac.uk/ols4/api/search?q={value}&rows=1",
        "good": "CL:0000540",
        "bad": "LUXAR_NO_SUCH_TERM_2089",
        "count_path": ("response", "numFound"),
    },
    "www.genecards.org": {
        "mode": "human",
        "reason": "Cloudflare returns 403 for both valid and invalid genes",
        "verified_in": "#2091",
        "last_checked": "2026-08-24",
    },
    "www.proteinatlas.org": {
        "mode": "json-count",
        "url_template": (
            "https://www.proteinatlas.org/api/search_download.php"
            "?search={value}&format=json&columns=g&compress=no"
        ),
        "good": "TP53",
        "bad": "LUXAR_NO_SUCH_GENE_2089",
        "count_path": (),
    },
    "ned.ipac.caltech.edu": {
        "mode": "body-marker",
        # The byname page is a Drupal shell that fetches its result client-side,
        # so a bogus name answers 200 with a body carrying no object data at all
        # and differing only in the echoed name and the per-request Drupal
        # tokens. Its own data route is not reachable from outside the page, so
        # this probes the sibling name resolver, which omits the Preferred
        # block when nothing resolves.
        "url_template": "https://ned.ipac.caltech.edu/srs/ObjectLookup?name={value}",
        "good": "PGC17223",
        "bad": "LUXARNOSUCH2089",
        # The resolved object name proves the PGC number matched a record.
        "good_marker": "Large Magellanic Cloud",
    },
    "scholar.google.com": {
        "mode": "human",
        # A result-container class does discriminate a hit from a miss here, so
        # the obstacle is not the marker but the rate limiter: this user agent
        # is served real results at first, then a bot check and sustained 429s
        # once the address is flagged, and the flagged state outlasts a run. The
        # audit reads that as "good and bad both rejected" — a moved route — so
        # its verdict would track Google's throttling, not the URL. (The cheap
        # markers are out anyway: the query is echoed back on a miss, and result
        # titles are term-bolded, so a queried title is never contiguous.)
        "reason": "a probe sees Google's rate limiter rather than the route",
        "verified_in": "#2091",
        "last_checked": "2026-08-24",
    },
    "www.uniprot.org": {
        "mode": "status",
        # The client-routed public shell answers 200 for arbitrary paths, so this
        # request confirms availability but cannot prove the route is unchanged.
        "url_template": "https://rest.uniprot.org/uniprotkb/{value}",
        "good": "P04637",
        "bad": "LUXAR2089",
    },
    "www.youtube.com": {
        "mode": "human",
        "reason": "search accepts every query and bogus text still returns results",
        "verified_in": "#2091",
        "last_checked": "2026-08-24",
    },
}

# Keep in sync with hover-template.ts's PLACEHOLDER_PATTERN; hover_image_label is HTML-only.
LINK_PLACEHOLDERS = ("hover_key", "hover_label", "hover_node", "hover_index")
