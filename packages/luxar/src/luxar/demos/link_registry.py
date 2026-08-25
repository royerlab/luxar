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
            "https://simbad.cds.unistra.fr/simbad/sim-basic?Ident=Sun",
        }
    ),
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
    "www.uniprot.org": frozenset(
        {"https://www.uniprot.org/uniprotkb/{hover_key}/entry"}
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
        "good": ("Betelgeuse", "Rigel", "Sun"),
        "bad": "LUXAR_NO_SUCH_STAR_2089",
        "good_marker": ("<h1>Betelgeuse", "<h1>Rigel", "<h1>Sun"),
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
    "www.uniprot.org": {
        "mode": "status",
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
