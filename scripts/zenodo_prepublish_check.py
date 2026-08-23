"""Pre-publish gate for the three Zenodo drafts. READ-ONLY: no writes, no publish.

Run this immediately before publishing by hand. Every check is one an error
actually made during this campaign, so none of them is hypothetical.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

MAN = Path("packages/luxar/src/luxar/demos/data_manifest.json")
RECORDS = (("cc-by", 21912280), ("cc-by-sa", 21912282), ("h2afva", 21912284))
TOK = os.environ["ZENODO_TOKEN"]


def dep(i: int) -> dict:
    """GET one deposition. Read-only: this module issues no other verb."""
    req = urllib.request.Request(  # noqa: S310 - literal https URL, built here
        f"https://zenodo.org/api/deposit/depositions/{i}", method="GET"
    )
    # UNREDIRECTED, matching scripts/zenodo_upload_draft.py: urllib's redirect
    # handler copies Request.headers onto the follow-up request, including to
    # whatever host a Location names. This also keeps the token out of the URL.
    req.add_unredirected_header("Authorization", f"Bearer {TOK}")
    with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310
        return json.load(resp)


def human(b):
    mb = b / 1e6
    return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb:.1f} MB"


man = json.loads(MAN.read_text())
pins, ds_total = {}, {}
for name, d in man["datasets"].items():
    if d.get("bucket") != "zenodo":
        continue
    tot = 0
    lists = (
        [v.get("files", []) for v in d["variants"].values()]
        if "variants" in d
        else [d.get("files", [])]
    )
    for files in lists:
        for f in files:
            pins[f["name"]] = (d["record"], f["bytes"], f["sha256"])
            tot += f["bytes"]
    ds_total[name] = tot

fails, warns = [], []
for rec, i in RECORDS:
    d = dep(i)
    m, files = d["metadata"], d["files"]
    names = {f["filename"]: f for f in files}
    tag = f"[{rec}]"

    if d["submitted"]:
        fails.append(f"{tag} ALREADY SUBMITTED — stop")
    # 1. every hosted file is pinned at the right size
    for n, f in names.items():
        p = pins.get(n)
        if p is None:
            fails.append(f"{tag} {n} on the record but NOT pinned in the manifest")
        elif p[1] != f["filesize"]:
            fails.append(f"{tag} {n} pinned {p[1]:,} but hosted {f['filesize']:,}")
    for n, (r, b, _) in pins.items():
        if r == rec and n not in names:
            fails.append(f"{tag} {n} pinned but ABSENT from the record")
    # 2. no scratch/probe files
    for n in names:
        if n.startswith("_") or n.endswith((".corrupt", ".part", ".tmp")):
            fails.append(f"{tag} scratch file on the record: {n}")
    # 3. description sizes agree with reality
    desc = m.get("description", "")
    for mt in re.finditer(r"<li><code>([^<]+)</code>\s*\(([\d.]+\s?[MG]B)", desc):
        entry, claimed = mt.group(1), mt.group(2)
        b = ds_total.get(entry) or (pins.get(entry) or (None, None))[1]
        if b is None:
            warns.append(f"{tag} description names {entry}, not resolvable to a pin")
        elif claimed.replace(" ", "") != human(b).replace(" ", ""):
            fails.append(
                f"{tag} description says {entry} is {claimed}, actually {human(b)}"
            )
    # 4. the contents table is present and sized right
    rows = desc.count("<tr>")
    if "<table>" not in desc:
        fails.append(f"{tag} description has no contents table")
    elif rows != len(names) + 1:
        fails.append(
            f"{tag} table has {rows} rows for {len(names)} files (want {len(names) + 1})"
        )
    if "<thead" in desc:
        warns.append(f"{tag} description contains <thead>, which Zenodo strips")
    # 5. publication switches still off (they flip AT publish time, by hand)
    r = man["records"][rec]
    if r.get("published") or r.get("base_url"):
        warns.append(f"{tag} manifest already marks this published/base_url set")
    # 6. required metadata
    for k in ("title", "creators", "license", "upload_type"):
        if not m.get(k):
            fails.append(f"{tag} metadata missing {k}")

print(f"checked {len(RECORDS)} records, {len(pins)} pins")
for w in warns:
    print(f"  WARN {w}")
for f in fails:
    print(f"  FAIL {f}")
print(
    f"\n{'READY' if not fails else 'NOT READY'}: {len(fails)} failures, {len(warns)} warnings"
)
sys.exit(1 if fails else 0)
