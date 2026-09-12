"""Report where a captured Zenodo record contradicts the manifest's attribution.

`data_manifest.json` and `scripts/zenodo_record_text/*.html` hold two
independent copies of the same provenance claim: the manifest's ``attribution``
field, and the record description as published on Zenodo. Nothing compared
them, which is how "Imaging described in Royer et al., ..." survived on the
Drosophila timelapse record while the manifest said the imaging itself is
unpublished and the 2016 paper only describes the *instrument*. Saying a paper
describes the imaging turns the record into a data publication it is not.

Saying a paper describes the *instrument* or the *method* is the correct framing
and is never a finding: "the SiMView instrument is described in Royer et al."
and "Imaging method: Royer, L. A., ..." both have to stay clean. So is saying
the *archive* was "published as" a deposit, which is not a publication verb
this audit looks for at all.

This is an offline, report-only audit (stdlib only, no network). It cannot be a
hard test: the live record text is authored on Zenodo by the maintainer, so a
real contradiction persists in the repository until that edit and the following
re-capture land. `make check-external-references` is the report-only home.

That extends to this repository's own tests. `scripts/tests` runs inside
`ci.yml`'s **required** `python-tests` job, and `^scripts/zenodo_record_text/`
is in that workflow's python diff domain, so the one test that exercises the
live files asserts only repository-controlled properties - never a count, a
level, or the absence of a finding. Otherwise the re-capture PR that carries the
fix would be the thing that reddens the gate.

Report lines start with a level marker from the shared audit vocabulary:

``[OK]``
    No claim that a publication describes the imaging was found for a dataset.
``[HUMAN]``
    Record-level prose asserts a publication describes the imaging, but that
    prose names no dataset and not every dataset on the record claims
    unpublished imaging, so a person has to decide who it refers to.
``[STALE]``
    A dataset's own record text still asserts a publication describes the
    imaging.
``[CONFIG]``
    Nothing could be compared: an input is missing, an input is unreadable, or
    no dataset anywhere was in scope at all (the fail-closed case, where no
    input is missing and there is simply nothing for this audit to check).

`scripts/run_external_reference_audits.py` does **not** grade those markers the
way it grades a ``parse_levels=True`` producer: this leg is registered with
``parse_levels=False``, so its grade comes from the exit status, and only
``[CONFIG]`` participates (via that runner's ``_COMMAND_ERROR_MARKERS``, which
promotes a non-zero exit to ERROR). Everything else that must be seen therefore
has to exit non-zero, ``[HUMAN]`` included, or it would grade PASS and vanish.

Exit status: 0 clean, 1 drift found or a human check is needed, 2 an input is
broken.

Bounds
------
Only Zenodo-hosted manifest datasets are inspected, and only those where either
the manifest attribution or the dataset's own record bullet states that the
imaging is unpublished. That scope is resolved *before* any input-level
``[CONFIG]``, so a manifest entry on a record nobody has captured yet is silent
rather than a broken input. A dataset is tied to a bullet by that bullet's
*leading* ``<code>`` label; later ``<code>`` spans in a bullet are prose (a
genotype, a sibling file name) and never claim it. Record-level prose names no
dataset, so it is blamed on a dataset only when every dataset on the record
makes the unpublished-imaging claim, and is a record-scoped ``[HUMAN]`` line
otherwise.

Detection is a proximity rule, and its recall bounds are all deliberate. A
claim is not seen when the text between "imaging" and the publication verb

* is longer than 80 characters, or crosses a full stop or a **semicolon** -
  which is what keeps "Imaging was done; the method is described in X" clean,
  at the cost of not seeing a claim written across an inline genotype like
  ``w; His2Av::mRFP1; +``;
* ends with a replacement head noun (instrument, microscope, method, protocol,
  pipeline, technique, setup, apparatus, software), because then that noun and
  not the imaging is what the publication covers;
* ends with "not" or "never" and an optional short tail, because a denial is
  the opposite of the claim;
* ends in a manner adverbial ("... as previously described in ...");

or when the imaging is demoted to the object of a preposition ("details *of the
imaging* are described in ..."). The audit reports; it never decides.
"""

from __future__ import annotations

import html
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "packages/luxar/src/luxar/demos/data_manifest.json"
SNAPSHOT_DIR = REPO_ROOT / "scripts/zenodo_record_text"

# The imaging has to be the subject. The manifest spells this "the imaging
# itself is unpublished" (x3); record text uses the bare "This imaging is
# unpublished", and a reword to "has never been separately published" is just as
# plausible, so the negated-publication shapes are matched too. This pattern is
# applied to record bullets as well as to manifest attributions, so a one-sided
# reword on either side cannot quietly drop a dataset out of the audit.
_UNPUBLISHED_IMAGING = re.compile(
    r"\b(?:"
    r"unpublished\s+imaging"
    r"|imaging\b[^.;]{0,60}?\b(?:"
    r"unpublished"
    r"|(?:not|never)\s+(?:\w+\s+){0,3}?(?:published|deposited|released)"
    r")"
    r")\b",
    re.IGNORECASE,
)
# The contradiction is a candidate verb phrase within one sentence of the word
# "imaging"; `_asserts_a_publication_covers_the_imaging` then decides whether
# the imaging is really what the publication covers. The window is 80 characters
# because the live records write long inline species and acquisition strings
# ("Imaging of this His2Av::mRFP1 Drosophila melanogaster embryo is described
# in ..." spends 57 of them), and `[^.;]` keeps it inside one clause.
# The `;` bound is what it costs: the live genotype is written `w; His2Av::mRFP1;
# +`, so "Imaging of this w; His2Av::mRFP1; + embryo is described in ..." is NOT
# seen. That is the accepted trade - crossing a semicolon would flag "Imaging was
# done; the method is described in X", where the two clauses have two subjects.
# The droso intro's true "... fitted to 3D Gaussian splats per timepoint and
# published as a single 4D archive" survives this for two independent reasons:
# that sentence says "imaged", not "imaging", so the anchor never fires at all;
# and were it reworded to "imaging", "published AS" is not "published in|by".
_PUBLICATION_DESCRIBES_IMAGING = re.compile(
    r"\bimaging\b(?P<gap>[^.;]{0,80}?)"
    r"\b(?:described|documented|reported|published|presented|detailed"
    r"|characterised|characterized)\s+(?:in|by)\b",
    re.IGNORECASE,
)
# "The imaging has not been published in any article", "The imaging data are not
# published in any paper": a denial is the opposite of the claim being hunted.
# Only `not` and `never` negate the publication verb. `no`/`nor`/`without` reach
# a modifier of the imaging instead ("with no post-processing", "without further
# processing", "with no deconvolution nor denoising") and suppressed real claims.
# "(no. 3 of the series)" stays out of scope regardless - the abbreviation's full
# stop bounds the clause, which is a different rule.
_NEGATED = re.compile(r"\b(?:not|never)\b[^,;]{0,24}$", re.IGNORECASE)
# A replacement HEAD NOUN: something other than the imaging is what the
# publication covers. These are the maintainer's correct framings, and they must
# never be flagged. Artefact nouns (archive, record, paper, fit, splat) are
# deliberately absent: this gap is also where a locative or partitive modifier
# OF the imaging sits, so listing them silently cleared "The imaging in this
# archive is described in Royer et al." and its siblings while earning nothing -
# removing them left the live report byte-identical.
_INTERVENING_SUBJECT = re.compile(
    r"\b(?:instruments?|microscopes?|microscopy|methods?|methodology|protocols?"
    r"|pipelines?|techniques?|setup|apparatus|software)\b[^,;]{0,24}$",
    re.IGNORECASE,
)
# "Imaging was carried out as described in Royer et al.", "Imaging was performed
# as previously described in ..." - a manner adverbial, not an assertion that the
# paper describes this imaging. The window tolerates an intervening adverb but
# stops at a comma or semicolon, so "The imaging is, as we note, described in
# ..." is still a claim about the imaging.
_MANNER_CLAUSE = re.compile(r"\bas\b[^,;]{0,24}$", re.IGNORECASE)
# "The microscope used for this imaging is described in Royer et al." - the
# imaging is the object of a preposition, so it is not what "is described".
_IMAGING_IS_NOT_THE_SUBJECT = re.compile(
    r"\b(?:for|of|in|on|during|with|by|from|to|about|through|after|before)\s+"
    r"(?:this|that|the|these|those|our|their|its|such)?\s*$",
    re.IGNORECASE,
)
_SUBJECT_CONTEXT_CHARS = 40
# A bullet or label may carry attributes (`<li class="...">`) or nested markup;
# matching the bare tag would misattribute one dataset's bullet to another. No
# attribute-bearing `<li>` or `<code>` exists in any of the four live snapshots
# today, so this is a guard against a plausible edit, not a fixed defect.
_LIST_ITEM = re.compile(r"<li\b[^>]*>.*?</li>", re.DOTALL)
_CODE = re.compile(r"<code\b[^>]*>(.*?)</code>", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
# A block boundary is a sentence boundary. `<h3>` headings and a `<td><code>`
# table are both live, and joining a heading to the paragraph after it with a
# space alone reads as one sentence ("Imaging Documented in Royer et al.") that
# neither element asserts.
_BLOCK_END = re.compile(r"</(?:p|h[1-6]|li|td|th|tr)\s*>|<br\s*/?>", re.IGNORECASE)
_BLOCK_MARK = "\x00"
_BLOCK_BOUNDARY = re.compile(rf"\s*(?:{_BLOCK_MARK}\s*)+")
# A full stop only ends a sentence when the next word starts one AND the word it
# follows is not an abbreviation. Citations are thick with "et al.", "L. A." and
# "Nat. Biotechnol.", and cutting a quote at the first period drops the very
# reference that makes it actionable.
_SENTENCE_END = re.compile(r"\.(?=\s+[A-Z(]|\s*$)")
_ABBREVIATIONS = frozenset(
    {
        # Citation vocabulary.
        "al",
        "ed",
        "eds",
        "cf",
        "vs",
        "approx",
        "fig",
        "figs",
        "no",
        "vol",
        "pp",
        "dr",
        "prof",
        "univ",
        "inst",
        "dept",
        # Journal-name abbreviations. Only `nat` and `biotechnol` occur in the
        # manifest today, and only `al` and `nat` are ever consulted by the
        # live corpus; the rest is forward-looking, for a citation reworded or
        # a dataset added later.
        "acad",
        "am",
        "annu",
        "biol",
        "biophys",
        "biotechnol",
        "chem",
        "commun",
        "curr",
        "dev",
        "eur",
        "genet",
        "lett",
        "med",
        "methods",
        "microsc",
        "mol",
        "nat",
        "natl",
        "neurosci",
        "opt",
        "phys",
        "proc",
        "res",
        "rev",
        "sci",
        "struct",
    }
)
_MAX_QUOTE_CHARS = 240

REMEDY = (
    "Remedy: the snapshots under scripts/zenodo_record_text/ are the "
    "authoritative capture of the LIVE record text, not the output of "
    "scripts/gen_zenodo_records.py (see that directory's README.md). Fix the "
    "wording on Zenodo, then re-capture with "
    "`python3 scripts/zenodo_record_text/capture.py`. Never hand-edit the "
    "HTML and never paste `gen_zenodo_records.py --render` output into it: the "
    "published descriptions carry framing the generator does not produce."
)
_ATTENTION_LEVELS = frozenset({"STALE", "HUMAN", "CONFIG"})


@dataclass(frozen=True)
class Finding:
    """One report line: a level marker, who it is about, and why."""

    level: str
    record: str
    dataset: str
    message: str
    quotes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecordText:
    """A captured record description, split into bullets and record prose.

    A bullet carries at most one label - its *leading* ``<code>`` span - so the
    label is a single optional string rather than a set.
    """

    bullets: tuple[tuple[str | None, str], ...]
    prose: str


def _as_sentence_break(match: re.Match[str]) -> str:
    """A block boundary reads as ". " unless the text already terminated."""
    preceding = match.string[: match.start()].rstrip()
    return " " if preceding.endswith((".", "!", "?")) else ". "


def _plain_text(fragment: str) -> str:
    """Strip markup and collapse whitespace so patterns see plain prose.

    Block-closing tags become sentence terminators first: a heading, a list
    item or a table cell is a sentence of its own, and joining it to the next
    block with a space alone manufactures a claim neither block makes.

    Tags are stripped *before* entities are unescaped; the reverse order turns
    a literal ``&lt;dataset&gt;`` into a tag and eats the word inside it.
    """
    marked = _BLOCK_END.sub(_BLOCK_MARK, fragment)
    text = re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", marked)))
    return _BLOCK_BOUNDARY.sub(_as_sentence_break, text).strip()


def _parse_record_text(source: str) -> RecordText:
    """Split a captured description into per-file bullets and record prose.

    Each bullet is paired with its leading ``<code>`` label, which is how a
    bullet is tied back to a manifest dataset. Record prose is everything
    outside any bullet.
    """
    bullets = []
    for item in _LIST_ITEM.findall(source):
        leading = _CODE.search(item)
        label = _plain_text(leading.group(1)) if leading else ""
        bullets.append((label or None, _plain_text(item)))
    prose = _plain_text(_LIST_ITEM.sub(" ", source))
    return RecordText(tuple(bullets), prose)


def _dataset_labels(dataset: dict[str, Any]) -> frozenset[str]:
    """Every label a record bullet may use for this dataset.

    The published records are inconsistent: `cc-by` and `cc-by-sa` label their
    bullets with the manifest key, while `h2afva` and `droso-timelapse` use the
    archive file names. Both are accepted, matched exactly so that
    ``h2afva`` never claims ``h2afva_51tp.gsplats.zarr.zip``'s bullet.
    """
    labels = set()
    raw_files = dataset.get("files", ())
    files = list(raw_files) if isinstance(raw_files, list) else []
    variants = dataset.get("variants", {})
    if isinstance(variants, dict):
        for variant in variants.values():
            if not isinstance(variant, dict):
                continue
            variant_files = variant.get("files", ())
            if isinstance(variant_files, list):
                files.extend(variant_files)
    for entry in files:
        if isinstance(entry, dict) and entry.get("name"):
            labels.add(str(entry["name"]))
    return frozenset(labels)


def _ends_an_abbreviation(text: str, dot: int) -> bool:
    """Whether the word ending at ``dot`` is an initial or a known abbreviation."""
    start = dot
    while start > 0 and text[start - 1].isalpha():
        start -= 1
    word = text[start:dot]
    return bool(word) and (len(word) == 1 or word.lower() in _ABBREVIATIONS)


def _sentence_ends(text: str) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` of every full stop that really ends a sentence."""
    for match in _SENTENCE_END.finditer(text):
        if _ends_an_abbreviation(text, match.start()):
            continue
        yield match.start(), match.end()


def _quote(text: str, start: int, end: int) -> str:
    """Quote the sentence the match sits in, bounded in length."""
    left = 0
    right = len(text)
    for stop_start, stop_end in _sentence_ends(text):
        if stop_end <= start:
            left = stop_end
        elif stop_start >= end:
            right = stop_end
            break
    sentence = text[left:right].strip()
    if len(sentence) > _MAX_QUOTE_CHARS:
        sentence = sentence[:_MAX_QUOTE_CHARS].rstrip() + " ..."
    return sentence


def _asserts_a_publication_covers_the_imaging(gap: str, before: str) -> bool:
    """Whether the imaging itself is what the matched publication verb covers.

    ``gap`` is the text between the word "imaging" and the verb; ``before`` is
    the text leading up to "imaging". A candidate survives only when neither
    denies the claim nor hands the verb a different subject.
    """
    if _NEGATED.search(gap) or _MANNER_CLAUSE.search(gap):
        return False
    if _INTERVENING_SUBJECT.search(gap):
        return False
    return not _IMAGING_IS_NOT_THE_SUBJECT.search(before)


def _contradictions(text: str) -> tuple[str, ...]:
    quotes = []
    for match in _PUBLICATION_DESCRIBES_IMAGING.finditer(text):
        before = text[max(0, match.start() - _SUBJECT_CONTEXT_CHARS) : match.start()]
        if _asserts_a_publication_covers_the_imaging(match.group("gap"), before):
            quotes.append(_quote(text, match.start(), match.end()))
    return tuple(quotes)


def _hosted_by_record(datasets: dict[str, Any]) -> dict[str, list[str]]:
    hosted: dict[str, list[str]] = {}
    for key, dataset in sorted(datasets.items()):
        if not isinstance(dataset, dict) or dataset.get("bucket") != "zenodo":
            continue
        hosted.setdefault(str(dataset.get("record") or ""), []).append(key)
    return hosted


def _claims_unpublished_imaging(dataset: dict[str, Any], bullets: list[str]) -> bool:
    """Either side may raise the claim, so a one-sided reword cannot disarm it.

    Gating on the manifest alone let a reworded attribution drop a dataset out
    of the audit with its record text still wrong, and left the inverse drift -
    a record claiming unpublished imaging while the manifest cites a
    publication - permanently unexamined.
    """
    attribution = str(dataset.get("attribution") or "")
    if _UNPUBLISHED_IMAGING.search(attribution):
        return True
    return any(_UNPUBLISHED_IMAGING.search(body) for body in bullets)


def audit(
    manifest_path: Path = MANIFEST_PATH,
    snapshot_dir: Path = SNAPSHOT_DIR,
) -> list[Finding]:
    """Compare every unpublished-imaging claim with its captured record text."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        datasets = manifest["datasets"]
        if not isinstance(datasets, dict):
            # Otherwise a `"datasets": "oops"` manifest escapes the documented
            # exit-2 contract as an uncaught AttributeError and a traceback.
            raise TypeError(f"'datasets' is a {type(datasets).__name__}, not an object")
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [
            Finding(
                "CONFIG",
                "-",
                "-",
                f"cannot read datasets from {manifest_path}: {error}",
            )
        ]

    findings: list[Finding] = []
    for record, keys in sorted(_hosted_by_record(datasets).items()):
        findings.extend(
            _record_findings(record, keys, datasets, snapshot_dir / f"{record}.html")
        )
    if not findings:
        # Fail closed: with nothing to compare, "no findings" would read as a
        # clean bill of health for a claim nobody checked.
        return [
            Finding(
                "CONFIG",
                "-",
                "-",
                f"no zenodo dataset in {manifest_path.name} and no captured "
                "record bullet states that the imaging is unpublished, so "
                "nothing was compared",
            )
        ]
    return findings


def _record_findings(
    record: str,
    keys: list[str],
    datasets: dict[str, Any],
    snapshot: Path,
) -> list[Finding]:
    """Every finding one captured record contributes.

    Scope is resolved before any input-level ``CONFIG``. A record whose
    datasets make no unpublished-imaging claim is not this audit's business, so
    an uncaptured record - the ordinary manifest-first, capture-later sequence -
    is silent instead of escalating the whole aggregator leg to ERROR and
    relabelling real drift as "could not run". Only the manifest side can be
    consulted before the snapshot is read, which is exactly the side that
    survives a missing or bullet-less capture.
    """
    claimed_in_manifest = [
        key
        for key in keys
        if _UNPUBLISHED_IMAGING.search(str(datasets[key].get("attribution") or ""))
    ]
    try:
        source = snapshot.read_text(encoding="utf-8")
    except OSError as error:
        if not claimed_in_manifest:
            return []
        return [
            Finding(
                "CONFIG",
                record or "(unset)",
                ", ".join(claimed_in_manifest),
                f"no captured record text to compare: {error}",
            )
        ]
    text = _parse_record_text(source)
    if not text.bullets:
        if not claimed_in_manifest:
            return []
        return [
            Finding(
                "CONFIG",
                record or "(unset)",
                ", ".join(claimed_in_manifest),
                f"{snapshot} has no per-file bullets, so no per-dataset claim "
                "could be located",
            )
        ]

    bullets = {key: _bullets_for(datasets[key], key, text) for key in keys}
    inspected = [
        key for key in keys if _claims_unpublished_imaging(datasets[key], bullets[key])
    ]
    if not inspected:
        return []

    # Record prose names no dataset, so it may only be blamed on one when every
    # dataset on the record makes the same unpublished-imaging claim.
    prose_is_unambiguous = set(inspected) == set(keys)
    prose_quotes = _contradictions(text.prose)
    findings = [
        _dataset_finding(
            record,
            key,
            bullets[key],
            prose_quotes if prose_is_unambiguous else (),
            snapshot,
        )
        for key in inspected
    ]
    if prose_quotes and not prose_is_unambiguous:
        findings.append(
            Finding(
                "HUMAN",
                record or "(unset)",
                "(record prose)",
                f"{snapshot.name} asserts a publication describes the imaging, "
                "but record prose names no dataset and only some of this "
                f"record's datasets ({', '.join(inspected)}) claim unpublished "
                "imaging, so a human must decide which dataset it refers to",
                tuple(f"record prose: {quote}" for quote in prose_quotes),
            )
        )
    return findings


def _bullets_for(dataset: dict[str, Any], key: str, text: RecordText) -> list[str]:
    """The bodies of the bullets whose leading label names this dataset."""
    labels = _dataset_labels(dataset) | {key}
    return [body for label, body in text.bullets if label in labels]


def _dataset_finding(
    record: str,
    key: str,
    bullets: list[str],
    prose_quotes: tuple[str, ...],
    snapshot: Path,
) -> Finding:
    if not bullets:
        return Finding(
            "CONFIG",
            record or "(unset)",
            key,
            f"no bullet in {snapshot.name} names {key} or any of its files, so "
            "its attribution could not be compared",
        )
    quotes = [f"bullet: {quote}" for body in bullets for quote in _contradictions(body)]
    quotes += [f"record prose: {quote}" for quote in prose_quotes]
    if not quotes:
        return Finding(
            "OK",
            record or "(unset)",
            key,
            f"no claim that a publication describes the imaging was found in "
            f"{snapshot.name}",
        )
    return Finding(
        "STALE",
        record or "(unset)",
        key,
        "the manifest says the imaging itself is unpublished, but "
        f"{snapshot.name} asserts a publication describes the imaging",
        tuple(quotes),
    )


def render(findings: list[Finding]) -> str:
    """Render the report; one marker line per dataset, quotes indented."""
    lines = []
    for finding in findings:
        lines.append(
            f"[{finding.level}]".ljust(9)
            + f"{finding.dataset} (record {finding.record}): {finding.message}"
        )
        lines.extend(f"         {quote}" for quote in finding.quotes)
    levels = {finding.level for finding in findings}
    if levels & _ATTENTION_LEVELS:
        lines.extend(("", REMEDY))
    return "\n".join(lines) + "\n"


def exit_status(findings: list[Finding]) -> int:
    """0 clean, 1 needs attention, 2 an input is broken (broken input wins).

    A ``HUMAN`` finding exits non-zero for the same reason a ``STALE`` one
    does: the audit runner grades this leg by exit status, so a zero exit would
    grade PASS and hide the line.
    """
    levels = {finding.level for finding in findings}
    if "CONFIG" in levels:
        return 2
    return 1 if levels & {"STALE", "HUMAN"} else 0


def main() -> int:
    findings = audit()
    print(render(findings), end="")
    return exit_status(findings)


if __name__ == "__main__":
    sys.exit(main())
