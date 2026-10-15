"""Guard the authoring rules in ``.agents/skills/README.md``.

The README asks every ``SKILL.md`` to stay under 500 lines and to push detail
into ``references/``. Nothing enforced it, and ``luxar-gsplat-pipeline`` grew to
777 lines before a blind review graded it unusable for that reason: an agent
loads the whole SKILL.md on every invocation, so lore that only one question in
twenty needs is paid for by all twenty. The cure — moving lore into
``references/*.md`` — has its own failure mode, a reference nothing links to, so
it is never loaded at all. Both are checked here, along with the frontmatter and
the ``.claude/skills`` symlink the README tells authors to add.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
SKILLS_DIR = REPO / ".agents" / "skills"
CLAUDE_SKILLS_DIR = REPO / ".claude" / "skills"

#: The README's "Keep SKILL.md lean (< ~500 lines)" rule, as a hard ceiling.
MAX_SKILL_LINES = 500

_FRONTMATTER = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.DOTALL)


def _skill_dirs() -> list[Path]:
    if not SKILLS_DIR.is_dir():  # installed wheel: no repository tree
        return []
    return sorted(p for p in SKILLS_DIR.iterdir() if (p / "SKILL.md").is_file())


SKILLS = _skill_dirs()


def test_the_skills_directory_is_found() -> None:
    """Fail closed: an empty parametrization would pass every check below."""
    if not SKILLS_DIR.is_dir():
        pytest.skip(".agents/skills not available (installed package)")
    assert SKILLS, f"no SKILL.md found under {SKILLS_DIR}"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.name)
def test_skill_md_stays_under_the_line_budget(skill: Path) -> None:
    n_lines = len((skill / "SKILL.md").read_text(encoding="utf-8").splitlines())
    assert n_lines < MAX_SKILL_LINES, (
        f"{skill.name}/SKILL.md is {n_lines} lines; the .agents/skills/README.md "
        f"rule is < {MAX_SKILL_LINES}. Move measured cases and lore into "
        f"{skill.name}/references/*.md and link them from SKILL.md, keeping the "
        "decision-making core in SKILL.md."
    )


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.name)
def test_frontmatter_names_the_skill_and_describes_it(skill: Path) -> None:
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    match = _FRONTMATTER.match(text)
    assert match, f"{skill.name}/SKILL.md must open with YAML frontmatter"
    body = match["body"]
    assert re.search(rf"^name:\s*{re.escape(skill.name)}\s*$", body, re.MULTILINE), (
        f"{skill.name}/SKILL.md frontmatter `name:` must equal its directory name"
    )
    assert re.search(r"^description:", body, re.MULTILINE), (
        f"{skill.name}/SKILL.md frontmatter needs a `description:` (it drives "
        "implicit invocation)"
    )


_MD_LINK = re.compile(r"\]\(([^)\s]+)\)")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def _slug(heading: str) -> str:
    """GitHub's heading anchor: lowercase, punctuation dropped, spaces -> '-'."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    return {_slug(h) for h in _HEADING.findall(path.read_text(encoding="utf-8"))}


def _orphan_references(skill: Path, seen: set[Path]) -> list[str]:
    return [
        f"references/{ref.name} is not reachable from SKILL.md through "
        "markdown links, so no agent will ever load it"
        for ref in sorted((skill / "references").glob("*.md"))
        if ref.resolve() not in seen
    ]


def skill_link_problems(skill: Path) -> list[str]:
    """Dead links, dead anchors, and references unreachable from SKILL.md.

    Reachability follows MARKDOWN LINKS only (``[text](references/x.md)`` or a
    relative ``[text](x.md)`` between references), starting at SKILL.md. A
    filename mentioned in prose or backticks does not count — an agent follows
    links — so two references that only link to each other are an orphan cycle.
    """
    problems: list[str] = []
    root = (skill / "SKILL.md").resolve()
    seen: set[Path] = set()
    queue = [root]
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        rel_current = current.relative_to(skill.resolve())
        for target in _MD_LINK.findall(current.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("mailto:"):
                continue
            file_part, _, anchor = target.partition("#")
            dest = (current.parent / file_part).resolve() if file_part else current
            if file_part and not file_part.endswith(".md"):
                continue
            if not dest.is_file():
                problems.append(f"{rel_current} links to {target}: no such file")
                continue
            if anchor and anchor not in _anchors(dest):
                problems.append(f"{rel_current} links to {target}: no such heading")
            if dest.is_relative_to(skill.resolve()):
                queue.append(dest)
    problems.extend(_orphan_references(skill, seen))
    return problems


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.name)
def test_every_reference_is_reachable_and_every_link_resolves(skill: Path) -> None:
    """A reference nothing links to is never loaded; a dead link loads nothing."""
    problems = skill_link_problems(skill)
    assert not problems, f"{skill.name}: " + "; ".join(problems)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_link_checker_flags_an_orphan_cycle_and_dead_links(tmp_path: Path) -> None:
    """The checker itself: prose mentions and a closed cycle do not count."""
    skill = tmp_path / "demo-skill"
    _write(
        skill / "SKILL.md",
        "# Demo\n\nSee [live](references/live.md#usage) and [gone](references/gone.md).\n"
        "Also mentions references/a.md in prose only.\n",
    )
    _write(
        skill / "references/live.md", "# Live\n\n## Usage\n\n[back](../SKILL.md#nope)\n"
    )
    _write(skill / "references/a.md", "# A\n\n[b](b.md)\n")
    _write(skill / "references/b.md", "# B\n\n[a](a.md)\n")

    problems = "\n".join(skill_link_problems(skill))

    assert "references/a.md is not reachable" in problems
    assert "references/b.md is not reachable" in problems
    assert "references/live.md is not reachable" not in problems
    assert "references/gone.md: no such file" in problems
    assert "../SKILL.md#nope: no such heading" in problems
    assert "live.md#usage" not in problems


def test_link_checker_accepts_a_fully_linked_skill(tmp_path: Path) -> None:
    skill = tmp_path / "ok-skill"
    _write(skill / "SKILL.md", "# Ok\n\n[one](references/one.md)\n")
    _write(skill / "references/one.md", "# One\n\nSee [two](two.md#the-detail).\n")
    _write(skill / "references/two.md", "# Two\n\n## The detail\n")
    assert skill_link_problems(skill) == []


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.name)
def test_claude_code_symlink_points_at_the_skill(skill: Path) -> None:
    link = CLAUDE_SKILLS_DIR / skill.name
    assert link.is_symlink(), (
        f".claude/skills/{skill.name} is missing; run "
        f"`ln -sfn ../../.agents/skills/{skill.name} .claude/skills/{skill.name}`"
    )
    assert link.resolve() == skill.resolve()


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.name)
def test_readme_lists_the_skill(skill: Path) -> None:
    readme = (SKILLS_DIR / "README.md").read_text(encoding="utf-8")
    assert f"| `{skill.name}` |" in readme, (
        f"add a `{skill.name}` row to the table in .agents/skills/README.md"
    )
