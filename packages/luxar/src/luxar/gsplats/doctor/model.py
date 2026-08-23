"""The vocabulary ``gsplat doctor`` reports in: findings, fixes, and a report.

Kept apart from the checks themselves so a new check only has to import this and
land in the registry — the runner, the CLI rendering and the JSON shape all key
off these types and need no edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    import zarr

__all__ = ["Finding", "DoctorReport", "Severity", "SEVERITIES"]

#: How much a finding matters. ``error`` = the store is wrong or renders wrong;
#: ``warning`` = degraded but valid; ``note`` = worth knowing.
Severity = str
SEVERITIES: tuple = ("error", "warning", "note")


@dataclass
class Finding:
    """One diagnosed condition, and — when it can be repaired — how.

    ``fix`` is a closure over whatever the check already computed, so a repair
    never re-derives its own diagnosis (and so cannot disagree with the finding
    the user was shown). It mutates the open store and is called only after the
    user asks for it; ``None`` means the condition is real but not something the
    doctor can safely repair, in which case ``remedy`` must say what will.
    """

    #: Stable identifier of the check that produced this, e.g. ``split-planes``.
    check: str
    severity: Severity
    #: Store-relative path of the offending node; ``""`` for the root.
    path: str
    #: One line, in the imperative present: what IS the case.
    summary: str
    #: Why it matters — the consequence a reader would otherwise have to guess.
    detail: str = ""
    #: What repairs it, whether or not the doctor can do it.
    remedy: str = ""
    fix: Optional[Callable[[], None]] = None
    #: Set by the runner once a fix has actually run.
    fixed: bool = False

    @property
    def fixable(self) -> bool:
        return self.fix is not None

    def as_dict(self) -> Dict[str, Any]:
        """JSON-safe view (drops the closure)."""
        return {
            "check": self.check,
            "severity": self.severity,
            "path": self.path,
            "summary": self.summary,
            "detail": self.detail,
            "remedy": self.remedy,
            "fixable": self.fixable,
            "fixed": self.fixed,
        }


@dataclass
class DoctorReport:
    """Everything one ``doctor`` run diagnosed, and what it repaired."""

    path: str
    #: True when the run was allowed to write.
    fix: bool = False
    findings: List[Finding] = field(default_factory=list)
    #: Checks that ran, in order — so a clean bill of health can say what it
    #: actually looked at rather than just "no problems found".
    checks_run: List[str] = field(default_factory=list)
    #: What a FRESH run of the same checks reports after the repairs; ``None``
    #: when nothing was repaired. A repair can leave a lesser condition standing
    #: — removing a misleading tree still leaves parts with no exact order — so
    #: "is the store healthy now?" has to be answered by re-reading the store,
    #: not by the fix closures having been called.
    residual: Optional[List[Finding]] = None

    @property
    def problems(self) -> List[Finding]:
        """Findings that are not merely informational."""
        return [f for f in self.findings if f.severity != "note"]

    @property
    def unresolved(self) -> List[Finding]:
        """Problems still standing — measured against the store as it now is."""
        if self.residual is not None:
            return [f for f in self.residual if f.severity != "note"]
        return [f for f in self.problems if not f.fixed]

    @property
    def healthy(self) -> bool:
        return not self.unresolved

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "fix": self.fix,
            "healthy": self.healthy,
            "checks_run": list(self.checks_run),
            "findings": [f.as_dict() for f in self.findings],
            "residual": (
                None if self.residual is None else [f.as_dict() for f in self.residual]
            ),
        }


#: A check: given the opened root group, yield what it finds. Checks must not
#: write — they attach a ``fix`` closure and the runner decides whether to run it.
Check = Callable[["zarr.Group"], List[Finding]]
