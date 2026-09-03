# Security Policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Use GitHub's [private vulnerability reporting](https://github.com/royerlab/luxar/security/advisories/new)
on this repository. It creates a private thread visible only to the maintainers,
and it is the preferred route because it keeps the report, the fix and the
advisory in one place.

If that is unavailable to you, email **loic.royer@czbiohub.org** with `luxar
security` in the subject line.

Please include what you need to make the problem reproducible: affected version
or commit, platform, and the smallest input or sequence that triggers it. A
proof of concept helps; a working exploit is not required and we would rather
you did not publish one.

You should get an acknowledgement within a few working days. Luxar is maintained
by a small research group, so please allow reasonable time for a fix before any
public disclosure, and tell us if you have a disclosure deadline — we would
rather plan around it than be surprised by it.

## What is in scope

Luxar compiles scientific data into Zarr archives and renders them in a browser.
The parts where a security issue is most plausible:

- **Reading untrusted data.** The Python compiler and the viewer both parse
  archives that a user may have obtained from someone else. Anything that turns
  a malformed or hostile archive into code execution, a path traversal, or a
  write outside the intended directory is in scope. Note the viewer treats a
  scene URL as data: a scene should never be able to escape the page.
- **The demo data fetcher.** `luxar.demos` downloads datasets over HTTPS and
  verifies them against sha256 pins in a committed manifest. A way to make it
  accept unpinned or mismatched bytes, or to write outside the cache directory,
  is in scope.
- **`luxar serve` and `luxar export`.** These serve files over HTTP. Path
  traversal or serving outside the intended root is in scope.
- **The published artifacts.** Anything that could let a third party publish to
  `luxar` on PyPI or `@royerlab/luxar-viewer` on npm, or tamper with a release,
  is very much in scope.

## What is not

- Denial of service caused by simply pointing Luxar at a very large dataset.
  Rendering is bounded by hardware by design; "a 100 GB volume made my browser
  slow" is a performance report, and welcome as an ordinary issue.
- Findings from an automated scanner with no demonstrated impact on Luxar.
- Vulnerabilities in a dependency that do not affect how Luxar uses it. Report
  those upstream; if Luxar's usage *is* affected, we want to hear about it.

## Supported versions

Luxar is pre-1.0 and released as CalVer. Only the most recent release is
supported: fixes land on `main` and go out in the next release rather than
being backported.
