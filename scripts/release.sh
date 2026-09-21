#!/usr/bin/env bash
#
# release.sh — safe, robust release driver for Luxar.
#
# The publish pipeline is tag-triggered: pushing a `v<version>` tag fires
# .github/workflows/publish.yml, which builds the viewer + wheel on a clean
# Linux runner and publishes to PyPI via OIDC trusted publishing. This script
# does the SAFE local half: exhaustive preflight checks, then create + push the
# tag at the current (green) main commit. It NEVER pushes to main (main is
# branch-protected with enforce_admins), and it NEVER uploads to PyPI directly.
#
# Usage (normally via the Makefile):
#   bash scripts/release.sh --dry-run     # preflight only; mutate nothing  (make release-check)
#   bash scripts/release.sh               # preflight + tag + push tag      (make release)
#
# The release version is read from packages/luxar/src/luxar/__init__.py
# (__version__). Bump that via a normal PR FIRST (see `make set-version`), then
# promote it to main with CI green before running this to tag the release.
#
# Env overrides (all optional):
#   SKIP_CI_CHECK=1   skip the "main CI is green" gate (NOT recommended)
#   CONFIRM=<tag>     non-interactive confirmation (must equal the tag, e.g. v2026.06.29)
#   REMOTE=origin     git remote to push to (default: origin)
#
set -euo pipefail

# ---- pretty output -----------------------------------------------------------
if [[ -t 1 ]]; then
  RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLU=$'\033[36m'; BLD=$'\033[1m'; NC=$'\033[0m'
else
  RED=''; GRN=''; YLW=''; BLU=''; BLD=''; NC=''
fi
ok()   { echo "${GRN}✓${NC} $*"; }
info() { echo "${BLU}•${NC} $*"; }
warn() { echo "${YLW}⚠${NC}  $*"; }
die()  { echo "${RED}✗ $*${NC}" >&2; exit 1; }
step() { echo; echo "${BLD}$*${NC}"; }

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
REMOTE="${REMOTE:-origin}"
BRANCH="main"
VERSION_FILE="packages/luxar/src/luxar/__init__.py"
WORKFLOW=".github/workflows/publish.yml"
WORKFLOW_NPM=".github/workflows/publish-npm.yml"

# Always operate from the repo root (directory containing this script's parent).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "${BLD}Luxar release$([[ $DRY_RUN == 1 ]] && echo ' — DRY RUN')${NC}"
echo "repo: $REPO_ROOT"

# ---- 1. tools ----------------------------------------------------------------
step "1. Toolchain"
for t in git gh; do command -v "$t" >/dev/null 2>&1 || die "'$t' not found on PATH."; done
gh auth status >/dev/null 2>&1 || die "GitHub CLI is not authenticated — run 'gh auth login'."
ok "git, gh present; gh authenticated"

SLUG="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null)" || die "cannot resolve GitHub repo (gh repo view)."
info "repo slug: $SLUG"

# ---- 2. branch + clean tree --------------------------------------------------
step "2. Working state"
CUR_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "$CUR_BRANCH" == "$BRANCH" ]] || die "must release from '$BRANCH' (on '$CUR_BRANCH'). Check out main first."
ok "on branch $BRANCH"

[[ -z "$(git status --porcelain)" ]] || die "working tree is dirty. Commit/stash/clean it first (a release tags an exact, reviewed commit)."
ok "working tree clean"

[[ -f "$WORKFLOW" ]] || die "$WORKFLOW missing — the tag would trigger no publish workflow."
git ls-files --error-unmatch "$WORKFLOW" >/dev/null 2>&1 || die "$WORKFLOW is not committed — commit it (via PR) before releasing."
ok "PyPI publish workflow present and committed"

# A PyPI *pending* trusted publisher does NOT reserve the project name: `luxar`
# stays claimable by anyone until the first successful upload. Report the
# registry state so the operator knows whether this tag CLAIMS the name (404)
# or publishes into an EXISTING project (200 — verify it is ours). Never fatal:
# a network blip must not block a release, and a 200 is the steady state after
# the first release.
if command -v curl >/dev/null 2>&1; then
  PYPI_CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 https://pypi.org/pypi/luxar/json || echo "000")"
  case "$PYPI_CODE" in
    404) info "PyPI project 'luxar' does not exist yet — this release will claim the name (a pending publisher does not reserve it)." ;;
    200) info "PyPI project 'luxar' exists — verify https://pypi.org/project/luxar/ is ours before tagging." ;;
    *)   warn "could not query pypi.org (HTTP $PYPI_CODE); skipped the project-name check." ;;
  esac
fi

# The npm viewer publish is also tag-triggered on v*, but publish-npm.yml gates
# the actual `npm publish` on ENABLE_NPM_PUBLISH from the job's vars context.
# GitHub resolves that context environment -> repository -> organization and
# compares strings case-insensitively, so the preflight must do the same. The
# three states stay distinct: "off" and "could not check" are different answers.
if [[ -f "$WORKFLOW_NPM" ]] && git ls-files --error-unmatch "$WORKFLOW_NPM" >/dev/null 2>&1; then
  npm_variable_at() { # $1=API endpoint; sets NPM_SWITCH when found
    local result
    if ! result="$(gh api "$1" --paginate \
      --jq '.variables[] | select(.name == "ENABLE_NPM_PUBLISH") | "found\t\(.value)\tvalue-end"' \
      2>/dev/null)"; then
      return 2
    fi
    if [[ "$result" == found$'\t'*$'\t'value-end ]]; then
      result="${result#found$'\t'}"
      NPM_SWITCH="${result%$'\t'value-end}"
      return 0
    fi
    return 1
  }

  resolve_npm_switch() {
    local endpoints sources index status
    endpoints=(
      "repos/$SLUG/environments/npm/variables"
      "repos/$SLUG/actions/variables"
      "repos/$SLUG/actions/organization-variables"
    )
    sources=("npm environment" "repository" "organization")
    for ((index = 0; index < ${#endpoints[@]}; index++)); do
      if npm_variable_at "${endpoints[$index]}"; then
        NPM_SOURCE="${sources[$index]}"
        return 0
      else
        status=$?
      fi
      if [[ $status -eq 2 ]]; then
        NPM_LOOKUP_ERROR="${sources[$index]}"
        return 2
      fi
    done
    return 1
  }

  NPM_SWITCH=""
  NPM_SOURCE=""
  NPM_LOOKUP_ERROR=""
  if resolve_npm_switch; then
    NPM_SWITCH_NORMALIZED="$(printf '%s\tvalue-end' "$NPM_SWITCH" | tr '[:upper:]' '[:lower:]')"
    NPM_SWITCH_NORMALIZED="${NPM_SWITCH_NORMALIZED%$'\t'value-end}"
    if [[ "$NPM_SWITCH_NORMALIZED" == "true" ]]; then
      ok "npm workflow committed, ENABLE_NPM_PUBLISH=$NPM_SWITCH from $NPM_SOURCE — the tag WILL stage @luxar/viewer (manual 2FA approval publishes it)"
    else
      warn "npm workflow committed, but ENABLE_NPM_PUBLISH='$NPM_SWITCH' from $NPM_SOURCE (not 'true') — the tag will NOT publish to npm."
    fi
  else
    NPM_RESOLVE_STATUS=$?
    if [[ $NPM_RESOLVE_STATUS -eq 2 ]]; then
      warn "could not read $NPM_LOOKUP_ERROR variables (gh api) — cannot tell whether the tag will publish to npm."
    else
      warn "npm workflow committed, but ENABLE_NPM_PUBLISH is UNSET — the tag will build and pack"
      warn "  @luxar/viewer and then skip the publish."
      warn "  npm has no 'pending publisher', so the FIRST publish must be a manual,"
      warn "  token-authenticated 'npm publish'. Steps: publish-npm.yml header."
    fi
  fi
else
  warn "$WORKFLOW_NPM missing/uncommitted — the tag will NOT publish @luxar/viewer to npm."
fi

# ---- 3. sync with remote -----------------------------------------------------
step "3. Sync with $REMOTE/$BRANCH"
git fetch --quiet "$REMOTE" "$BRANCH" --tags || die "git fetch failed."
LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git rev-parse "$REMOTE/$BRANCH")"
if [[ "$LOCAL_SHA" != "$REMOTE_SHA" ]]; then
  die "local $BRANCH ($LOCAL_SHA) != $REMOTE/$BRANCH ($REMOTE_SHA). Pull/rebase so you tag exactly what's on the remote."
fi
ok "in sync at ${LOCAL_SHA:0:12}"

# ---- 4. version + tag --------------------------------------------------------
step "4. Version & tag"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$VERSION_FILE")"
[[ -n "$VERSION" ]] || die "could not read __version__ from $VERSION_FILE."
# CalVer: zero-padded YYYY.MM.DD
[[ "$VERSION" =~ ^[0-9]{4}\.[0-9]{2}\.[0-9]{2}$ ]] || \
  die "version '$VERSION' is not CalVer YYYY.MM.DD (zero-padded). Run 'make set-version' and promote it first."
TAG="v$VERSION"
ok "release version: $VERSION  →  tag: $TAG"

# Python, viewer, and citation metadata must describe the same release, or a
# published artifact would carry inconsistent version metadata.
if command -v python3 >/dev/null 2>&1; then
  # The tag is derived from VERSION just above, so --expect-tag is a restatement
  # here. Its independent check matters in the tag-triggered publish workflows.
  python3 scripts/check_version_consistency.py --expect-tag "$TAG" \
    || die "Release version mismatch. Run 'make set-version DATE=$VERSION' and promote it first."
  ok "Python, viewer, citation, and tag versions are consistent"
else
  warn "python3 not found; skipped release version-consistency check"
fi

TODAY="$(date +%Y.%m.%d)"
if [[ "$VERSION" != "$TODAY" ]]; then
  warn "version ($VERSION) is not today ($TODAY). CalVer normally = release day; proceeding because the tree may have been prepared earlier."
fi

# tag must not already exist locally or remotely
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  die "tag $TAG already exists locally. Versions are immutable on PyPI — bump the date and re-prep."
fi
if git ls-remote --exit-code --tags "$REMOTE" "refs/tags/$TAG" >/dev/null 2>&1; then
  die "tag $TAG already exists on $REMOTE. This version was already released."
fi
ok "tag $TAG is free (local + remote)"

# ---- 5. changelog state -----------------------------------------------------
step "5. Changelog state"
# Release prep is `make changelog` -> `make set-version` -> `make changelog-release`,
# and nothing verified that any of it had happened. The script could tag with
# every fragment still unfolded, shipping a CHANGELOG.md that does not mention
# the version being released. Both halves are checked because they fail
# independently: fragments can be folded without the release cut, and the cut
# can be made before a late fragment lands.
PENDING_FRAGMENTS=0
if [[ -d changelog.d ]]; then
  for _f in changelog.d/*.md; do
    [[ -e "$_f" ]] || continue
    [[ "$(basename "$_f")" == "README.md" ]] && continue
    PENDING_FRAGMENTS=$((PENDING_FRAGMENTS + 1))
  done
fi
if [[ $PENDING_FRAGMENTS -gt 0 ]]; then
  die "changelog.d/ still holds $PENDING_FRAGMENTS unfolded fragment(s).
  Run 'make changelog' to fold them into CHANGELOG.md, then 'make set-version DATE=$VERSION'
  and 'make changelog-release' to cut the section. Preview either with the -draft variants."
fi
ok "changelog.d/ is empty (all fragments folded)"

[[ -f CHANGELOG.md ]] || die "CHANGELOG.md is missing."
if ! grep -q "\[$VERSION\]\|## $VERSION" CHANGELOG.md; then
  die "CHANGELOG.md does not name version $VERSION.
  The fragments are folded but the release cut has not been made: run 'make changelog-release'
  (it names the cut after __version__, so run it AFTER 'make set-version')."
fi
ok "CHANGELOG.md names $VERSION"

# ---- 6. main CI is green -----------------------------------------------------
step "6. CI status on $REMOTE/$BRANCH"
if [[ "${SKIP_CI_CHECK:-0}" == "1" ]]; then
  warn "SKIP_CI_CHECK=1 — NOT verifying CI. You are tagging an unverified commit."
else
  # Pull the *required* contexts straight from branch protection, then assert
  # each is success on this exact SHA. Robust against added/renamed checks.
  #
  # Every read below is REQUIRED to succeed. This endpoint needs admin on the
  # repository, so a token without it 403s here as a matter of course — and the
  # old code swallowed that with `|| true`, left REQUIRED empty, and dropped to a
  # fallback that only inspected already-completed check-runs. Two silent API
  # failures (protection + check-runs, both plausible under one expired token)
  # therefore produced a PASSING gate that had examined nothing at all. "Could
  # not check" is not "nothing to check": the same three-state discipline
  # resolve_npm_switch already applies above.
  PROT_ERR="$(mktemp)"
  if ! PROTECTION_RAW="$(gh api "repos/$SLUG/branches/$BRANCH/protection/required_status_checks" \
                           --jq '.contexts[]?' 2>"$PROT_ERR")"; then
    _msg="$(head -3 "$PROT_ERR" | tr '\n' ' ')"; rm -f "$PROT_ERR"
    if [[ "$_msg" == *"HTTP 403"* ]]; then
      _hint="This endpoint requires admin on the repository; re-run with an admin token."
    elif [[ "$_msg" == *"HTTP 404"* ]]; then
      _hint="No classic branch protection was found for this branch; check repository rulesets."
    else
      _hint="Check the GitHub response and retry."
    fi
    die "cannot read branch protection for $BRANCH on $SLUG: ${_msg:-no error output}
  $_hint SKIP_CI_CHECK=1 bypasses the whole gate, but then you are tagging a
  commit nothing has verified."
  fi
  rm -f "$PROT_ERR"

  # macOS ships bash 3.2 (no `mapfile`), so read into the array manually.
  REQUIRED=()
  while IFS= read -r _line; do
    [[ -n "$_line" ]] && REQUIRED+=("$_line")
  done <<< "$PROTECTION_RAW"
  if [[ ${#REQUIRED[@]} -eq 0 ]]; then
    die "branch protection on $BRANCH lists no required status checks.
  There is nothing to verify, so a green result here would carry no information."
  fi
  ok "branch protection requires ${#REQUIRED[@]} context(s)"

  # --paginate, because the default page holds 30 check-runs and this repo
  # routinely exceeds that; a required check past the cut read as MISSING.
  # (`npm_variable_at` at the top of this file already paginates — §5 did not.)
  # --paginate + --jq runs the filter PER PAGE, so ask for a stream of objects
  # (JSON Lines) rather than an array: concatenated arrays are not valid JSON.
  CHECKS_ERR="$(mktemp)"
  if ! CHECKS_JSON="$(gh api "repos/$SLUG/commits/$REMOTE_SHA/check-runs" --paginate \
                        --jq '.check_runs[] | {name, conclusion, status}' 2>"$CHECKS_ERR")"; then
    _msg="$(head -3 "$CHECKS_ERR" | tr '\n' ' ')"; rm -f "$CHECKS_ERR"
    die "cannot read check-runs for ${REMOTE_SHA:0:12} on $SLUG: ${_msg:-no error output}
  Refusing to treat an unreadable CI state as a green one."
  fi
  rm -f "$CHECKS_ERR"
  STATUS_ERR="$(mktemp)"
  if ! STATUS_JSON="$(gh api "repos/$SLUG/commits/$REMOTE_SHA/status" --paginate \
                        --jq '.statuses[] | {context: .context, state: .state}' 2>"$STATUS_ERR")"; then
    _msg="$(head -3 "$STATUS_ERR" | tr '\n' ' ')"; rm -f "$STATUS_ERR"
    die "cannot read commit statuses for ${REMOTE_SHA:0:12} on $SLUG: ${_msg:-no error output}
  Refusing to treat an unreadable CI state as a green one."
  fi
  rm -f "$STATUS_ERR"

  # name -> conclusion for check-runs, falling back to context -> state for the
  # legacy commit-status API. A check that exists but has not finished reports
  # PENDING, which is not success and so fails the gate: a queued required check
  # means the commit is not verified yet, not that it passed.
  conclusion_of() { # $1=check name -> success|<conclusion>|PENDING|MISSING
    local name="$1" c
    c="$(printf '%s\n' "$CHECKS_JSON" | python3 -c '
import sys, json
want = sys.argv[1]
runs = [json.loads(l) for l in sys.stdin if l.strip()]
hit = [r for r in runs if r["name"] == want]
if not hit:
    print("MISSING")
elif any(r["status"] != "completed" for r in hit):
    print("PENDING")
elif any(r["conclusion"] != "success" for r in hit):
    print(next((r["conclusion"] or "PENDING" for r in hit if r["conclusion"] != "success"), "PENDING"))
else:
    print("success")
' "$name")" || c=MISSING
    if [[ "$c" == "MISSING" ]]; then
      c="$(printf '%s\n' "$STATUS_JSON" | python3 -c '
import sys, json
want = sys.argv[1]
sts = [json.loads(l) for l in sys.stdin if l.strip()]
hit = [x for x in sts if x["context"] == want]
print(hit[0]["state"] if hit else "MISSING")
' "$name")" || c=MISSING
    fi
    echo "$c"
  }

  FAILED=0
  for chk in "${REQUIRED[@]}"; do
    res="$(conclusion_of "$chk")"
    if [[ "$res" == "success" ]]; then ok "  $chk: success"
    else warn "  $chk: $res"; FAILED=1; fi
  done
  [[ $FAILED -eq 0 ]] || die "CI on $REMOTE/$BRANCH is not green for ${REMOTE_SHA:0:12}. Wait for it, fix it, or (last resort) SKIP_CI_CHECK=1."
  ok "all required checks green on ${REMOTE_SHA:0:12}"
fi

# ---- 7. summary + confirm + act ----------------------------------------------
step "7. Plan"
cat <<EOF
  Will create annotated tag ${BLD}$TAG${NC} at ${REMOTE_SHA:0:12} on $BRANCH
  and push it to $REMOTE, which triggers:
    $WORKFLOW  →  build viewer+wheel (Linux/OIDC)  →  publish ${BLD}luxar $VERSION${NC} to PyPI.
    $WORKFLOW_NPM  →  build lib bundle  →  STAGE ${BLD}@luxar/viewer${NC} on npm (if configured).
      Staged is NOT published: run ${BLD}npm stage approve <id>${NC} (2FA) to make it installable.
  Confirm the Apple-silicon native backend release verification ran for ${REMOTE_SHA:0:12}.
  This is the real, public, irreversible release (PyPI/npm versions cannot be reused).
EOF

if [[ $DRY_RUN == 1 ]]; then
  echo; ok "DRY RUN complete — all preflight checks passed. Nothing was tagged or pushed."
  echo "  Run ${BLD}make release${NC} to perform the release."
  exit 0
fi

# confirmation
if [[ -n "${CONFIRM:-}" ]]; then
  [[ "$CONFIRM" == "$TAG" ]] || die "CONFIRM='$CONFIRM' does not match tag '$TAG'. Aborting."
  info "confirmed non-interactively via CONFIRM"
else
  echo
  read -r -p "Type the tag '${BLD}$TAG${NC}' to publish, anything else to abort: " reply
  [[ "$reply" == "$TAG" ]] || die "aborted (got '$reply')."
fi

step "8. Tag & push"
git tag -a "$TAG" -m "Luxar $VERSION" "$LOCAL_SHA"
ok "created $TAG"
git push "$REMOTE" "refs/tags/$TAG"
ok "pushed $TAG to $REMOTE"

echo
ok "${BLD}Release $TAG initiated.${NC}"
echo "  Watch the publish run:  ${BLU}gh run watch --exit-status$NC  (or)  https://github.com/$SLUG/actions/workflows/publish.yml"
echo "  PyPI (after success):   ${BLU}https://pypi.org/project/luxar/$VERSION/$NC"
echo
echo "  If the publish job fails, the tag is already public. Fix forward with a NEW"
echo "  date version (PyPI never allows reusing a version), or delete the remote tag"
echo "  with: git push $REMOTE :refs/tags/$TAG"
