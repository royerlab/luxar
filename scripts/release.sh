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
# (__version__). Bump that via a normal PR FIRST (see `make set-version`), let it
# merge to main with CI green, then run this to tag the release.
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

# The npm viewer publish is also tag-triggered on v*. It's optional at launch
# (publishes only if the npm-side trusted publisher / NPM_TOKEN is configured),
# so a missing/uncommitted file is a warning, not a hard stop.
if [[ -f "$WORKFLOW_NPM" ]] && git ls-files --error-unmatch "$WORKFLOW_NPM" >/dev/null 2>&1; then
  ok "npm publish workflow present and committed"
else
  warn "$WORKFLOW_NPM missing/uncommitted — the tag will NOT publish @royerlab/luxar-viewer to npm."
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
  die "version '$VERSION' is not CalVer YYYY.MM.DD (zero-padded). Run 'make set-version' and merge it first."
TAG="v$VERSION"
ok "release version: $VERSION  →  tag: $TAG"

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

# ---- 5. main CI is green -----------------------------------------------------
step "5. CI status on $REMOTE/$BRANCH"
if [[ "${SKIP_CI_CHECK:-0}" == "1" ]]; then
  warn "SKIP_CI_CHECK=1 — NOT verifying CI. You are tagging an unverified commit."
else
  # Pull the *required* contexts straight from branch protection, then assert
  # each is success on this exact SHA. Robust against added/renamed checks.
  # NB: macOS ships bash 3.2 (no `mapfile`), so read into the array manually.
  REQUIRED=()
  while IFS= read -r _line; do
    [[ -n "$_line" ]] && REQUIRED+=("$_line")
  done < <(gh api "repos/$SLUG/branches/$BRANCH/protection/required_status_checks" \
             --jq '.contexts[]?' 2>/dev/null || true)
  if [[ ${#REQUIRED[@]} -eq 0 ]]; then
    warn "no required status checks found on branch protection — falling back to 'all check-runs must pass'."
  fi
  # name -> conclusion for check-runs, and context -> state for legacy statuses
  CHECKS_JSON="$(gh api "repos/$SLUG/commits/$REMOTE_SHA/check-runs" --jq '[.check_runs[] | {name, conclusion, status}]' 2>/dev/null || echo '[]')"
  STATUS_JSON="$(gh api "repos/$SLUG/commits/$REMOTE_SHA/status"     --jq '[.statuses[]  | {context: .context, state: .state}]' 2>/dev/null || echo '[]')"

  conclusion_of() { # $1=check name -> prints success|<other>|MISSING
    local name="$1" c
    c="$(echo "$CHECKS_JSON" | python3 -c "import sys,json; n=sys.argv[1]; d=json.load(sys.stdin); m=[x for x in d if x['name']==n]; print(m[0]['conclusion'] if m and m[0]['status']=='completed' else ('PENDING' if m else 'MISSING'))" "$name" 2>/dev/null || echo MISSING)"
    if [[ "$c" == "MISSING" ]]; then
      c="$(echo "$STATUS_JSON" | python3 -c "import sys,json; n=sys.argv[1]; d=json.load(sys.stdin); m=[x for x in d if x['context']==n]; print(m[0]['state'] if m else 'MISSING')" "$name" 2>/dev/null || echo MISSING)"
    fi
    echo "$c"
  }

  FAILED=0
  if [[ ${#REQUIRED[@]} -gt 0 ]]; then
    for chk in "${REQUIRED[@]}"; do
      res="$(conclusion_of "$chk")"
      if [[ "$res" == "success" ]]; then ok "  $chk: success"
      else warn "  $chk: $res"; FAILED=1; fi
    done
  else
    # fallback: any non-success completed check fails the gate
    bad="$(echo "$CHECKS_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print('\n'.join(f\"{x['name']}:{x['conclusion']}\" for x in d if x['status']=='completed' and x['conclusion'] not in ('success','neutral','skipped')))" 2>/dev/null || true)"
    [[ -n "$bad" ]] && { warn "failing checks: $bad"; FAILED=1; }
  fi
  [[ $FAILED -eq 0 ]] || die "CI on $REMOTE/$BRANCH is not green for ${REMOTE_SHA:0:12}. Wait for it, fix it, or (last resort) SKIP_CI_CHECK=1."
  ok "all required checks green on ${REMOTE_SHA:0:12}"
fi

# ---- 6. summary + confirm + act ----------------------------------------------
step "6. Plan"
cat <<EOF
  Will create annotated tag ${BLD}$TAG${NC} at ${REMOTE_SHA:0:12} on $BRANCH
  and push it to $REMOTE, which triggers:
    $WORKFLOW  →  build viewer+wheel (Linux/OIDC)  →  publish ${BLD}luxar $VERSION${NC} to PyPI.
    $WORKFLOW_NPM  →  build lib bundle  →  publish ${BLD}@royerlab/luxar-viewer${NC} to npm (if configured).
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

step "7. Tag & push"
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
