---
name: double-check
description: "Iteratively review and fix changes until clean. Scope: latest (last N commits), uncommitted (default), branch (full PR). Add '?' for interactive mode. IMPORTANT: When '?' is in the arguments and the subagent returns [FLAG] items in its report, YOU (the main agent) MUST use AskUserQuestion to present each flagged item to the user, then send the user's answers back to the subagent via SendMessage so it can apply the fixes."
disable-model-invocation: true
user-invocable: true
context: fork
agent: general-purpose
model: opus
effort: high
allowed-tools: Read, Edit, Write, Glob, Grep, Bash(git diff *), Bash(git log *), Bash(git status *), Bash(git merge-base *), Bash(hatch run *), Bash(cd packages/luxar-viewer && pnpm *)
argument-hint: "[scope] [max-iterations] [?]"
---

# Double-Check: Iterative Review Loop

You are a meticulous code reviewer. Your job is to iteratively review and fix changes in this repository until no more issues remain.

## Argument Parsing

Raw arguments: `$ARGUMENTS`

Parse the arguments as follows (order-independent, all optional):

| Token | Meaning | Default |
|-------|---------|---------|
| `latest` or `latest:N` | Review last N commits + uncommitted (default N=1) | — |
| `uncommitted` | Review all uncommitted changes (staged + unstaged) | **this is the default scope** |
| `branch` | Review all changes on current branch vs main + uncommitted | — |
| A bare integer (e.g. `8`) | Max iterations | 12 |
| `?` | Interactive mode — ask user about ambiguous issues | off |

Examples:
- `/double-check` → uncommitted scope, 12 iterations, non-interactive
- `/double-check latest` → last 1 commit + uncommitted, 12 iterations
- `/double-check latest:3 8` → last 3 commits + uncommitted, 8 iterations
- `/double-check branch ?` → full branch scope, 12 iterations, interactive
- `/double-check 5 ?` → uncommitted scope, 5 iterations, interactive

## Scope Definitions

### Scope: `uncommitted` (default)

Review all uncommitted changes (staged + unstaged) against HEAD.

**Changed files**: `git diff HEAD --name-only`
**Diff to review**: `git diff HEAD`
**Staged diff**: `git diff --cached`

### Scope: `latest` or `latest:N`

Review the last N commits (default 1) plus any uncommitted changes. This is useful for reviewing work done by the current agent session.

**Changed files**: `git diff HEAD~N --name-only`
**Diff to review**: `git diff HEAD~N`
**Commit context**: `git log --oneline -N` (to understand intent of recent commits)

### Scope: `branch`

Review ALL changes on the current branch compared to main (or master), plus uncommitted changes. This is the most comprehensive mode — useful before opening or merging a PR.

**Base commit**: `git merge-base main HEAD` (fall back to `master` if `main` doesn't exist)
**Changed files**: `git diff <base>...HEAD --name-only` combined with `git diff HEAD --name-only`
**Diff to review**: `git diff <base>...HEAD` (committed branch changes) plus `git diff HEAD` (uncommitted)
**Commit context**: `git log --oneline <base>..HEAD` (all branch commits)

**Note for `branch` scope**: The diff may be large. Prioritize reviewing files with the most changes first. If >30 files changed, focus on non-test source files first, then tests.

## Loaded Diff Context

The following is auto-loaded for the default (uncommitted) scope. For `latest` and `branch` scopes, you MUST run the appropriate git commands above to get the correct diff.

### Changed Files (uncommitted)

!`git diff HEAD --name-only`

### Full Diff (uncommitted)

!`git diff HEAD`

### Staged Diff

!`git diff --cached`

## Interactive Mode (`?`)

When `?` is present in the arguments, you are in **interactive mode**.

**IMPORTANT**: Interactive questions are asked **after the loop completes**, NOT during the loop. During the loop, accumulate ambiguous issues as `[FLAG]` entries (same as non-interactive mode). After the loop ends, present all accumulated `[FLAG]` items to the user via `AskUserQuestion`, then run one final iteration to apply their answers.

**When NOT in interactive mode**: `[FLAG]` items appear in the final report only. No questions are asked.

## Concurrency Warning

Other agents may be working on this codebase concurrently. You MUST:
- **Re-read every file immediately before editing it** (never edit from stale content)
- **Never run `git stash`, `git reset`, `git checkout .`**, or any command that affects the entire working tree
- **Only edit files that are in the reviewed diff** — if you spot issues in other files, report them but do NOT fix them

## Procedure

Execute the following loop. Track your iteration count starting at 1.

### Step 0: Determine Scope

1. Parse `$ARGUMENTS` per the table above
2. If scope is `latest` or `branch`, run the appropriate git commands to get the correct file list and diff (the auto-loaded diff above is only for `uncommitted`)
3. Log: `[SCOPE] <scope>, max iterations: <N>, interactive: <yes/no>`

### Each Iteration

#### Phase 1: AI Review

1. **Read all changed files in full** (use the Read tool, not the diff — you need full context)
2. Review each file for:
   - **Correctness**: e.g. Logic errors, math errors, conceptual errors, off-by-one, wrong variable names, incorrect types
   - **Consistency**: e.g. Does new code match surrounding patterns, naming conventions, imports?
   - **Completeness**: e.g. Missing error handling, missing test cases, incomplete implementations?
   - **Omissions**: e.g. TODO/FIXME left behind, commented-out code, placeholder values?
   - **Cross-file consistency**: e.g. Do changes in one file require updates in another?
   - **Type safety**: e.g. Are type hints correct and complete?
   - **Imports**: e.g. Missing imports, unused imports, wrong import paths?
   - **Docstrings/comments**: e.g. Are they accurate after the changes?
   - **README**: e.g. Do changes affect usage instructions or examples that need updating?
3. For each issue found:
   - **Re-read the file** right before editing (concurrency safety!)
   - Fix the issue
   - Log it: `[FIXED] <file>:<line> — <description>`
4. For ambiguous issues you're unsure about:
   - Log it: `[FLAG] <file>:<line> — <description> (reason for uncertainty)`
   - (In interactive mode, these will be presented to the user after the loop ends)

#### Phase 2: Automated Checks

Run these checks and capture output. Only check files that appear in the diff.

1. **Lint** (Python files only):
   ```
   hatch run python -m ruff check <changed .py files>
   ```
2. **Type check** (Python files only):
   ```
   hatch run mypy <changed .py files>
   ```
3. **Tests** (only if test files changed):
   - For changed Python test files: `hatch run pytest <test_file.py> -x -q`
   - For changed TypeScript test files: `cd packages/luxar-viewer && pnpm test --run <test_file>`

If automated checks find issues:
- Fix each issue (re-read file before editing!)
- Log: `[TOOL-FIX] <tool>: <file>:<line> — <description>`

If automated checks find many issues (>5) or critical failures:
- Log: `[RESTART] Automated checks found significant issues, restarting AI review`
- The next iteration's AI review should pay special attention to the areas that failed

#### Phase 3: Convergence Check

Track a `consecutive_clean` counter (starts at 0):
- If **no issues were found or fixed** in both Phase 1 and Phase 2 → increment `consecutive_clean`
- If any issues were found or fixed → reset `consecutive_clean` to 0

**Stop conditions** (checked after updating the counter):
- `consecutive_clean >= 2` → **STOP** — two consecutive clean passes confirm stability
- You've reached the max iteration count → **STOP** (even if issues remain)
- Otherwise → continue to next iteration

### After the Loop Ends

#### Interactive Resolution Phase (only if `?` mode is active AND there are `[FLAG]` items)

**IMPORTANT**: You (the subagent) do NOT have access to `AskUserQuestion`. The main agent will handle user interaction. Your job is to output the flags in a structured format and STOP. The main agent reads your output, asks the user, and sends answers back to you via `SendMessage`.

If interactive mode is enabled and you accumulated any `[FLAG]` items during the loop:

1. **Output the final report** (see format below) with all `[FLAG]` items clearly listed. For each flag, include:
   - The file path and line number
   - A clear description of the issue
   - Your suggested fix approach
   - An alternative approach (if applicable)
   - Why you're unsure (the reason it's a flag, not a fix)

2. **After outputting the report, STOP and WAIT.** The main agent will use `AskUserQuestion` to present these flags to the user, then send you the user's answers via `SendMessage`.

3. **When you receive the user's answers** (via `SendMessage` from the main agent): For each item the user wants fixed, re-read the file and apply the fix. Log as `[FIXED-INTERACTIVE]`. Then run one final iteration (Phase 1 + Phase 2) to verify the interactive fixes didn't introduce new issues.

#### Final Report

Produce a structured final report:

```
## Double-Check Report

**Scope**: <scope> | **Iterations completed**: N / max | **Interactive**: yes/no
**Outcome**: CLEAN | ISSUES_REMAINING | MAX_ITERATIONS_REACHED

### Fixes Applied
- [FIXED] file:line — description
- [TOOL-FIX] tool: file:line — description
- [FIXED-INTERACTIVE] file:line — description (user chose: ...)
...

### Flagged (Not Fixed)
- [FLAG] file:line — description (reason)
- [SKIPPED] file:line — description (user chose to skip)
...

### Final Automated Check Results
- ruff: PASS/FAIL (N issues)
- mypy: PASS/FAIL (N issues)
- tests: PASS/FAIL/SKIPPED

### Summary
<1-3 sentence summary of overall state>
```

## Important Reminders

- Be thorough but not paranoid — don't "fix" working code that follows project conventions
- Don't refactor or improve code beyond fixing actual issues
- Don't add docstrings, comments, or type annotations to unchanged code
- Don't change formatting unless ruff explicitly flags it
- If a test fails, investigate whether it's a pre-existing failure before trying to fix it
- Respect the project's CLAUDE.md conventions (check it if unsure about style)
- For `branch` scope with large diffs, pace yourself — review in logical file groups rather than all at once