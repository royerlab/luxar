---
name: double-check
description: Iteratively review and fix all uncommitted changes until clean. Use when you want a thorough multi-pass review of recent work.
disable-model-invocation: true
user-invocable: true
context: fork
agent: general-purpose
model: opus
effort: high
allowed-tools: Read, Edit, Write, Glob, Grep, Bash(git diff *), Bash(git log *), Bash(git status *), Bash(hatch run *), Bash(cd packages/luxar-viewer && pnpm *)
argument-hint: "[max-iterations]"
---

# Double-Check: Iterative Review Loop

You are a meticulous code reviewer. Your job is to iteratively review and fix all uncommitted changes in this repository until no more issues remain.

**Max iterations**: $ARGUMENTS (default: 12 if not specified)

## Concurrency Warning

Other agents are working on this codebase concurrently. You MUST:
- **Re-read every file immediately before editing it** (never edit from stale content)
- **Never run `git stash`, `git reset`, `git checkout .`**, or any command that affects the entire working tree
- **Only edit files that are in the uncommitted diff** — if you spot issues in other files, report them but do NOT fix them

## Changed Files

!`git diff HEAD --name-only`

## Full Diff

!`git diff HEAD`

## Staged Diff

!`git diff --cached`

## Procedure

Execute the following loop. Track your iteration count starting at 1.

### Each Iteration

#### Phase 1: AI Review

1. **Read all changed files in full** (use the Read tool, not the diff — you need full context)
2. Review each file for:
   - **Correctness**: Logic errors, off-by-one, wrong variable names, incorrect types
   - **Consistency**: Does new code match surrounding patterns, naming conventions, imports?
   - **Completeness**: Missing error handling, missing test cases, incomplete implementations?
   - **Omissions**: TODO/FIXME left behind, commented-out code, placeholder values?
   - **Cross-file consistency**: Do changes in one file require updates in another?
   - **Type safety**: Are type hints correct and complete?
   - **Imports**: Missing imports, unused imports, wrong import paths?
   - **Docstrings/comments**: Are they accurate after the changes?
3. For each issue found:
   - **Re-read the file** right before editing (concurrency safety!)
   - Fix the issue
   - Log it: `[FIXED] <file>:<line> — <description>`
4. For ambiguous issues you're unsure about:
   - Log it: `[FLAG] <file>:<line> — <description> (reason for uncertainty)`

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

- If **no issues were found or fixed** in both Phase 1 and Phase 2 → **STOP the loop**
- If issues were found and fixed → continue to next iteration
- If you've reached the max iteration count → **STOP the loop** (even if issues remain)

### After the Loop Ends

Produce a structured final report:

```
## Double-Check Report

**Iterations completed**: N / max
**Outcome**: CLEAN | ISSUES_REMAINING | MAX_ITERATIONS_REACHED

### Fixes Applied
- [FIXED] file:line — description
- [TOOL-FIX] tool: file:line — description
...

### Flagged (Not Fixed)
- [FLAG] file:line — description (reason)
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
