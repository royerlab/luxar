#### The pre-commit mypy hook and one device test were red on macOS

`os.sched_getaffinity` is Linux-only, and two places assumed otherwise. mypy
flagged `attr-defined` at `gsplats/utils/device.py:168` on darwin, failing the
whole `mypy (luxar package)` pre-commit hook for every macOS committer — the
call was already guarded at runtime, so the failure was purely a type-checking
artefact. The lookup now goes through `getattr`, matching the idiom the same
function already uses for `process_cpu_count` two lines above.

`test_effective_cpu_count_uses_affinity` had the mirror-image problem: it
patched `os.sched_getaffinity` without `raising=False`, so on macOS it errored
on the patch itself rather than testing anything. The affinity path had no
coverage on this platform, and neither did the fallback that macOS actually
takes. Both are covered now, along with a present-but-failing
`sched_getaffinity`.
