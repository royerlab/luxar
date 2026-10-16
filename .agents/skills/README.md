# Luxar Agent Skills

This directory holds **Agent Skills** for Luxar — reusable, model-invoked
instructions that teach an AI coding agent how to drive Luxar's CLI and APIs.

`SKILL.md` is a cross-tool open standard: the same skill works in **Claude Code,
OpenAI Codex, Gemini CLI, Cursor**, and other compatible agents.

## Layout

```
.agents/skills/<skill-name>/
├── SKILL.md          # required: YAML frontmatter (name, description) + instructions
├── references/       # optional: detailed docs loaded on demand
├── scripts/          # optional: executable helpers
└── assets/           # optional: templates / resources
```

`.agents/skills/` is the vendor-neutral location (scanned by Codex up the
directory tree; also where packages like FastAPI/Typer ship skills). Claude Code
reads `.claude/skills/<name>/`, so each skill here is symlinked from there:

```bash
ln -sfn ../../.agents/skills/<skill-name> .claude/skills/<skill-name>
```

## How users get these skills

- **Clone the repo** — agents auto-discover skills in `.agents/skills/`
  (Codex) and `.claude/skills/` (Claude Code). No install step.
- **Invoke** — Claude Code: `/<skill-name>` or automatic; Codex: `$<skill-name>`,
  `/skills`, or automatic. Implicit invocation is driven by the `description`
  field, so keep it specific about *what* the skill does and *when* to use it.

## Available skills

| Skill | Purpose |
| --- | --- |
| `luxar-install` | Install Luxar on a machine — `pip install` into a venv first (Python 3.12 per OS, extras, CPU-only torch, runtime Metal, NVRTC CUDA and numba CPU kernels), then the source-checkout dev route: the optional ahead-of-time CUDA extension, Slurm/HPC cluster (no-sudo path, `SLURM=1`). |
| `luxar-visualization` | Build a `.luxar.zarr` scene from a dataset (Points/Lines/GSplats, Dimensions, transforms) and view it. |
| `luxar-gsplat-pipeline` | Calibrate, fit, and build LOD for Gaussian splats (`luxar gsplat` cal -> fit -> lod, tiling), full CLI options + Python API. |
| `luxar-hpc-batch-fit` | Fit a whole nD timelapse at scale — local multi-GPU (`batch-fit run`) or Slurm/Bruno (`batch-fit submit`); status/validate/merge/cancel + GPU benchmark. |
| `luxar-gsplat-edit` | Post-fit toolbox on a `.gsplats.zarr`: slice, transform, cull, filter, partition, merge, convert, migrate-format, info/render/compare/view/napari. |
| `luxar-data-loading` | Load an nD image/volume (`.zarr`/OME-Zarr/`.tiff`/`.npy`/`.npz`); channel/timepoint/array-key selection and `--axes` overrides. |
| `luxar-export` | Package a scene for sharing: standalone offline folder (viewer + data + `serve.py`) or native macOS/Linux app bundles. |

## Authoring a new skill

1. Create `.agents/skills/<name>/SKILL.md` with `name` + `description` frontmatter.
2. Keep `SKILL.md` under 500 lines; push measured cases and lore into
   `references/*.md` and link each one from `SKILL.md` (or a sibling reference).
   `packages/luxar/src/luxar/tests/test_agent_skills.py` enforces the line budget,
   the links, the frontmatter, the symlink, and the table row (step 4).
3. Symlink it into `.claude/skills/` (see above).
4. Add a row to the table above.
