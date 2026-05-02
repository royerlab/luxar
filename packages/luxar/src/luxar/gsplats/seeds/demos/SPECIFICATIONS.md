# luxar.gsplats.seeds.demos - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2026-05-02

## Purpose

`luxar.gsplats.seeds.demos` contains runnable demonstration scripts for comparing Gaussian-splat seed generation strategies on small example microscopy datasets.

---

## Core Concepts

### Demo Scripts

Demo scripts are executable examples, not library APIs. They should be deterministic when practical, small enough for local use, and clear about optional dependencies such as napari or scikit-image.

### Seed Method Comparison

The demos visualize seed locations from multiple seed generators so developers can qualitatively compare coverage, edge sensitivity, and redundancy before fitting.

---

## Data Structures

Demo scripts use standard NumPy arrays and seed arrays:

```text
volume: float[Z, Y, X] or float[Y, X]
seeds: float[N, D]
```

**Invariants**:
- Seed coordinates are in voxel coordinates.
- Displayed seed dimensionality must match the demo volume dimensionality.

---

## Algorithms

### Demo Execution

**Algorithm**:
1. Load or synthesize a small input volume.
2. Run one or more seed methods.
3. Present side-by-side visualizations or summaries.
4. Exit without writing persistent project artifacts unless explicitly requested.

---

## Validation Rules

- Missing optional dependencies should produce a clear error explaining how to install them.
- Demo data must be bounded in size to avoid accidental long-running examples.
- Scripts should not require CUDA; GPU acceleration may be used only opportunistically.

---

## Related Specifications

- `luxar.gsplats.seeds` - seed generation algorithms (`../SPECIFICATIONS.md`)
- `luxar.gsplats.fitting` - fitting pipeline that consumes seeds (`../../fitting/SPECIFICATIONS.md`)

---

## Changelog

- **v1.0.0** (2026-05-02): Initial specification.
