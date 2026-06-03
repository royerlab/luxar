# Shaders Barrel — Stable GLSL3 Re-Export Path

> Barrel-only folder. Re-exports the GLSL3 vertex/fragment string constants for the three visual material stacks (Point, Line, GSplat) from their canonical homes under `../materials/<kind>/shader-glsl.ts`.

## Why this folder exists

The GLSL3 sources have moved to live next to each geometry's material wrapper
(`../materials/point/shader-glsl.ts`, `../materials/line/shader-glsl.ts`,
`../materials/gsplat/shader-glsl.ts`). This barrel preserves the historical
import path `rendering/shaders` so external harnesses can keep importing
shader strings under one stable spelling regardless of future re-orgs of the
materials tree.

Per the project's "Keep GLSL shaders as reference" rule (see `CLAUDE.md` /
project memory), the GLSL3 strings are never deleted and the parity check
against the TSL `NodeMaterial` counterparts runs on every PR.

## Public API

`index.ts` re-exports six constants:

| Symbol                   | Source                               |
| ------------------------ | ------------------------------------ |
| `POINT_VERTEX_SHADER`    | `../materials/point/shader-glsl.ts`  |
| `POINT_FRAGMENT_SHADER`  | `../materials/point/shader-glsl.ts`  |
| `LINE_VERTEX_SHADER`     | `../materials/line/shader-glsl.ts`   |
| `LINE_FRAGMENT_SHADER`   | `../materials/line/shader-glsl.ts`   |
| `GSPLAT_VERTEX_SHADER`   | `../materials/gsplat/shader-glsl.ts` |
| `GSPLAT_FRAGMENT_SHADER` | `../materials/gsplat/shader-glsl.ts` |

These are plain GLSL3 string constants — the same strings that
`PointMaterial`, `LineMaterial`, and `GSplatMaterial` feed into
`THREE.ShaderMaterial` under the WebGL2 backend.

## Where the shader strings are consumed

The canonical consumers import directly from `../materials/<kind>/shader-glsl.ts`
(not from this barrel):

- `../materials/point/material-glsl.ts` — passes `POINT_VERTEX_SHADER` /
  `POINT_FRAGMENT_SHADER` to `THREE.ShaderMaterial`.
- `../materials/line/material-glsl.ts` — same for the line pair.
- `../materials/gsplat/material-glsl.ts` — same for the gsplat pair.

This barrel's re-exports exist as a stable spelling for out-of-tree consumers
and regression harnesses (e.g. `src/tests/unit/rendering/shader-hot-path.test.ts`,
`src/tests/e2e/tsl-shader-parity.spec.ts`) that compare these GLSL3 sources
against their TSL `NodeMaterial` counterparts. Removing or renaming the
barrel would force those harnesses to track the materials tree's internal
layout.

Picking shaders are not re-exported here — they live under
`../picking/<kind>/shaders.ts` and are not part of the visual-material parity
surface.

## See Also

- `../README.md` — Rendering package overview (file layout, parity policy)
- `../materials/_shared/README.md` — `ShaderSource` registry and the GLSL↔TSL
  parity pattern this barrel feeds into
- `../materials/point/shader-glsl.ts`, `../materials/line/shader-glsl.ts`,
  `../materials/gsplat/shader-glsl.ts` — canonical homes of the six strings
