#### Bake the element-texture width as a compile-time shader constant

The Points / Lines / GSplats vertex stages (visual and picking, both
backends) indexed their per-node element texture with `base % W` /
`base / W`, reading `W` per vertex via `textureSize()`. The width is a
per-layout session constant (capped at 4096 on every device), so it is
now baked at material construction — per-layout defines
(`LUXAR_LINE_TEX_W` / `LUXAR_POINT_TEX_W` / `LUXAR_SPLAT_TEX_W`) on the
GLSL materials, a literal int node in the TSL graphs. A compile-time
constant lets the shader compiler strength-reduce the per-vertex integer
division: measured −7% on the quad line primitive's whole GPU pass at
4 M segments (RTX 3070, WebGPU timestamp queries), where a uniform
recovered almost none of it. A unit suite pins every material's define
to the width the texture writers allocate at, and pins the shader
sources free of `textureSize` so the query cannot quietly return.
