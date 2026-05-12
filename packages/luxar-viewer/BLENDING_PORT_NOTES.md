# Custom blending modes — port reference for `NodeMaterial`

Catalogue of every blending state Luxar uses today, with the
equivalent under `NodeMaterial`. Source-of-truth for the actual
state shapes is `src/rendering/blending-state.ts:getCompleteBlendingState()`;
materials apply via `applyBlendingStateToMaterial` (same file).
Per-geometry overrides live in
`{point,line,gsplat}-material.ts::applyBlendingMode`.

This file is a port-time checklist, not a behaviour spec.

## State shape glossary

`THREE.Material` exposes a fixed set of blending properties that
both `ShaderMaterial` and `NodeMaterial` inherit:

- `material.blending: THREE.Blending` (`NoBlending` | `NormalBlending`
  | `AdditiveBlending` | `SubtractiveBlending` | `MultiplyBlending`
  | `CustomBlending`).
- `material.blendEquation: THREE.BlendingEquation` (`AddEquation` |
  `SubtractEquation` | `ReverseSubtractEquation` | `MinEquation` |
  `MaxEquation`).
- `material.blendSrc`, `material.blendDst`: `THREE.BlendingDstFactor`
  /`BlendingSrcFactor` (`OneFactor`, `ZeroFactor`, `SrcAlphaFactor`,
  `OneMinusSrcAlphaFactor`, etc.).
- `material.blendEquationAlpha`, `material.blendSrcAlpha`,
  `material.blendDstAlpha`: per-alpha overrides; `null` means
  "track the RGB channels."

These properties live on `THREE.Material` (the common base), so
**NodeMaterial inherits them unchanged**. The blending state-machine
on the renderer side is identical between `WebGLRenderer` and
`WebGPURenderer`; only the *shader output* the state composites
changes. Each section below records the state and any shader-side
concern.

## Mode 1 — `additive` (gsplat linear sum)

**State** (`blending-state.ts:additive` branch and `gsplat-material.ts:441-460`):

```
blending          = CustomBlending
blendEquation     = AddEquation
blendSrc          = OneFactor
blendDst          = OneFactor
depthTest         = false
depthWrite        = false
transparent       = true
```

**Why CustomBlending instead of `AdditiveBlending`**: gsplats compute
their contribution as a *linear sum* of Gaussian samples. The standard
`THREE.AdditiveBlending` uses `SrcAlphaFactor + OneFactor`, which
multiplies source colour by alpha at composite time — but the gsplat
shader has already pre-weighted RGB by amplitude. Multiplying again
squares the alpha and dims the result. `OneFactor + OneFactor` adds
the shader's output verbatim, which is what we want for the
mathematical sum projection.

**Shader-output contract**: `gsplat-shaders.ts` outputs
`fragColor = vec4(weighted_rgb, alpha)` where `weighted_rgb` is
already pre-multiplied. TSL port emits the same node-graph output;
the renderer state above is unchanged.

**NodeMaterial parity**: identical state, properties live on
`Material` base class. No port change needed in this section.

## Mode 2 — `max` (point / line / gsplat max projection)

**State** (`blending-state.ts:max` branch):

```
blending          = CustomBlending
blendEquation     = MaxEquation
blendSrc          = OneFactor
blendDst          = OneFactor
depthTest         = true
depthWrite        = false
transparent       = true
```

**Per-channel alpha override (gsplat only)** — `gsplat-material.ts:444-460`:

```
blendEquationAlpha = MaxEquation
blendSrcAlpha      = OneFactor
blendDstAlpha      = OneFactor
```

The alpha override prevents max-mode alpha from accumulating
additively when the RGB channels max. Without it the alpha climbs
unboundedly while RGB plateaus, producing visible "ghost" frames
on bright pixels.

**Shader-output contract**: `LUXAR_MAX_RGB_CONTRIBUTION` shader
define switches the fragment to emit `rgb * alpha` (pre-multiplied
contribution) so the framebuffer max captures contribution-weighted
colour, not flat full-bright. See `point-shaders.ts:131-148`,
`line-shaders.ts` similar. The TSL port must add an equivalent
conditional branch in the node graph — likely
`If(USE_MAX_RGB_CONTRIBUTION, vec3(color) * alpha, vec3(color))`.

**NodeMaterial parity**: state identical. Per-channel-alpha
properties are also on the `Material` base class — verified in the
Three.js source. The shader-side define becomes a TSL conditional.

## Mode 3 — `normal` (standard alpha)

**State** (`blending-state.ts:normal` branch):

```
blending          = NormalBlending
blendEquation     = AddEquation        // Three.js sets these automatically
blendSrc          = SrcAlphaFactor     // when blending = NormalBlending,
blendDst          = OneMinusSrcAlphaFactor  // but listed for completeness
depthTest         = true
depthWrite        = opacity >= 0.99   // opaque variant writes depth
transparent       = true
```

**NodeMaterial parity**: identical. `NormalBlending` is a preset on
the base `Material`.

## Mode 4 — `additive` / `luminous` (point and line additive)

**State** (`blending-state.ts:additive` and `luminous` branches):

```
blending          = AdditiveBlending
depthTest         = false (additive) | true (luminous)
depthWrite        = false
transparent       = true
```

`AdditiveBlending` is a preset whose underlying state is
`AddEquation + SrcAlphaFactor + OneFactor`. The shader emits unweighted
RGB plus `alpha = intensity * opacity`, and the framebuffer multiplies
by alpha at composite time.

**NodeMaterial parity**: identical. `AdditiveBlending` is a preset on
the base `Material`.

## Mode 5 — `opaque`

**State** (`blending-state.ts:opaque` branch):

```
blending          = NoBlending
depthTest         = true
depthWrite        = true
transparent       = false
```

Picking materials (`{point,line,gsplat}-picking-material.ts`) all use
`NoBlending` with `depthTest = depthWrite = true` — the pick buffer
records the brightest contributor at each pixel via
`gl_FragDepth = 1.0 - brightness`.

**NodeMaterial parity**: identical. `NoBlending` is a preset.

## Watch out for

1. **Per-channel alpha overrides** (`gsplat-material.ts:444-460`).
   These are real material-property writes — they need to remain in
   the material's `applyBlendingMode` method post-port. Materials
   that don't set them get `null` (Three.js default), which means
   the alpha channel tracks RGB. The current code is explicit about
   when it diverges; preserve that explicitness in the TSL port.

2. **`shaderOutputMode` is not a Three.js property.** It's a
   Luxar-internal hint returned by `getCompleteBlendingState` to
   tell materials *which shader define to set*
   (`LUXAR_MAX_RGB_CONTRIBUTION`). Under TSL, this becomes a runtime
   uniform / node-graph switch rather than a `#define`. Materials
   that consume the hint (point and line) need to be ported
   accordingly.

3. **Customblending mode flip cleanup.** When `applyBlendingMode`
   switches *out* of max-mode (e.g., to normal), it must reset
   `blendEquation`, `blendSrc`, `blendDst`, and the per-channel
   alpha overrides — otherwise the material strands `MaxEquation`
   state on top of the next non-max mode. See
   `gsplat-material.ts:466-470` and `line-material.ts:346-350` for
   the existing reset logic. Carry it into the TSL port.

## References

- [Three.js manual — "Custom blending equations and equations"](https://threejs.org/manual/en/custom-blending-equations.html)
- [Three.js manual — "Using WebGPURenderer"](https://threejs.org/manual/en/webgpurenderer.html)
- `src/rendering/blending-state.ts` — central state shape.
- `src/rendering/{point,line,gsplat}-material.ts::applyBlendingMode`
  — per-geometry blending overrides.
