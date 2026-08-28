#### Mesh textures can stay GPU-compressed with KTX2/Basis (#2190)

Mesh texture authoring now accepts `texture_encoding="ktx2"` for uint8 RGB/RGBA
input. The optional path invokes Khronos `toktx`, defaults to UASTC, exposes
ETC1S and codec-specific quality settings, preserves alpha, and keeps Pillow out
of the authoring path by feeding Netpbm directly to `toktx`. Single-channel and
HDR inputs are rejected explicitly and remain supported by `raw`.

The viewer transcodes KTX2 through Three.js for both WebGL and WebGPU, ships the
Basis transcoder assets in standalone builds, and rejects the node clearly when
the device has no native compressed-texture target instead of silently expanding
to RGBA8. Preflight remains device-independent and charges one byte per pixel
plus the full mip tail.

The shared Earth builder keeps its portable WebP default; KTX2 remains an
explicit opt-in until the demo migration and republish are completed together.
