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

The four shared Earth demos now author their basemaps as UASTC KTX2 by default,
cutting their combined resident texture footprint from RGBA8's 4 bytes per pixel
to the compressed mip-chain budget of about 4/3 bytes per pixel. If `toktx` is
not installed, globe authoring reports the downgrade and falls back to WebP
quality 90 rather than failing gallery generation.

Measured with `toktx` 4.4.2 at UASTC quality 2, one real 8193x8192 Blue Marble
tile is 48.57 MiB versus 5.78 MiB as WebP quality 90. Two tiles therefore raise
the basemap wire payload from 11.56 MiB to 97.14 MiB, while the viewer preflight
totals about 335.6 MiB per node and retains about 176.4 MiB below its 512 MiB
limit. UASTC remains the default despite the wire cost because the shared globe
path also carries colour-coded scientific surfaces, where ETC1S block artifacts
can alter data-like colour boundaries.
