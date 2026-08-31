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

Measured with `toktx` 4.4.2 at UASTC quality 2, the two real 8193x8192 Blue
Marble tiles are 48.57 and 58.69 MiB, versus 5.78 and 8.17 MiB as WebP quality
90. The basemap wire payload therefore rises from 13.95 MiB to 107.26 MiB. The
larger tile's viewer preflight totals about 386.2 MiB and retains about 125.8 MiB
below the 512 MiB per-node limit. UASTC remains the default despite the wire cost
because the shared globe path also carries colour-coded scientific surfaces,
where ETC1S block artifacts can alter data-like colour boundaries.
