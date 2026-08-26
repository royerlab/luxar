#### Mesh nodes carry UVs and a texture (#2175)

Mesh gains per-vertex texture coordinates and a base-colour image, so a surface
can be textured instead of relying on per-vertex colour. This is the Python
write side; the viewer half follows.

The motivation is not subtle. Four demos render the Earth as a point cloud
because there was no textured primitive to do it with — `demo_ocean_currents_earth`
and `demo_global_rivers_earth` at 8,000,000 points each, `demo_earthquakes_3d` at
2,000,000, `demo_biodiversity_planetary_scale` at 700,000 — and three of them
sample NASA Blue Marble per point. `demo_earthquakes_3d` records the reason
outright: *"120k was far too few and it showed… `demo_ocean_currents_earth` uses
8M for the same texture on the same sphere."* Per-vertex colour genuinely cannot
do that job, since resolving a 2048x1024 equirectangular image needs roughly 2M
vertices. A UV sphere is ~20k triangles plus one image.

**Two payloads, one array.** `texture_encoding="raw"` writes an `(H, W, C)`
array; `png`/`webp`/`jpeg` write a 1-D uint8 array of encoded bytes — the same
shape `image_label_bytes` has always used for per-vertex hover thumbnails, so an
encoded texture is a fourth use of a shipped mechanism rather than a new one.
Encoded bytes go into a zarr *array* rather than a loose file beside the store
deliberately: the store abstraction is what makes `.zarr.zip`, remote HTTP,
consolidated metadata and `luxar optimise` work, and `content_hash` covers
arrays. A loose PNG would not be hashed, so editing a texture would not
invalidate a warm viewer cache and users would keep seeing the old image with no
way to tell.

**HDR works, and implies `raw`.** The image codecs are integer-only and no
browser decodes float, so a float texture must be `raw`. Within `raw` it follows
the element-colour contract exactly rather than inventing a second one — float
dtype plus any value above 1.0 is HDR, `PRECISION` keeps float32, `AUTO`
quantizes through `geolog_perchannel_u16`, and the RGB min/max is stamped as
`texture_data_range` the way `color_data_range` already is. Note float16 is
deliberately not offered on disk: the colour encoder already refuted it as
measurably worse than its quantized alternative for wide-range positives.

**The declared dimensions are load-bearing, not metadata.** `texture_width`,
`texture_height` and `texture_channels` are stamped unconditionally — including
for `raw`, where they duplicate the array's own shape. That redundancy is the
point: a compressed image is a decompression bomb, the viewer loads arbitrary
`?src=` URLs, and its mesh preflight budgets a node *before fetching a chunk*. A
reader must never have to open the payload to learn how large it decodes to. They
are writer-reserved for the same reason, so a caller-supplied value cannot
disagree with the payload, and a disagreement is refused rather than silently
resolved in favour of one source.

**One base colour, so texture joins the existing exclusion.** `colors`,
`colormap` and `texture` are mutually exclusive; per-vertex tinting *of* a texture
is a reasonable thing to want and is deliberately not what this does, since it
would need its own shader variant and its own composition semantics. `uvs` and
`texture` are a pair, refused apart for the same reason `normals` and
`normal_dims` are: each renders *something* alone — a texture with no UVs samples
one arbitrary texel across every triangle, UVs with no texture cost a per-vertex
array to affect nothing — and silent-but-wrong is the case that pairing already
exists to make loud.

Textures are refused on the three structural routes for now, each named with its
own reason rather than dismissed: a `partition=` part would duplicate the whole
image, a `substitutive_lod=` level needs its UVs resampled at the collapse
targets, and an `additive_lod=` shell would store the image once per shell. A UV
sphere needs none of them.

Two smaller things fell out. `LuxarZarrCompiler.write_mesh` now forwards to the
writer by keyword — it forwarded positionally, and inserting a parameter silently
shifted every argument after it, which mypy caught only because the shifted types
happened to disagree. And UVs are deliberately not clamped to `[0, 1]` on the way
to disk, because a UV outside the unit square is how a detail texture tiles under
`texture_wrap="repeat"`.
