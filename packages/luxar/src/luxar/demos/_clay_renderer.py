"""GPU "clay" renderer for the PDB turntables (moderngl, offscreen, no window).

The molecular surface is computed ONCE (by PyMOL, exported as one OBJ mesh per
chain — see :mod:`luxar.demos._pdb_turntable`) and shaded here on the GPU: a
matte surface per chain in a pastel shade of the story colour, three soft
lights with a little specular, and **screen-space ambient occlusion**, over a
transparent background. 2x supersampling gives the antialiasing. Frames stream
straight into ffmpeg as raw RGBA.

Why not ray-trace: PyMOL's ``ray`` rebuilds and traces the whole surface for
every frame (2-6 s each on a laptop, no GPU path), so ten 900-frame turntables
took five and a half hours. Here the geometry is uploaded once and a frame costs
a few milliseconds; the whole set renders in the time ffmpeg needs to encode it.

Passes per frame, all at the supersampled size ``S = size * supersample``:

1. G-buffer: albedo (rgba8, alpha 1 on geometry, 0 background), view-space
   normal (rgba16f) and depth.
2. SSAO: hemisphere kernel (``ao_samples``) with a 4x4 rotation-noise tile and a
   range check; radius is a fraction of the bounding radius so the look is the
   same on a toxin and on photosystem II.
3. 4x4 blur of the occlusion.
4. Composite + downsample: lighting x AO, then a 2x2 average in premultiplied
   space, un-premultiplied to straight alpha (a plain average of RGB with the
   transparent background darkens every silhouette edge).

Coordinates: world y is the turntable axis. :func:`principal_frame` maps the
mesh's longest principal axis onto y (the structure stands on its long axis),
the second onto x. The camera sits on +z looking at the origin, at the distance
that fits the bounding sphere at ``fov_deg`` at every angle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

RGB = tuple[float, float, float]

MODERNGL_INSTALL_HINT = (
    "moderngl (GPU offscreen rendering) is needed for the turntables:\n"
    "    pip install 'luxar[demos]'        # or: luxar demo deps --install"
)


# ----------------------------------------------------------------------------
# Geometry (pure numpy, unit-tested without a GPU)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Mesh:
    """An unindexed triangle list with per-vertex normals and one flat colour."""

    positions: np.ndarray  # (N, 3) float32, N a multiple of 3
    normals: np.ndarray  # (N, 3) float32
    color: RGB


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a Wavefront OBJ (``v`` / ``vn`` / triangular ``f a//b``) into flat
    per-corner positions and normals, ``(3F, 3)`` each.

    Only what PyMOL's surface export writes is supported: no texture coords,
    triangles only, ``v//vn`` face corners (a bare ``v`` corner takes the
    position index as its normal index).
    """
    text = path.read_text()
    lines = text.split("\n")
    v_lines = [ln[2:] for ln in lines if ln.startswith("v ")]
    n_lines = [ln[3:] for ln in lines if ln.startswith("vn ")]
    f_lines = [ln[2:] for ln in lines if ln.startswith("f ")]
    if not v_lines or not f_lines:
        raise ValueError(f"{path}: no geometry")
    positions = np.array(" ".join(v_lines).split(), dtype=np.float32).reshape(-1, 3)
    normals = (
        np.array(" ".join(n_lines).split(), dtype=np.float32).reshape(-1, 3)
        if n_lines
        else None
    )
    corners = " ".join(f_lines).replace("//", " ").split()
    idx = np.array(corners, dtype=np.int64)
    if len(idx) == 3 * len(f_lines):  # bare `v` corners
        vi = ni = idx - 1
    elif len(idx) == 6 * len(f_lines):  # `v//vn` corners
        idx = idx.reshape(-1, 2) - 1
        vi, ni = idx[:, 0], idx[:, 1]
    else:
        raise ValueError(f"{path}: faces are not triangles with v//vn corners")
    pos = positions[vi]
    if normals is None:
        # Flat normals per triangle.
        tri = pos.reshape(-1, 3, 3)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        nrm = np.repeat(n, 3, axis=0).astype(np.float32)
    else:
        nrm = normals[ni]
    return pos.astype(np.float32), nrm.astype(np.float32)


def principal_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre and rotation that stand a point cloud on its longest axis.

    Returns ``(center, R)`` with ``R`` a right-handed 3x3 rotation such that
    ``(p - center) @ R.T`` puts the largest principal axis along +y (vertical,
    the turntable axis), the second along +x and the smallest along +z.
    """
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]  # largest first
    e_long, e_mid, e_short = (evecs[:, i] for i in order)
    rows = np.stack([e_mid, e_long, e_short])  # rows: new x, new y, new z
    if np.linalg.det(rows) < 0:
        rows[2] = -rows[2]
    return center.astype(np.float32), rows.astype(np.float32)


def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Row-major OpenGL perspective matrix (upload transposed)."""
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def rotation_y(deg: float) -> np.ndarray:
    """Row-major 4x4 rotation about +y (positive = front moves to the right)."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


# ----------------------------------------------------------------------------
# Shaders
# ----------------------------------------------------------------------------

_GEOM_VS = """
#version 330
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
in vec3 in_pos;
in vec3 in_nrm;
out vec3 v_nrm;
void main() {
    vec4 p = u_view * u_model * vec4(in_pos, 1.0);
    v_nrm = mat3(u_view) * mat3(u_model) * in_nrm;
    gl_Position = u_proj * p;
}
"""

_GEOM_FS = """
#version 330
uniform vec3 u_color;
in vec3 v_nrm;
layout(location = 0) out vec4 f_albedo;
layout(location = 1) out vec4 f_nrm;
void main() {
    f_albedo = vec4(u_color, 1.0);
    f_nrm = vec4(normalize(v_nrm), 1.0);
}
"""

_QUAD_VS = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_SSAO_FS = """
#version 330
uniform sampler2D u_depth;
uniform sampler2D u_nrm;
uniform sampler2D u_noise;
uniform mat4 u_proj;
uniform mat4 u_inv_proj;
uniform vec3 u_kernel[64];
uniform int u_samples;
uniform float u_radius;
uniform float u_bias;
uniform vec2 u_noise_scale;
in vec2 v_uv;
out float f_ao;

vec3 view_pos(vec2 uv) {
    float d = texture(u_depth, uv).r;
    vec4 clip = vec4(uv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
    vec4 v = u_inv_proj * clip;
    return v.xyz / v.w;
}

void main() {
    float d = texture(u_depth, v_uv).r;
    if (d >= 1.0) { f_ao = 1.0; return; }
    vec3 frag = view_pos(v_uv);
    vec3 n = normalize(texture(u_nrm, v_uv).xyz);
    vec3 rnd = texture(u_noise, v_uv * u_noise_scale).xyz;
    vec3 t = normalize(rnd - n * dot(rnd, n));
    vec3 b = cross(n, t);
    mat3 tbn = mat3(t, b, n);
    float occlusion = 0.0;
    for (int i = 0; i < u_samples; ++i) {
        vec3 sp = frag + tbn * u_kernel[i] * u_radius;
        vec4 o = u_proj * vec4(sp, 1.0);
        vec2 ouv = o.xy / o.w * 0.5 + 0.5;
        if (ouv.x < 0.0 || ouv.x > 1.0 || ouv.y < 0.0 || ouv.y > 1.0) continue;
        float sd = view_pos(ouv).z;
        float range = smoothstep(0.0, 1.0, u_radius / max(abs(frag.z - sd), 1e-6));
        occlusion += (sd >= sp.z + u_bias ? 1.0 : 0.0) * range;
    }
    f_ao = 1.0 - occlusion / float(u_samples);
}
"""

_BLUR_FS = """
#version 330
uniform sampler2D u_ao;
in vec2 v_uv;
out float f_ao;
void main() {
    vec2 ts = 1.0 / vec2(textureSize(u_ao, 0));
    float acc = 0.0;
    for (int x = -2; x < 2; ++x)
        for (int y = -2; y < 2; ++y)
            acc += texture(u_ao, v_uv + vec2(float(x), float(y)) * ts).r;
    f_ao = acc / 16.0;
}
"""

_COMPOSITE_FS = """
#version 330
uniform sampler2D u_albedo;
uniform sampler2D u_nrm;
uniform sampler2D u_ao;
uniform int u_ss;               // supersample factor
uniform float u_ao_strength;
uniform float u_ao_power;
uniform float u_ambient;
uniform vec3 u_key_dir;         // unit vectors TOWARD the light, view space
uniform vec3 u_fill_dir;
uniform vec3 u_top_dir;
uniform float u_key;
uniform float u_fill;
uniform float u_top;
uniform float u_spec;
uniform float u_spec_power;
in vec2 v_uv;
out vec4 f_color;

vec4 shade(ivec2 px) {
    vec4 albedo = texelFetch(u_albedo, px, 0);
    if (albedo.a <= 0.0) return vec4(0.0);
    vec3 n = normalize(texelFetch(u_nrm, px, 0).xyz);
    float ao = mix(1.0, pow(clamp(texelFetch(u_ao, px, 0).r, 0.0, 1.0), u_ao_power), u_ao_strength);
    vec3 v = vec3(0.0, 0.0, 1.0);
    float diff = u_key * max(dot(n, u_key_dir), 0.0)
               + u_fill * max(dot(n, u_fill_dir), 0.0)
               + u_top * max(dot(n, u_top_dir), 0.0);
    vec3 h = normalize(u_key_dir + v);
    float spec = u_spec * pow(max(dot(n, h), 0.0), u_spec_power);
    // AO darkens the ambient term fully and the direct light partly (a lit
    // crevice still reads as a crevice), as PyMOL's ray-traced AO does.
    vec3 c = albedo.rgb * (u_ambient * ao + diff * mix(1.0, ao, 0.6)) + spec * ao;
    return vec4(c, 1.0);  // premultiplied == straight while alpha is 1
}

void main() {
    ivec2 base = ivec2(gl_FragCoord.xy) * u_ss;
    vec4 acc = vec4(0.0);
    for (int x = 0; x < u_ss; ++x)
        for (int y = 0; y < u_ss; ++y)
            acc += shade(base + ivec2(x, y));
    acc /= float(u_ss * u_ss);
    // Un-premultiply so the WebM's alpha plane carries straight colour.
    f_color = acc.a > 0.0 ? vec4(acc.rgb / acc.a, acc.a) : vec4(0.0);
}
"""


# ----------------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------------


class ClayRenderer:
    """Offscreen GPU renderer for one turntable; create once, render many frames.

    ``size`` is the output edge in pixels; ``supersample`` the antialiasing
    factor (2 = 4 samples per pixel). ``ao_radius_frac`` is the SSAO radius as
    a fraction of the mesh's bounding radius (molecular pockets are a few Å deep on
    a structure tens of Å across); ``ao_strength`` 0 disables AO and ``ao_power``
    steepens its curve (1 = raw occlusion).
    Raises ``RuntimeError`` with an install hint when moderngl is missing and
    whatever moderngl raises when no GL context can be created (a headless
    Linux box without EGL/X, for instance).
    """

    def __init__(
        self,
        size: int = 768,
        *,
        supersample: int = 2,
        fov_deg: float = 22.0,
        ao_samples: int = 32,
        ao_radius_frac: float = 0.18,
        ao_strength: float = 1.0,
        ao_power: float = 2.8,
    ) -> None:
        from luxar.demos._dependencies import require_module

        moderngl = require_module("moderngl")  # raises with the pinned install hint
        self._gl = moderngl
        self.size = int(size)
        self.ss = int(supersample)
        self.S = self.size * self.ss
        self.fov_deg = fov_deg
        self.ao_samples = min(int(ao_samples), 64)
        self.ao_radius_frac = ao_radius_frac
        self.ao_strength = ao_strength
        self.ao_power = ao_power
        self.ctx: Any = moderngl.create_standalone_context()
        ctx = self.ctx
        self._geom = ctx.program(vertex_shader=_GEOM_VS, fragment_shader=_GEOM_FS)
        self._ssao = ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_SSAO_FS)
        self._blur = ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_BLUR_FS)
        self._comp = ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_COMPOSITE_FS)

        S = self.S
        self._t_albedo = ctx.texture((S, S), 4)
        self._t_nrm = ctx.texture((S, S), 4, dtype="f2")
        self._t_depth = ctx.depth_texture((S, S))
        self._fbo_g = ctx.framebuffer(
            color_attachments=[self._t_albedo, self._t_nrm],
            depth_attachment=self._t_depth,
        )
        self._t_ao = ctx.texture((S, S), 1, dtype="f2")
        self._fbo_ao = ctx.framebuffer(color_attachments=[self._t_ao])
        self._t_blur = ctx.texture((S, S), 1, dtype="f2")
        self._fbo_blur = ctx.framebuffer(color_attachments=[self._t_blur])
        self._t_out = ctx.texture((self.size, self.size), 4)
        self._fbo_out = ctx.framebuffer(color_attachments=[self._t_out])
        for t in (self._t_albedo, self._t_nrm, self._t_depth, self._t_ao, self._t_blur):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            t.repeat_x = t.repeat_y = False
        # A moderngl depth texture defaults to a shadow-compare sampler, which
        # returns 0/1 comparison results instead of depth: every read came back
        # 1.0 and the SSAO pass treated the whole frame as background.
        self._t_depth.compare_func = ""

        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self._quad_vbo = ctx.buffer(quad.tobytes())
        self._quad_ssao = ctx.vertex_array(
            self._ssao, [(self._quad_vbo, "2f", "in_pos")]
        )
        self._quad_blur = ctx.vertex_array(
            self._blur, [(self._quad_vbo, "2f", "in_pos")]
        )
        self._quad_comp = ctx.vertex_array(
            self._comp, [(self._quad_vbo, "2f", "in_pos")]
        )

        rng = np.random.default_rng(7)
        kernel = np.zeros((64, 3), dtype="f4")
        for i in range(self.ao_samples):
            v = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.15, 1)])
            v /= np.linalg.norm(v)
            scale = i / self.ao_samples
            kernel[i] = v * (0.1 + 0.9 * scale * scale)
        self._ssao["u_kernel"].write(kernel.tobytes())
        self._ssao["u_samples"].value = self.ao_samples
        noise = np.zeros((4, 4, 3), dtype="f4")
        noise[..., 0] = rng.uniform(-1, 1, (4, 4))
        noise[..., 1] = rng.uniform(-1, 1, (4, 4))
        self._t_noise = ctx.texture((4, 4), 3, noise.tobytes(), dtype="f4")
        self._t_noise.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._t_noise.repeat_x = self._t_noise.repeat_y = True
        self._ssao["u_noise_scale"].value = (S / 4.0, S / 4.0)

        self._vaos: list[tuple[Any, RGB]] = []
        self._radius = 1.0
        self._view = np.eye(4, dtype="f4")
        self._proj = np.eye(4, dtype="f4")

    # -- scene -------------------------------------------------------------

    def set_meshes(self, meshes: Sequence[Mesh]) -> None:
        """Upload the meshes (already in their final frame) and fit the camera."""
        for vao, _ in self._vaos:
            vao.release()
        self._vaos = []
        allpos = np.concatenate([m.positions for m in meshes])
        centre = 0.5 * (allpos.min(axis=0) + allpos.max(axis=0))
        radius = float(np.linalg.norm(allpos - centre, axis=1).max())
        self._radius = max(radius, 1e-3)
        for m in meshes:
            data = np.hstack([m.positions - centre, m.normals]).astype("f4")
            vbo = self.ctx.buffer(data.tobytes())
            vao = self.ctx.vertex_array(
                self._geom, [(vbo, "3f 3f", "in_pos", "in_nrm")]
            )
            self._vaos.append((vao, m.color))
        # Camera on +z, fitting the bounding sphere with a small margin.
        dist = self._radius / math.sin(math.radians(self.fov_deg) / 2) * 1.06
        view = np.eye(4, dtype="f4")
        view[2, 3] = -dist
        self._view = view
        near = max(dist - self._radius * 1.5, dist * 0.05)
        far = dist + self._radius * 1.5
        self._proj = perspective(self.fov_deg, 1.0, near, far)
        inv = np.linalg.inv(self._proj).astype("f4")
        self._geom["u_view"].write(self._view.T.tobytes())
        self._geom["u_proj"].write(self._proj.T.tobytes())
        self._ssao["u_proj"].write(self._proj.T.tobytes())
        self._ssao["u_inv_proj"].write(inv.T.tobytes())
        self._ssao["u_radius"].value = self._radius * self.ao_radius_frac
        self._ssao["u_bias"].value = self._radius * 0.002

    # -- frame -------------------------------------------------------------

    def render(self, angle_deg: float) -> bytes:
        """One frame at the given turntable angle: straight-alpha RGBA bytes,
        ``size x size``, bottom row first (OpenGL order)."""
        moderngl = self._gl
        ctx = self.ctx
        model = rotation_y(angle_deg)
        self._geom["u_model"].write(model.T.tobytes())

        self._fbo_g.use()
        ctx.viewport = (0, 0, self.S, self.S)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        self._fbo_g.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        for vao, color in self._vaos:
            self._geom["u_color"].value = tuple(float(c) for c in color)
            vao.render(moderngl.TRIANGLES)
        ctx.disable(moderngl.DEPTH_TEST)

        self._fbo_ao.use()
        self._t_depth.use(0)
        self._t_nrm.use(1)
        self._t_noise.use(2)
        self._ssao["u_depth"].value = 0
        self._ssao["u_nrm"].value = 1
        self._ssao["u_noise"].value = 2
        self._quad_ssao.render(moderngl.TRIANGLE_STRIP)

        self._fbo_blur.use()
        self._t_ao.use(0)
        self._blur["u_ao"].value = 0
        self._quad_blur.render(moderngl.TRIANGLE_STRIP)

        self._fbo_out.use()
        ctx.viewport = (0, 0, self.size, self.size)
        self._fbo_out.clear(0.0, 0.0, 0.0, 0.0)
        self._t_albedo.use(0)
        self._t_nrm.use(1)
        self._t_blur.use(2)
        c = self._comp
        c["u_albedo"].value = 0
        c["u_nrm"].value = 1
        c["u_ao"].value = 2
        c["u_ss"].value = self.ss
        c["u_ao_strength"].value = self.ao_strength
        c["u_ao_power"].value = self.ao_power
        c["u_ambient"].value = 0.38
        c["u_key_dir"].value = _unit(-0.45, 0.65, 0.75)
        c["u_fill_dir"].value = _unit(0.8, 0.15, 0.6)
        c["u_top_dir"].value = _unit(0.0, 1.0, 0.35)
        c["u_key"].value = 0.55
        c["u_fill"].value = 0.22
        c["u_top"].value = 0.12
        c["u_spec"].value = 0.10
        c["u_spec_power"].value = 30.0
        self._quad_comp.render(moderngl.TRIANGLE_STRIP)
        return self._fbo_out.read(components=4)

    def release(self) -> None:
        for vao, _ in self._vaos:
            vao.release()
        self._vaos = []
        self.ctx.release()


def _unit(x: float, y: float, z: float) -> tuple[float, float, float]:
    n = math.sqrt(x * x + y * y + z * z)
    return (x / n, y / n, z / n)


# ----------------------------------------------------------------------------
# Turntable driver
# ----------------------------------------------------------------------------


def meshes_from_objs(paths: Sequence[Path], colors: Sequence[RGB]) -> list[Mesh]:
    """Load one OBJ per chain, stand the whole assembly on its longest axis."""
    loaded = [load_obj(p) for p in paths]
    allpos = np.concatenate([p for p, _ in loaded])
    centre, rot = principal_frame(allpos)
    meshes = []
    for (pos, nrm), color in zip(loaded, colors):
        meshes.append(
            Mesh(
                positions=((pos - centre) @ rot.T).astype(np.float32),
                normals=(nrm @ rot.T).astype(np.float32),
                color=color,
            )
        )
    return meshes


def ffmpeg_pipe_command(ffmpeg: str, size: int, fps: int, out_webm: Path) -> list[str]:
    """ffmpeg reading raw RGBA frames on stdin, writing a VP9 WebM WITH alpha.

    ``-vf vflip`` because OpenGL reads the bottom row first; ``-auto-alt-ref 0``
    because VP9 silently drops the alpha plane when alt-ref frames are on.
    """
    return [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{size}x{size}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-vf",
        "vflip",
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "yuva420p",
        "-auto-alt-ref",
        "0",
        "-b:v",
        "0",
        "-crf",
        "30",
        "-row-mt",
        "1",
        "-deadline",
        "good",
        "-cpu-used",
        "2",
        "-an",
        str(out_webm),
    ]


def render_turntable_video(
    meshes: Sequence[Mesh],
    *,
    frames: int,
    fps: int,
    size: int,
    webm: Path,
    poster: Path,
    ffmpeg: str,
    renderer: Optional[ClayRenderer] = None,
    turn_direction: float = 1.0,
) -> None:
    """Render ``frames`` frames of one full turn and encode them.

    ``turn_direction`` +1 spins so the front moves to the right (positive
    rotation about +y); -1 the other way. Frame 0 is saved as the PNG poster.
    """
    import subprocess

    from PIL import Image

    own = renderer is None
    r = renderer or ClayRenderer(size)
    try:
        r.set_meshes(meshes)
        cmd = ffmpeg_pipe_command(ffmpeg, r.size, fps, webm)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)  # noqa: S603
        assert proc.stdin is not None
        step = 360.0 / frames * turn_direction
        try:
            for i in range(frames):
                rgba = r.render(i * step)
                if i == 0:
                    img = Image.frombytes("RGBA", (r.size, r.size), rgba)
                    img.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(poster)
                proc.stdin.write(rgba)
        finally:
            proc.stdin.close()
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with status {rc} encoding {webm}")
    finally:
        if own:
            r.release()
