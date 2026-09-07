"""Build-time audio synthesis and AAC encoding for demo sound effects.

The sound layer plays stored MP3/AAC clips only (``SOUND_SPEC.md`` §3.1), and
Python has no encoder of its own, so a demo that wants a procedural effect —
the per-cluster hum of the stories demo, an ambisonic bed — renders PCM with
numpy and hands it to whichever encoder the build box has:

1. **macOS ``afconvert``** (always present on macOS) → AAC in ``.m4a``;
2. **``ffmpeg``** when it is on ``PATH`` → the same;
3. otherwise a ``UserWarning`` and ``None``: the demo builds without the
   effect rather than failing, exactly like :mod:`luxar.demos._narration`.

Both encoders take a multichannel WAV, so the same path serves a mono hum and a
four-channel first-order ambisonic bed. ``LUXAR_AUDIO_ENCODER``
(``afconvert`` / ``ffmpeg`` / ``none``) overrides the detection — ``none`` keeps
a CI build silent and deterministic.

Clips are cached by a hash of the synthesis parameters, so rebuilding a scene
re-encodes nothing.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess  # noqa: S404 - fixed argv, no shell, see the encoders below
import tempfile
import warnings
import wave
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np
from arbol import aprint

#: Environment override for the encoder choice (``afconvert`` / ``ffmpeg`` / ``none``).
AUDIO_ENCODER_ENV = "LUXAR_AUDIO_ENCODER"

#: AAC bitrate for synthesised effects. Plenty for a hum; four ambisonic
#: channels share it, which is still transparent for a slow-moving bed.
AAC_BITRATE = 128_000

#: An encoder turns a WAV file into an AAC ``.m4a``; it raises on failure.
Encoder = Callable[[Path, Path], None]


def resolve_encoder(encoder: Optional[str] = None) -> Optional[str]:
    """Pick the encoder: explicit → ``LUXAR_AUDIO_ENCODER`` → detection.

    Returns ``None`` when nothing usable is available (or ``none`` was asked).
    """
    choice = encoder if encoder is not None else os.environ.get(AUDIO_ENCODER_ENV)
    if choice is not None:
        choice = choice.strip().lower()
        if choice == "none":
            return None
        if choice in ("afconvert", "ffmpeg"):
            return choice
        raise ValueError(
            f"Unknown audio encoder {choice!r}; expected 'afconvert', 'ffmpeg' or 'none'"
        )
    if shutil.which("afconvert"):
        return "afconvert"
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    return None


def _afconvert(wav: Path, out: Path) -> None:
    exe = shutil.which("afconvert")
    if not exe:
        raise RuntimeError("afconvert is not available")
    proc = subprocess.run(  # noqa: S603 - fixed argv
        [exe, "-f", "m4af", "-d", "aac", "-b", str(AAC_BITRATE), str(wav), str(out)],
        capture_output=True,
    )
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"afconvert failed: {proc.stderr.decode()[:300]}")


def _ffmpeg(wav: Path, out: Path) -> None:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg is not available")
    proc = subprocess.run(  # noqa: S603 - fixed argv
        [
            exe,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(wav),
            "-c:a",
            "aac",
            "-b:a",
            str(AAC_BITRATE),
            # The container by name: the target is written under a temporary
            # suffix, so ffmpeg cannot infer it from the extension.
            "-f",
            "ipod",
            str(out),
        ],
        capture_output=True,
    )
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[:300]}")


_ENCODERS: Dict[str, Encoder] = {"afconvert": _afconvert, "ffmpeg": _ffmpeg}


def write_wav(pcm: np.ndarray, sample_rate: int, path: Path) -> None:
    """Write ``pcm`` (``(n,)`` or ``(n, channels)`` floats in ``[-1, 1]``) as 16-bit WAV."""
    data = np.asarray(pcm, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError(
            f"pcm must be (n,) or (n, channels) with n >= 1, got {data.shape}"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError("pcm must be finite")
    peak = float(np.abs(data).max())
    if peak > 1.0:
        data = data / peak
    ints = np.round(data * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(int(data.shape[1]))
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(ints.tobytes())


def encode_pcm(
    pcm: np.ndarray,
    sample_rate: int,
    out_path: Path,
    *,
    encoder: Optional[str] = None,
    encoders: Optional[Mapping[str, Encoder]] = None,
) -> Optional[Path]:
    """Encode ``pcm`` to AAC at ``out_path`` (``.m4a``); ``None`` when no encoder exists.

    ``encoders`` swaps the encoder table (tests inject a fake so nothing shells
    out). The file is written atomically: a failed encode leaves nothing behind.
    """
    chosen = resolve_encoder(encoder)
    if chosen is None:
        warnings.warn(
            "No AAC encoder available (afconvert on macOS, or ffmpeg on PATH): "
            "synthesised sound effects are skipped and the scene is built "
            "without them.",
            UserWarning,
            stacklevel=2,
        )
        return None
    table = encoders if encoders is not None else _ENCODERS
    render = table.get(chosen)
    if render is None:
        raise ValueError(f"No encoder registered for {chosen!r}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".m4a.tmp")
    with tempfile.TemporaryDirectory(prefix="luxar_audio_") as tmpdir:
        wav = Path(tmpdir) / "clip.wav"
        write_wav(pcm, sample_rate, wav)
        try:
            render(wav, tmp)
            if not tmp.exists() or tmp.stat().st_size == 0:
                raise RuntimeError(f"{chosen} produced no audio")
            tmp.replace(out_path)
        finally:
            if tmp.exists():
                tmp.unlink()
    return out_path


def synthesis_cache_key(kind: str, params: Mapping[str, object]) -> str:
    """Stable digest of what decides the audio: the effect kind and its parameters."""
    items = "|".join(f"{k}={params[k]!r}" for k in sorted(params))
    return hashlib.sha256(f"{kind}|{items}".encode("utf-8")).hexdigest()[:16]


#: Relative partials of the hum: (multiple of the fundamental, amplitude). The
#: slight detunes beat against each other, which is what makes it breathe.
HUM_PARTIALS: Sequence[tuple[float, float]] = (
    (1.0, 1.0),
    (1.003, 0.55),
    (2.0, 0.30),
    (2.996, 0.16),
    (4.01, 0.07),
)


def hum_pcm(
    frequency_hz: float, seconds: float, sample_rate: int = 48_000
) -> np.ndarray:
    """A warm, loop-clean mono hum at ``frequency_hz``, ``seconds`` long.

    Every partial's frequency is snapped to a multiple of ``1 / seconds`` so each
    sinusoid completes a whole number of cycles in the clip and the loop point
    is seamless — no crossfade, no click. The slow amplitude wobble is snapped
    the same way. Peak-normalised to 0.5 so the authored ``gain`` has headroom.
    """
    if frequency_hz <= 0 or seconds <= 0 or sample_rate <= 0:
        raise ValueError("frequency_hz, seconds and sample_rate must be > 0")
    n = int(round(seconds * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    base = 1.0 / seconds  # the loop's fundamental frequency

    def snap(freq: float) -> float:
        return max(base, round(freq / base) * base)

    signal = np.zeros(n, dtype=np.float64)
    for i, (ratio, amp) in enumerate(HUM_PARTIALS):
        f = snap(frequency_hz * ratio)
        phase = 0.37 * i
        signal += amp * np.sin(2 * np.pi * f * t + phase)
    wobble = 1.0 + 0.18 * np.sin(2 * np.pi * snap(0.13) * t)
    signal *= wobble
    peak = float(np.abs(signal).max()) or 1.0
    return (0.5 * signal / peak).astype(np.float32)


def synthesise_hum(
    frequency_hz: float,
    seconds: float,
    cache_dir: Path,
    *,
    sample_rate: int = 48_000,
    encoder: Optional[str] = None,
    encoders: Optional[Mapping[str, Encoder]] = None,
) -> Optional[Path]:
    """Return the path of an AAC hum clip, cached under ``cache_dir``; ``None`` without an encoder."""
    key = synthesis_cache_key(
        "hum",
        {
            "f": round(float(frequency_hz), 4),
            "s": float(seconds),
            "sr": int(sample_rate),
        },
    )
    clip = cache_dir / f"hum_{key}.m4a"
    if clip.exists() and clip.stat().st_size > 0:
        return clip
    aprint(f"🎵 Synthesising hum ({frequency_hz:.1f} Hz, {seconds:g} s)")
    return encode_pcm(
        hum_pcm(frequency_hz, seconds, sample_rate),
        sample_rate,
        clip,
        encoder=encoder,
        encoders=encoders,
    )


# =============================================================================
# Decoding (for re-encoding an existing clip) and first-order ambisonics
# =============================================================================


def _afconvert_decode(src: Path, wav: Path) -> None:
    exe = shutil.which("afconvert")
    if not exe:
        raise RuntimeError("afconvert is not available")
    proc = subprocess.run(  # noqa: S603 - fixed argv
        [exe, "-f", "WAVE", "-d", "LEI16", str(src), str(wav)], capture_output=True
    )
    if proc.returncode != 0 or not wav.exists():
        raise RuntimeError(f"afconvert decode failed: {proc.stderr.decode()[:300]}")


def _ffmpeg_decode(src: Path, wav: Path) -> None:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg is not available")
    proc = subprocess.run(  # noqa: S603 - fixed argv
        [
            exe,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-f",
            "wav",
            "-c:a",
            "pcm_s16le",
            str(wav),
        ],
        capture_output=True,
    )
    if proc.returncode != 0 or not wav.exists():
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode()[:300]}")


#: A decoder turns an encoded clip into a 16-bit PCM WAV; it raises on failure.
Decoder = Callable[[Path, Path], None]
_DECODERS: Dict[str, Decoder] = {
    "afconvert": _afconvert_decode,
    "ffmpeg": _ffmpeg_decode,
}


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV as ``(pcm (n, channels) float32 in [-1, 1], sample_rate)``."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {8 * w.getsampwidth()}-bit")
        channels = w.getnchannels()
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    ints = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
    return (ints.astype(np.float32) / 32768.0), rate


def decode_to_pcm(
    src: Path,
    *,
    decoder: Optional[str] = None,
    decoders: Optional[Mapping[str, Decoder]] = None,
) -> Optional[tuple[np.ndarray, int]]:
    """Decode an MP3/AAC clip to ``(pcm, sample_rate)`` via the same tools that encode.

    ``None`` when no decoder is available (the caller keeps the original clip).
    """
    chosen = resolve_encoder(decoder)
    if chosen is None:
        return None
    table = decoders if decoders is not None else _DECODERS
    render = table.get(chosen)
    if render is None:
        raise ValueError(f"No decoder registered for {chosen!r}")
    with tempfile.TemporaryDirectory(prefix="luxar_audio_") as tmpdir:
        wav = Path(tmpdir) / "clip.wav"
        render(Path(src), wav)
        return read_wav(wav)


def encode_foa(
    mono: np.ndarray, azimuth_deg: float, elevation_deg: float = 0.0
) -> np.ndarray:
    """First-order AmbiX encoding of a mono signal at ``(azimuth, elevation)``.

    Channels in ACN order ``W, Y, Z, X`` with SN3D normalisation: ``W = s``,
    ``Y = s·sin(az)·cos(el)``, ``Z = s·sin(el)``, ``X = s·cos(az)·cos(el)``.
    Azimuth is counter-clockwise from the front (positive = left), as in AmbiX.
    """
    s = np.asarray(mono, dtype=np.float64).reshape(-1)
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.stack(
        [s, s * np.sin(az) * np.cos(el), s * np.sin(el), s * np.cos(az) * np.cos(el)],
        axis=1,
    ).astype(np.float32)


def loop_crossfade(pcm: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    """Make a clip loop cleanly by blending its last ``seconds`` into its first.

    The tail is faded out while the head is faded in underneath it (equal-power,
    so the level does not dip), and the blended stretch replaces the tail: the
    clip gets ``seconds`` shorter and its last sample now leads straight into
    its first, which is what a looping player needs. A clip that ends quieter
    than it starts (the usual "pop" at the loop point) is exactly what this
    fixes. ``seconds <= 0`` returns the input unchanged; a clip shorter than
    twice the fade is refused.
    """
    data = np.asarray(pcm, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if seconds <= 0:
        return data
    n_fade = int(round(seconds * sample_rate))
    if n_fade <= 0:
        return data
    if data.shape[0] < 2 * n_fade:
        raise ValueError(
            f"loop crossfade of {seconds:g} s needs a clip at least {2 * seconds:g} s long "
            f"(got {data.shape[0] / sample_rate:.1f} s)"
        )
    t = np.linspace(0.0, 1.0, n_fade, endpoint=False, dtype=np.float32)[:, None]
    fade_out = np.cos(0.5 * np.pi * t)
    fade_in = np.sin(0.5 * np.pi * t)
    head = data[:n_fade]
    tail = data[-n_fade:]
    blended = tail * fade_out + head * fade_in
    return np.concatenate([data[n_fade:-n_fade], blended], axis=0)


def foa_from_stereo(pcm: np.ndarray, spread_deg: float = 50.0) -> np.ndarray:
    """Place a stereo (or mono) clip in a first-order field: L at ``+spread``, R at ``-spread``.

    The result is a ``(n, 4)`` AmbiX field a viewer can rotate against the
    camera: as the visitor turns left, the left channel drifts to the front.
    """
    data = np.asarray(pcm, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == 1:
        return encode_foa(data[:, 0], 0.0)
    if data.shape[1] != 2:
        raise ValueError(
            f"foa_from_stereo needs mono or stereo input, got {data.shape[1]} channels"
        )
    return encode_foa(data[:, 0], spread_deg) + encode_foa(data[:, 1], -spread_deg)


def synthesise_foa_from_clip(
    src: Path,
    cache_dir: Path,
    *,
    spread_deg: float = 50.0,
    loop_crossfade_s: float = 0.0,
    encoder: Optional[str] = None,
    encoders: Optional[Mapping[str, Encoder]] = None,
    decoders: Optional[Mapping[str, Decoder]] = None,
) -> Optional[Path]:
    """Re-encode a stereo clip as a 4-channel first-order ambisonic AAC field.

    ``loop_crossfade_s`` blends the clip's tail into its head first
    (:func:`loop_crossfade`) so the field loops without a pop. Cached under
    ``cache_dir`` by the source file's size + mtime, the spread and the fade;
    ``None`` when no decoder/encoder is available (the caller keeps the stereo clip).
    """
    src = Path(src)
    stat = src.stat()
    key = synthesis_cache_key(
        "foa",
        {
            "name": src.name,
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
            "spread": float(spread_deg),
            "loop": float(loop_crossfade_s),
        },
    )
    clip = cache_dir / f"foa_{key}.m4a"
    if clip.exists() and clip.stat().st_size > 0:
        return clip
    decoded = decode_to_pcm(src, decoder=encoder, decoders=decoders)
    if decoded is None:
        warnings.warn(
            "No audio decoder available (afconvert on macOS, or ffmpeg on PATH): the "
            "ambisonic bed is skipped and the stereo clip is used as is.",
            UserWarning,
            stacklevel=2,
        )
        return None
    pcm, rate = decoded
    aprint(
        f"🌐 Encoding {src.name} as a first-order ambisonic field (±{spread_deg:g}°"
        + (f", {loop_crossfade_s:g} s loop blend" if loop_crossfade_s > 0 else "")
        + ")"
    )
    looped = loop_crossfade(pcm, rate, loop_crossfade_s)
    return encode_pcm(
        foa_from_stereo(looped, spread_deg),
        rate,
        clip,
        encoder=encoder,
        encoders=encoders,
    )
