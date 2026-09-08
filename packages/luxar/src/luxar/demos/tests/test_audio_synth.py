"""Tests for ``luxar.demos._audio_synth`` — WAV writing, the hum, the encoder table, the cache.

The encoders are injected fakes: nothing here shells out.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from luxar.demos._audio_synth import (
    AUDIO_ENCODER_ENV,
    decode_to_pcm,
    encode_foa,
    encode_pcm,
    foa_from_stereo,
    hum_pcm,
    loop_crossfade,
    read_wav,
    resolve_encoder,
    synthesis_cache_key,
    synthesise_foa_from_clip,
    synthesise_hum,
    write_wav,
)
from luxar.validation.sound import sniff_audio_format

#: A minimal ISO-BMFF header the sound validator sniffs as AAC.
FAKE_M4A = b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 64


def _fake_encoders(calls: list[tuple[Path, Path]]):
    def render(wav: Path, out: Path) -> None:
        calls.append((wav, out))
        # The fake reads the WAV so a broken writer would surface here.
        with wave.open(str(wav), "rb") as w:
            assert w.getnframes() > 0
        out.write_bytes(FAKE_M4A)

    return {"afconvert": render, "ffmpeg": render}


def test_write_wav_round_trips_channels_rate_and_normalises_peaks(tmp_path) -> None:
    pcm = np.stack([np.linspace(-2, 2, 100), np.zeros(100)], axis=1)
    path = tmp_path / "x.wav"
    write_wav(pcm, 44_100, path)
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 2
        assert w.getframerate() == 44_100
        assert w.getsampwidth() == 2
        frames = np.frombuffer(w.readframes(100), dtype="<i2").reshape(100, 2)
    # Peak above 1 is normalised to full scale rather than clipped.
    assert frames[0, 0] == -32767 and frames[-1, 0] == 32767
    assert np.all(frames[:, 1] == 0)
    with pytest.raises(ValueError, match="finite"):
        write_wav(np.array([np.nan]), 48_000, tmp_path / "bad.wav")


def test_hum_is_loop_clean_and_headroomed() -> None:
    sr = 8_000
    pcm = hum_pcm(110.0, 2.0, sr)
    assert pcm.shape == (16_000,) and pcm.dtype == np.float32
    assert 0.49 < float(np.abs(pcm).max()) <= 0.5
    # Loop point: the signal is periodic in the clip length, so the sample
    # after the last one equals the first one (within float noise).
    again = hum_pcm(110.0, 2.0, sr)
    assert np.allclose(pcm, again)
    # ... so the wrap-around step is an ordinary consecutive-sample step.
    wrap = abs(float(pcm[0] - pcm[-1]))
    assert wrap <= float(np.abs(np.diff(pcm)).max()) * 1.05
    with pytest.raises(ValueError):
        hum_pcm(0, 1.0, sr)


def test_resolve_encoder_explicit_env_then_detection(monkeypatch) -> None:
    assert resolve_encoder("none") is None
    assert resolve_encoder("ffmpeg") == "ffmpeg"
    with pytest.raises(ValueError, match="Unknown audio encoder"):
        resolve_encoder("lame")
    monkeypatch.setenv(AUDIO_ENCODER_ENV, "none")
    assert resolve_encoder() is None
    monkeypatch.delenv(AUDIO_ENCODER_ENV)
    monkeypatch.setattr(
        "luxar.demos._audio_synth.shutil.which",
        lambda name: "/usr/bin/x" if name == "ffmpeg" else None,
    )
    assert resolve_encoder() == "ffmpeg"
    monkeypatch.setattr("luxar.demos._audio_synth.shutil.which", lambda _n: None)
    assert resolve_encoder() is None


def test_encode_pcm_without_an_encoder_warns_and_returns_none(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("luxar.demos._audio_synth.shutil.which", lambda _n: None)
    monkeypatch.delenv(AUDIO_ENCODER_ENV, raising=False)
    with pytest.warns(UserWarning, match="No AAC encoder"):
        assert encode_pcm(np.zeros(10), 48_000, tmp_path / "x.m4a") is None
    assert not (tmp_path / "x.m4a").exists()


def test_synthesise_hum_caches_by_parameters_and_yields_a_sniffable_clip(
    tmp_path,
) -> None:
    calls: list[tuple[Path, Path]] = []
    enc = _fake_encoders(calls)
    a = synthesise_hum(
        110.0, 1.0, tmp_path, sample_rate=8_000, encoder="ffmpeg", encoders=enc
    )
    assert a is not None and a.parent == tmp_path and a.suffix == ".m4a"
    assert sniff_audio_format(a.read_bytes()) == "aac"
    assert len(calls) == 1
    # Same parameters → cache hit, no encode.
    assert (
        synthesise_hum(
            110.0, 1.0, tmp_path, sample_rate=8_000, encoder="ffmpeg", encoders=enc
        )
        == a
    )
    assert len(calls) == 1
    # A different pitch is a different clip.
    b = synthesise_hum(
        220.0, 1.0, tmp_path, sample_rate=8_000, encoder="ffmpeg", encoders=enc
    )
    assert b != a and len(calls) == 2
    assert synthesis_cache_key("hum", {"a": 1}) != synthesis_cache_key("hum", {"a": 2})
    assert synthesis_cache_key("hum", {"a": 1, "b": 2}) == synthesis_cache_key(
        "hum", {"b": 2, "a": 1}
    )


def test_encode_pcm_leaves_no_partial_file_when_the_encoder_fails(tmp_path) -> None:
    def broken(_wav: Path, out: Path) -> None:
        out.write_bytes(b"half")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        encode_pcm(
            np.zeros(10),
            48_000,
            tmp_path / "x.m4a",
            encoder="ffmpeg",
            encoders={"ffmpeg": broken},
        )
    assert not (tmp_path / "x.m4a").exists()
    assert not (tmp_path / "x.m4a.tmp").exists()


def test_encode_foa_follows_ambix_acn_sn3d() -> None:
    s = np.array([1.0, -0.5], dtype=np.float32)
    front = encode_foa(s, 0.0)
    assert front.shape == (2, 4)
    assert np.allclose(front[:, 0], s)  # W
    assert np.allclose(front[:, 3], s)  # X = front
    assert np.allclose(front[:, 1], 0) and np.allclose(front[:, 2], 0)
    left = encode_foa(s, 90.0)
    assert np.allclose(left[:, 1], s) and np.allclose(left[:, 3], 0, atol=1e-6)
    up = encode_foa(s, 0.0, 90.0)
    assert np.allclose(up[:, 2], s) and np.allclose(up[:, 3], 0, atol=1e-6)


def test_foa_from_stereo_places_left_and_right_symmetrically() -> None:
    stereo = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    field = foa_from_stereo(stereo, spread_deg=90.0)
    assert field.shape == (2, 4)
    # Sample 0 is pure left → Y = +1; sample 1 pure right → Y = -1; W carries both.
    assert field[0, 1] == pytest.approx(1.0) and field[1, 1] == pytest.approx(-1.0)
    assert np.allclose(field[:, 0], 1.0)
    mono = foa_from_stereo(np.array([0.5, 0.5], dtype=np.float32))
    assert np.allclose(mono[:, 3], 0.5)  # a mono clip sits in front
    with pytest.raises(ValueError, match="mono or stereo"):
        foa_from_stereo(np.zeros((4, 3)))


def test_read_write_wav_round_trip(tmp_path) -> None:
    pcm = np.stack([np.linspace(-0.9, 0.9, 50), np.linspace(0.9, -0.9, 50)], axis=1)
    write_wav(pcm, 22_050, tmp_path / "rt.wav")
    back, rate = read_wav(tmp_path / "rt.wav")
    assert rate == 22_050 and back.shape == (50, 2)
    assert np.allclose(back, pcm, atol=1e-4)


def test_synthesise_foa_from_clip_decodes_encodes_and_caches(tmp_path) -> None:
    src = tmp_path / "bed.mp3"
    src.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 64)
    decodes: list[Path] = []

    def fake_decode(clip: Path, wav: Path) -> None:
        decodes.append(clip)
        write_wav(np.stack([np.ones(20), np.zeros(20)], axis=1) * 0.5, 8_000, wav)

    encoded: list[tuple[int, int]] = []

    def fake_encode(wav: Path, out: Path) -> None:
        with wave.open(str(wav), "rb") as w:
            encoded.append((w.getnchannels(), w.getframerate()))
        out.write_bytes(FAKE_M4A)

    kw = dict(
        encoder="ffmpeg",
        encoders={"ffmpeg": fake_encode},
        decoders={"ffmpeg": fake_decode},
    )
    out = synthesise_foa_from_clip(src, tmp_path / "foa", **kw)
    assert out is not None and out.suffix == ".m4a"
    assert encoded == [(4, 8_000)]
    assert sniff_audio_format(out.read_bytes()) == "aac"
    # Cached by the source's identity.
    assert synthesise_foa_from_clip(src, tmp_path / "foa", **kw) == out
    assert len(decodes) == 1


def test_synthesise_foa_from_clip_without_a_decoder_warns_and_keeps_the_stereo_clip(
    tmp_path, monkeypatch
) -> None:
    src = tmp_path / "bed.mp3"
    src.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 64)
    monkeypatch.setattr("luxar.demos._audio_synth.shutil.which", lambda _n: None)
    monkeypatch.delenv(AUDIO_ENCODER_ENV, raising=False)
    with pytest.warns(UserWarning, match="No audio decoder"):
        assert synthesise_foa_from_clip(src, tmp_path / "foa") is None
    assert decode_to_pcm(src) is None


def test_loop_crossfade_blends_the_tail_into_the_head_at_equal_power() -> None:
    sr = 100
    # A clip that ends quiet and starts loud: the classic loop pop.
    head = np.ones(300, dtype=np.float32)
    body = np.full(400, 0.5, dtype=np.float32)
    tail = np.zeros(300, dtype=np.float32)
    pcm = np.concatenate([head, body, tail])
    out = loop_crossfade(pcm, sr, 3.0)
    assert out.shape == (700, 1)
    # Untouched middle, then the blended stretch: quiet tail fading out under
    # the loud head fading in — it ends at (almost) the head's level, so the
    # loop wraps without a step.
    assert np.allclose(out[:400, 0], np.concatenate([head[300:], body])[:400])
    assert out[-1, 0] == pytest.approx(1.0, abs=0.02)
    assert out[400, 0] == pytest.approx(0.0, abs=0.02)
    # Equal power: with the tail on one channel and the head on the other, the
    # summed power across the blend stays constant (cos² + sin² = 1).
    two = np.zeros((1000, 2), dtype=np.float32)
    two[:200, 1] = 0.3  # head → channel 1
    two[-200:, 0] = 0.3  # tail → channel 0
    flat = loop_crossfade(two, sr, 2.0)
    assert np.allclose(np.sum(flat[-200:] ** 2, axis=1), 0.3**2, atol=1e-6)
    assert loop_crossfade(pcm, sr, 0.0).shape == (1000, 1)
    with pytest.raises(ValueError, match="at least"):
        loop_crossfade(pcm, sr, 6.0)


def test_loop_blend_is_part_of_the_foa_cache_key(tmp_path) -> None:
    src = tmp_path / "bed.mp3"
    src.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 64)

    def fake_decode(clip: Path, wav: Path) -> None:
        write_wav(np.stack([np.ones(4000), np.zeros(4000)], axis=1) * 0.5, 1_000, wav)

    frames: list[tuple[int, int]] = []

    def fake_encode(wav: Path, out: Path) -> None:
        with wave.open(str(wav), "rb") as w:
            frames.append((w.getnchannels(), w.getnframes()))
        out.write_bytes(FAKE_M4A)

    kw = dict(
        encoder="ffmpeg",
        encoders={"ffmpeg": fake_encode},
        decoders={"ffmpeg": fake_decode},
    )
    plain = synthesise_foa_from_clip(src, tmp_path / "foa", **kw)
    blended = synthesise_foa_from_clip(
        src, tmp_path / "foa", loop_crossfade_s=1.0, **kw
    )
    assert plain is not None and blended is not None and plain != blended
    # Four channels both times; the blended field is one second (1000 frames) shorter.
    assert frames == [(4, 4000), (4, 3000)]
