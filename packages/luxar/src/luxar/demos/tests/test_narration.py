"""Tests for ``luxar.demos._narration`` — engine choice, the cache, the fallbacks.

Everything runs against an injected fake engine: nothing here touches the
network or the system voice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from luxar.demos._narration import (
    ENGINE_EXTENSIONS,
    NARRATION_ENGINE_ENV,
    narration_cache_key,
    resolve_engine,
    synthesise,
)


def _fake_engines(calls: list[tuple[str, str, Path]]):
    def render(text: str, voice: str, out_path: Path) -> None:
        calls.append((text, voice, out_path))
        out_path.write_bytes(b"\xff\xfb\x90\x00" + text.encode("utf-8"))

    return {"openai": render, "say": render}


def test_resolve_engine_prefers_the_explicit_choice_then_env_then_detection(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(NARRATION_ENGINE_ENV, raising=False)
    monkeypatch.setattr("luxar.demos._narration.shutil.which", lambda _name: None)
    assert resolve_engine() is None
    assert resolve_engine("none") is None
    assert resolve_engine("say") == "say"

    monkeypatch.setenv(NARRATION_ENGINE_ENV, "none")
    assert resolve_engine() is None
    monkeypatch.setenv(NARRATION_ENGINE_ENV, "openai")
    assert resolve_engine() == "openai"
    assert resolve_engine("say") == "say"  # explicit beats env

    monkeypatch.delenv(NARRATION_ENGINE_ENV)
    # An ambient OPENAI_API_KEY must NOT select the paid engine. It is present
    # on many developer machines for unrelated reasons, and auto-selecting it
    # spent someone else's money without a decision being made.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert resolve_engine() is None
    # ...but an explicit request still works, key or no key.
    assert resolve_engine("openai") == "openai"
    monkeypatch.setenv(NARRATION_ENGINE_ENV, "openai")
    assert resolve_engine() == "openai"
    monkeypatch.delenv(NARRATION_ENGINE_ENV)

    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setattr(
        "luxar.demos._narration.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    assert resolve_engine() == "say"

    # The local engine is chosen even with a key present: free beats paid when
    # nobody asked for paid.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert resolve_engine() == "say"
    monkeypatch.delenv("OPENAI_API_KEY")

    with pytest.raises(ValueError, match="Unknown narration engine"):
        resolve_engine("espeak")


def test_synthesise_caches_by_engine_voice_and_text(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls: list[tuple[str, str, Path]] = []
    engines = _fake_engines(calls)

    # engine="openai" is explicit: the key alone no longer selects the paid
    # engine, and this test is about cache keying, not engine detection.
    clip = synthesise(
        "Hello  world.", "alloy", tmp_path, engine="openai", engines=engines
    )
    assert clip is not None and clip.suffix == ".mp3"
    assert clip.read_bytes().endswith(b"Hello world.")  # whitespace normalised
    assert len(calls) == 1
    assert (tmp_path / f"{clip.stem}.json").exists()
    provenance = json.loads((tmp_path / f"{clip.stem}.json").read_text())
    assert provenance == {"engine": "openai", "voice": "alloy", "text": "Hello world."}

    # Same text → cache hit, no synthesis.
    again = synthesise(
        "Hello world.", "alloy", tmp_path, engine="openai", engines=engines
    )
    assert again == clip
    assert len(calls) == 1

    # A different voice or text or engine is a different clip.
    other = synthesise(
        "Hello world.", "nova", tmp_path, engine="openai", engines=engines
    )
    assert other != clip and len(calls) == 2
    said = synthesise("Hello world.", "alloy", tmp_path, engine="say", engines=engines)
    assert said is not None and said.suffix == ".m4a" and len(calls) == 3
    assert narration_cache_key("openai", "alloy", "x") != narration_cache_key(
        "say", "alloy", "x"
    )


def test_synthesise_without_an_engine_warns_and_returns_none(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(NARRATION_ENGINE_ENV, raising=False)
    monkeypatch.setattr("luxar.demos._narration.shutil.which", lambda _name: None)
    with pytest.warns(UserWarning, match="No narration engine"):
        assert synthesise("text", "alloy", tmp_path) is None
    assert not any(tmp_path.iterdir()) if tmp_path.exists() else True


def test_synthesise_leaves_no_partial_clip_when_the_engine_fails(tmp_path) -> None:
    def broken(_text: str, _voice: str, out_path: Path) -> None:
        out_path.write_bytes(b"half")
        raise RuntimeError("engine exploded")

    with pytest.raises(RuntimeError, match="engine exploded"):
        synthesise(
            "text", "alloy", tmp_path, engine="openai", engines={"openai": broken}
        )
    assert list(tmp_path.glob("*.mp3")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_synthesise_refuses_empty_text(tmp_path) -> None:
    with pytest.raises(ValueError, match="empty"):
        synthesise("   ", "alloy", tmp_path, engine="say", engines=_fake_engines([]))


def test_engine_extensions_are_formats_the_sound_writer_accepts() -> None:
    from luxar.validation.sound import AUDIO_FORMAT_FILENAMES

    accepted = {
        Path(name).suffix.lstrip(".") for name in AUDIO_FORMAT_FILENAMES.values()
    }
    assert set(ENGINE_EXTENSIONS.values()) <= accepted
