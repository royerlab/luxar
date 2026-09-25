"""Build-time text-to-speech for demo narration.

A demo that narrates a story synthesises the clip when the SCENE is built, not
when it is viewed: the viewer only plays stored MP3/AAC files
(``docs/guides/specs/SOUND_SPEC.md`` §5). The engine is selected explicitly or
from the local free option:

1. **OpenAI TTS** when ``engine="openai"`` or
   ``LUXAR_NARRATION_ENGINE=openai`` is set — plain HTTPS to
   ``/v1/audio/speech`` (no SDK dependency), MP3 out; ``OPENAI_API_KEY`` then
   authenticates the request but never selects the paid engine by itself.
2. **macOS ``say``** when both ``say`` and ``afconvert`` are on ``PATH`` — the
   system voice rendered to AIFF, then AAC in an ``.m4a`` container.
3. Otherwise a ``UserWarning`` and ``None``: the demo builds without narration
   rather than failing (a Linux box without a key should still produce the
   scene).

Clips are cached under the demo's cache directory by a hash of
``(engine, model, voice, text)``, so rebuilding a scene whose text has not
changed costs nothing and asks no API for anything. ``LUXAR_NARRATION_ENGINE``
(``openai`` / ``say`` / ``none``) overrides the auto-detection — ``none`` is how
a CI build stays offline and silent.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess  # noqa: S404 - fixed argv, no shell, see _say_engine
import tempfile
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional

from arbol import aprint

#: Environment override for the engine choice (``openai`` / ``say`` / ``none``).
NARRATION_ENGINE_ENV = "LUXAR_NARRATION_ENGINE"

#: Pinned OpenAI model; part of the cache key, so a model bump re-synthesises.
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"
OPENAI_TIMEOUT_S = 120

#: ``engine name -> file extension`` of the clip it writes.
ENGINE_EXTENSIONS: Dict[str, str] = {"openai": "mp3", "say": "m4a"}

#: An engine renders ``text`` in ``voice`` into ``out_path`` (already carrying
#: the engine's extension). It raises on failure; it never returns partial files.
Engine = Callable[[str, str, Path], None]


def resolve_engine(engine: Optional[str] = None) -> Optional[str]:
    """Pick the engine: explicit → ``LUXAR_NARRATION_ENGINE`` → auto-detect.

    Returns ``None`` when no engine is available (or ``none`` was asked for).
    """
    choice = engine if engine is not None else os.environ.get(NARRATION_ENGINE_ENV)
    if choice is not None:
        choice = choice.strip().lower()
        if choice == "none":
            return None
        if choice not in ENGINE_EXTENSIONS:
            raise ValueError(
                f"Unknown narration engine {choice!r}; expected one of "
                f"{sorted(ENGINE_EXTENSIONS)} or 'none'"
            )
        return choice
    # Deliberately NOT auto-selected from OPENAI_API_KEY. That key is ambient
    # on many developer machines for unrelated reasons, and picking it up here
    # silently billed a paid endpoint the moment a demo generated narration —
    # someone else's money, spent without a decision. The paid path now needs
    # an explicit request: pass engine="openai" or set the engine env var.
    if shutil.which("say") and shutil.which("afconvert"):
        return "say"
    return None


#: Attempts per clip. A long tour makes dozens of requests, and one dropped
#: read must not throw away a build that has run for half an hour.
OPENAI_ATTEMPTS = 3


def _transient(error: Exception) -> bool:
    """A failure worth retrying: a network error, a timeout, 429 or a 5xx."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code == 429 or error.code >= 500
    return isinstance(error, OSError)


def _post_with_retry(request: urllib.request.Request) -> bytes:
    for attempt in range(1, OPENAI_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=OPENAI_TIMEOUT_S) as resp:  # noqa: S310 - fixed https URL
                return resp.read()
        except OSError as e:  # URLError, HTTPError and timeouts are all OSError
            if attempt == OPENAI_ATTEMPTS or not _transient(e):
                if isinstance(e, urllib.error.HTTPError):
                    detail = e.read().decode("utf-8", "replace")[:300]
                    raise RuntimeError(
                        f"OpenAI TTS failed: HTTP {e.code} {detail}"
                    ) from e
                raise RuntimeError(f"OpenAI TTS failed: {e}") from e
            aprint(f"  ⚠ OpenAI TTS attempt {attempt} failed ({e}); retrying")
            time.sleep(2**attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def _openai_engine(text: str, voice: str, out_path: Path) -> None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    body = json.dumps(
        {
            "model": OPENAI_TTS_MODEL,
            "voice": voice,
            "input": text,
            "response_format": "mp3",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OPENAI_TTS_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    payload = _post_with_retry(request)
    if not payload:
        raise RuntimeError("OpenAI TTS returned an empty body")
    out_path.write_bytes(payload)


def _say_engine(text: str, voice: str, out_path: Path) -> None:
    say = shutil.which("say")
    afconvert = shutil.which("afconvert")
    if not say or not afconvert:
        raise RuntimeError("macOS `say` / `afconvert` are not available")
    with tempfile.TemporaryDirectory(prefix="luxar_narration_") as tmp:
        aiff = Path(tmp) / "clip.aiff"
        argv = [say, "-o", str(aiff)]
        if voice:
            argv += ["-v", voice]
        argv.append(text)
        proc = subprocess.run(argv, capture_output=True)  # noqa: S603 - fixed argv
        if proc.returncode != 0:
            raise RuntimeError(f"say failed: {proc.stderr.decode()[:300]}")
        proc = subprocess.run(  # noqa: S603 - fixed argv
            [
                afconvert,
                "-f",
                "m4af",
                "-d",
                "aac",
                "-b",
                "96000",
                str(aiff),
                str(out_path),
            ],
            capture_output=True,
        )
        if proc.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"afconvert failed: {proc.stderr.decode()[:300]}")


_ENGINES: Dict[str, Engine] = {"openai": _openai_engine, "say": _say_engine}


def narration_cache_key(engine: str, voice: str, text: str) -> str:
    """Stable digest of what decides the audio: engine, model, voice, text."""
    model = OPENAI_TTS_MODEL if engine == "openai" else "system"
    payload = f"{engine}|{model}|{voice}|{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def synthesise(
    text: str,
    voice: str,
    cache_dir: Path,
    *,
    engine: Optional[str] = None,
    engines: Optional[Mapping[str, Engine]] = None,
) -> Optional[Path]:
    """Return the path of an MP3/AAC clip narrating ``text``, or ``None``.

    Cached under ``cache_dir`` by :func:`narration_cache_key`; a hit costs
    nothing. ``engine`` forces one engine (``openai`` / ``say`` / ``none``);
    otherwise :func:`resolve_engine` decides. ``engines`` swaps the engine table
    (tests inject a fake so nothing touches the network or the system voice).
    ``None`` means "no narration": the caller skips the node, and a warning says
    why.
    """
    text = " ".join(text.split())
    if not text:
        raise ValueError("narration text is empty")
    chosen = resolve_engine(engine)
    if chosen is None:
        warnings.warn(
            "No narration engine available: set LUXAR_NARRATION_ENGINE=openai "
            "for OpenAI TTS, or build on macOS for the system voice (`say`). "
            "The scene is built without narration.",
            UserWarning,
            stacklevel=2,
        )
        return None
    table = engines if engines is not None else _ENGINES
    render = table.get(chosen)
    if render is None:
        raise ValueError(f"No renderer registered for narration engine {chosen!r}")

    ext = ENGINE_EXTENSIONS[chosen]
    key = narration_cache_key(chosen, voice, text)
    cache_dir.mkdir(parents=True, exist_ok=True)
    clip = cache_dir / f"{key}.{ext}"
    if clip.exists() and clip.stat().st_size > 0:
        return clip

    aprint(f"🗣️  Synthesising narration ({chosen}, voice {voice!r}, {len(text)} chars)")
    tmp = clip.with_suffix(f".{ext}.tmp")
    try:
        render(text, voice, tmp)
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError(f"{chosen} produced no audio")
        tmp.replace(clip)
    finally:
        if tmp.exists():
            tmp.unlink()
    # Provenance beside the clip: what was said, by which voice and engine.
    (cache_dir / f"{key}.json").write_text(
        json.dumps({"engine": chosen, "voice": voice, "text": text}, indent=2),
        encoding="utf-8",
    )
    return clip
