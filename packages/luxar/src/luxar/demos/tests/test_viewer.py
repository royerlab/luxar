"""Tests for demo viewer port and command helpers."""

from pathlib import Path


class TestDemoPorts:
    """Stable per-dataset ports so demos never contend for 8000/5173."""

    def test_deterministic_and_in_range(self) -> None:
        from luxar.demos._support.runtime.viewer import demo_ports

        data, viewer = demo_ports("global_rivers_earth.luxar.zarr")
        assert (data, viewer) == demo_ports("global_rivers_earth.luxar.zarr")
        # Path components don't matter — only the dataset name does, so the
        # same demo maps to the same URL from any output directory.
        assert (data, viewer) == demo_ports(
            Path("/somewhere/else/global_rivers_earth.luxar.zarr")
        )
        assert 8001 <= data <= 8499
        assert 5200 <= viewer <= 5698
        # Never the bare `luxar serve` defaults.
        assert data != 8000 and viewer != 5173

    def test_different_demos_spread(self) -> None:
        from luxar.demos._support.runtime.viewer import demo_ports

        names = [f"demo_{i}.luxar.zarr" for i in range(24)]
        assert len({demo_ports(n) for n in names}) > 20

    def test_serve_command_appends_derived_ports(self) -> None:
        from luxar.demos._support.runtime.viewer import _serve_command, demo_ports

        data, viewer = demo_ports("x.luxar.zarr")
        cmd = _serve_command("x.luxar.zarr", open_browser=True, serve_args=None)
        assert cmd[-1] == "--auto-open"
        assert ["--port", str(data)] == cmd[cmd.index("--port") :][:2]
        assert ["--viewer-port", str(viewer)] == cmd[cmd.index("--viewer-port") :][:2]

    def test_serve_command_respects_pinned_ports(self) -> None:
        from luxar.demos._support.runtime.viewer import _serve_command

        cmd = _serve_command(
            "x.luxar.zarr",
            open_browser=False,
            serve_args=["--port", "9000"],
        )
        # The explicit pin survives and no second --port is appended.
        assert cmd.count("--port") == 1
        assert cmd[cmd.index("--port") + 1] == "9000"
        assert "--viewer-port" in cmd  # unpinned half still derived

        # The short spelling pins too — Click keeps the LAST occurrence of a
        # repeated option, so a missed pin would silently override the demo.
        for pinned in (["-p", "9100"], ["-p9100"]):
            cmd = _serve_command("x.luxar.zarr", open_browser=False, serve_args=pinned)
            assert "--port" not in cmd
            assert "--viewer-port" in cmd

        cmd = _serve_command(
            "x.luxar.zarr",
            open_browser=False,
            serve_args=["--viewer-port=6000"],
        )
        assert not any(a == "--viewer-port" for a in cmd[cmd.index("--viewer") :][1:])
        assert "--viewer-port=6000" in cmd
        assert "--port" in cmd  # unpinned half still derived

    def test_bundled_demo_table_has_no_full_pair_collisions(self) -> None:
        """No two bundled demo outputs share a full (data, viewer) pair.

        An identical pair reproduces the same-URL stale-tab trap this
        derivation exists to prevent. If adding a demo trips this, rename the
        output or widen the slot ranges.
        """
        from luxar.demos import registry
        from luxar.demos._support.runtime.viewer import demo_ports

        # Resolve names through the registry's own rule (a stem already ending
        # in `.zarr` is used verbatim) so the guard keeps checking the names
        # demos actually serve.
        pairs: dict[tuple[int, int], list[str]] = {}
        for d in registry.iter_demos():
            for path in registry.demo_output_paths(d, demos_dir=Path("demos")):
                pairs.setdefault(demo_ports(path), []).append(path.name)
        collisions = {k: v for k, v in pairs.items() if len(v) > 1}
        assert not collisions, f"port-pair collisions: {collisions}"


class TestServeFlagForwarding:
    """A demo's own --host / --no-open reach `luxar serve` for every demo."""

    def test_host_is_forwarded_from_the_demo_argv(self) -> None:
        from luxar.demos._support.runtime.viewer import _serve_command

        for argv in (["demo", "--host", "0.0.0.0"], ["demo", "--host=0.0.0.0"]):
            cmd = _serve_command("x.luxar.zarr", True, None, argv=argv)
            assert cmd[cmd.index("--host") + 1] == "0.0.0.0"
            assert cmd.count("--host") == 1

    def test_no_host_flag_keeps_the_loopback_default(self) -> None:
        from luxar.demos._support.runtime.viewer import _serve_command

        cmd = _serve_command("x.luxar.zarr", True, None, argv=["demo"])
        assert "--host" not in cmd
        assert cmd[-1] == "--auto-open"

    def test_no_open_suppresses_the_browser(self) -> None:
        from luxar.demos._support.runtime.viewer import _serve_command

        cmd = _serve_command("x.luxar.zarr", True, None, argv=["demo", "--no-open"])
        assert "--auto-open" not in cmd

    def test_a_host_already_in_serve_args_wins(self) -> None:
        """control_serve_args may already carry --host; never append a second."""
        from luxar.demos._support.runtime.viewer import _serve_command

        cmd = _serve_command(
            "x.luxar.zarr",
            False,
            ["--control", "--host", "10.0.0.5"],
            argv=["demo", "--host", "0.0.0.0"],
        )
        assert cmd.count("--host") == 1
        assert cmd[cmd.index("--host") + 1] == "10.0.0.5"

    def test_launch_viewer_reads_sys_argv(self, monkeypatch) -> None:
        """The ~130 `launch_viewer(path)` call sites inherit it with no edits."""
        from luxar.demos._support.runtime import viewer

        seen: list[list[str]] = []
        monkeypatch.setattr(
            viewer, "run_child_process", lambda cmd, **kw: seen.append(cmd) or 0
        )
        monkeypatch.setattr(
            viewer.sys, "argv", ["demo_lorenz.py", "--host", "0.0.0.0", "--no-open"]
        )
        viewer.launch_viewer("lorenz.luxar.zarr")
        assert "--auto-open" not in seen[0]
        assert seen[0][seen[0].index("--host") + 1] == "0.0.0.0"


def _launches_inside_tempdir(source: str) -> list[int]:
    """Line numbers of `launch_viewer(...)` calls inside a `TemporaryDirectory`.

    Structural (AST), so it is blind to how the path is spelled
    (`Path(tmpdir) / …`, `tmp_dir / …`, a helper's return value): a scratch
    directory for intermediates is fine, serving from inside one is not.
    """
    import ast

    def _name(node: object) -> str:
        func = getattr(node, "func", None)
        return str(getattr(func, "attr", None) or getattr(func, "id", None) or "")

    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        if not any(
            isinstance(item.context_expr, ast.Call)
            and _name(item.context_expr) == "TemporaryDirectory"
            for item in node.items
        ):
            continue
        for stmt in node.body:
            for inner in ast.walk(stmt):
                if isinstance(inner, ast.Call) and _name(inner) == "launch_viewer":
                    lines.append(inner.lineno)
    return lines


class TestServingDemosWriteToTheDemosDir:
    """Serving used to build into a TemporaryDirectory: no ✓, nothing to reuse."""

    def test_no_demo_serves_out_of_a_temporary_directory(self) -> None:
        demos_dir = Path(__file__).resolve().parents[1]
        offenders = [
            f"{path.name}:{line}"
            for path in sorted(demos_dir.glob("demo_*.py"))
            for line in _launches_inside_tempdir(path.read_text(encoding="utf-8"))
        ]
        assert offenders == []

    def test_the_guard_catches_every_spelling_of_the_old_pattern(self) -> None:
        for body in (
            "    output_path = Path(tmpdir) / 'a.luxar.zarr'\n"
            "    launch_viewer(output_path)\n",
            "    zarr_path = tmp_dir / 'a.luxar.zarr'\n"
            "    launch_viewer(zarr_path, serve_args=[])\n",
            "    if True:\n        viewer.launch_viewer(build(tmp))\n",
        ):
            source = (
                "import tempfile\n"
                "with tempfile.TemporaryDirectory(prefix='x') as tmp_dir:\n" + body
            )
            assert _launches_inside_tempdir(source), body
        clean = (
            "with tempfile.TemporaryDirectory() as tmp:\n"
            "    extract(tmp)\n"
            "launch_viewer(out)\n"
        )
        assert _launches_inside_tempdir(clean) == []
