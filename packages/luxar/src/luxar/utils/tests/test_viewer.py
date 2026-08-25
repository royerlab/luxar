"""Tests for demo viewer port and command helpers."""

from pathlib import Path


class TestDemoPorts:
    """Stable per-dataset ports so demos never contend for 8000/5173."""

    def test_deterministic_and_in_range(self) -> None:
        from luxar.utils.viewer import demo_ports

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
        from luxar.utils.viewer import demo_ports

        names = [f"demo_{i}.luxar.zarr" for i in range(24)]
        assert len({demo_ports(n) for n in names}) > 20

    def test_serve_command_appends_derived_ports(self) -> None:
        from luxar.utils.viewer import _serve_command, demo_ports

        data, viewer = demo_ports("x.luxar.zarr")
        cmd = _serve_command("x.luxar.zarr", open_browser=True, serve_args=None)
        assert cmd[-1] == "--open"
        assert ["--port", str(data)] == cmd[cmd.index("--port") :][:2]
        assert ["--viewer-port", str(viewer)] == cmd[cmd.index("--viewer-port") :][:2]

    def test_serve_command_respects_pinned_ports(self) -> None:
        from luxar.utils.viewer import _serve_command

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
        from luxar.utils.viewer import demo_ports

        # Resolve names through the registry's own rule (a stem already ending
        # in `.zarr` is used verbatim) so the guard keeps checking the names
        # demos actually serve.
        pairs: dict[tuple[int, int], list[str]] = {}
        for d in registry.iter_demos():
            for path in registry.demo_output_paths(d, demos_dir=Path("demos")):
                pairs.setdefault(demo_ports(path), []).append(path.name)
        collisions = {k: v for k, v in pairs.items() if len(v) > 1}
        assert not collisions, f"port-pair collisions: {collisions}"
