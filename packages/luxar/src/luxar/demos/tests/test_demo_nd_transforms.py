"""Exhaustive expected-value audit for the nD transform test bench."""

from .. import demo_nd_transforms as demo

EXPECTED_FRAME_LOCALS = {
    "IDENTITY": tuple(range(16)),
    "OFFSET +5": (None, None, None, None, None, *range(11)),
    "OFFSET -3": (*range(3, 16), None, None, None),
    "SCALE *2": (
        0,
        None,
        1,
        None,
        2,
        None,
        3,
        None,
        4,
        None,
        5,
        None,
        6,
        None,
        7,
        None,
    ),
    "SCALE *2 OFFSET +2": (
        None,
        None,
        0,
        None,
        1,
        None,
        2,
        None,
        3,
        None,
        4,
        None,
        5,
        None,
        6,
        None,
    ),
    "NESTED *2 THEN +1": (
        None,
        None,
        0,
        None,
        1,
        None,
        2,
        None,
        3,
        None,
        4,
        None,
        5,
        None,
        6,
        None,
    ),
    "REVERSE *-1 +15": tuple(range(15, -1, -1)),
}

EXPECTED_CHANNEL_LOCALS = {
    "IDENTITY": (0, 1, 2),
    "SWAP RED-GREEN": (1, 0, 2),
    "ROTATE": (1, 2, 0),
}


def test_every_world_frame_matches_the_readout_and_authored_marker_domain() -> None:
    """Audit all 16 world frames, including both out-of-range edges."""
    assert len(demo.FRAME_ROWS) == len(EXPECTED_FRAME_LOCALS)

    for row in demo.FRAME_ROWS:
        name = row.label.split("\n")[0]
        expected = EXPECTED_FRAME_LOCALS[name]
        actual = tuple(row.local_for_world(world) for world in range(demo.N_FRAMES))
        assert actual == expected
        assert set(row.local_frames()) == {
            local for local in expected if local is not None
        }


def test_every_world_channel_matches_the_readout_inverse_permutation() -> None:
    """Audit all three world channels for every permutation row."""
    assert len(demo.CHANNEL_ROWS) == len(EXPECTED_CHANNEL_LOCALS)

    for label, permutation, _note in demo.CHANNEL_ROWS:
        name = label.split("\n")[0]
        actual = tuple(
            demo.channel_local_for_world(permutation, world)
            for world in range(len(demo.CHANNELS))
        )
        assert actual == EXPECTED_CHANNEL_LOCALS[name]
