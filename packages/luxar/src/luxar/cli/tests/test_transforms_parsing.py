"""Unit tests for gsplat_ops.transforms_parsing helpers."""

from __future__ import annotations

import pytest
import typer

from luxar.cli.gsplat_ops.transforms_parsing import parse_axis_list


def test_parse_axis_list_valid() -> None:
    assert parse_axis_list("0,1,2", 4, "spatial-dims") == [0, 1, 2]


def test_parse_axis_list_tolerates_spaces_and_trailing_comma() -> None:
    assert parse_axis_list(" 0 , 1 , 2 ,", 4, "spatial-dims") == [0, 1, 2]


@pytest.mark.parametrize("value", ["", ",", "   "])
def test_parse_axis_list_empty_selection_raises(value: str) -> None:
    # An empty/whitespace-only selection returns no axes; it must be rejected
    # rather than flowing downstream as "zero axes" (NaN metrics, silent drop).
    with pytest.raises(typer.BadParameter):
        parse_axis_list(value, 4, "spatial-dims")


def test_parse_axis_list_non_integer_token_raises() -> None:
    with pytest.raises(typer.BadParameter):
        parse_axis_list("a,b,c", 4, "spatial-dims")


def test_parse_axis_list_out_of_range_raises() -> None:
    with pytest.raises(typer.BadParameter):
        parse_axis_list("0,1,9", 4, "spatial-dims")


def test_parse_axis_list_negative_raises() -> None:
    with pytest.raises(typer.BadParameter):
        parse_axis_list("-1,0", 4, "spatial-dims")


def test_parse_axis_list_duplicate_raises() -> None:
    # Headline defect (issue #765): a duplicate axis is double-counted.
    with pytest.raises(typer.BadParameter) as exc_info:
        parse_axis_list("0,0,1", 4, "spatial-dims")
    assert "duplicate" in str(exc_info.value).lower()


def test_parse_axis_list_require_wrong_count_raises() -> None:
    with pytest.raises(typer.BadParameter):
        parse_axis_list("0,1", 4, "x", require=3)


def test_parse_axis_list_require_correct_count() -> None:
    assert parse_axis_list("0,1,2", 4, "x", require=3) == [0, 1, 2]
