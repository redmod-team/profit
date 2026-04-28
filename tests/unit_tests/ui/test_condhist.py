import numpy as np
import pytest

from profit.ui.condhist import (
    checked_mask,
    conditional_histogram_figure,
    finite_values,
    structured_numeric_columns,
)


def test_structured_numeric_columns_keeps_numeric_fields_only():
    data = np.array(
        [(1.0, 2, "a"), (3.0, 4, "b")],
        dtype=[("x", "f8"), ("count", "i8"), ("label", "U1")],
    )

    columns = structured_numeric_columns(data)

    assert list(columns) == ["x", "count"]
    np.testing.assert_array_equal(columns["x"], np.array([1.0, 3.0]))
    np.testing.assert_array_equal(columns["count"], np.array([2, 4]))


def test_checked_mask_rejects_wrong_length():
    with pytest.raises(ValueError, match="mask length"):
        checked_mask([True, False], 3)


def test_finite_values_removes_nan_and_inf():
    values = finite_values([1.0, np.nan, np.inf, -np.inf, 2.0])

    np.testing.assert_array_equal(values, np.array([1.0, 2.0]))


def test_conditional_histogram_overlays_all_and_filtered_samples():
    columns = {
        "x": np.array([0.0, 1.0, 2.0, 3.0]),
        "y": np.array([10.0, 11.0, 12.0, 13.0]),
    }
    mask = np.array([False, True, True, False])

    fig = conditional_histogram_figure(columns, mask, bins=2, max_cols=2)

    assert len(fig.data) == 4
    assert fig.data[0].name == "all"
    assert fig.data[1].name == "filtered"
    np.testing.assert_array_equal(fig.data[1].x, np.array([1.0, 2.0]))
