from math import ceil

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def structured_numeric_columns(*arrays):
    columns = {}
    for array in arrays:
        for name in array.dtype.names:
            values = np.asarray(array[name]).reshape(-1)
            if np.issubdtype(values.dtype, np.number):
                key = name if name not in columns else f"output:{name}"
                columns[key] = values
    return columns


def finite_values(values):
    values = np.asarray(values).reshape(-1)
    return values[np.isfinite(values)]


def checked_mask(mask, size):
    if mask is None:
        return np.ones(size, dtype=bool)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if len(mask) != size:
        raise ValueError("mask length must match column length")
    return mask


def conditional_histogram_figure(columns, mask=None, bins=30, max_cols=3):
    if not columns:
        return go.Figure()

    first = next(iter(columns.values()))
    mask = checked_mask(mask, len(first))

    names = list(columns)
    ncols = min(max_cols, len(names))
    nrows = ceil(len(names) / ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=names)

    for index, name in enumerate(names):
        values = np.asarray(columns[name]).reshape(-1)
        all_values = finite_values(values)
        selected_values = finite_values(values[mask])
        row = index // ncols + 1
        col = index % ncols + 1
        showlegend = index == 0

        fig.add_trace(
            go.Histogram(
                x=all_values,
                nbinsx=bins,
                histnorm="probability",
                name="all",
                marker_color="rgba(110, 110, 110, 0.35)",
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Histogram(
                x=selected_values,
                nbinsx=bins,
                histnorm="probability",
                name="filtered",
                marker_color="rgba(28, 93, 153, 0.75)",
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Conditional marginal distributions",
        barmode="overlay",
        height=max(360, 300 * nrows),
        margin=dict(l=50, r=20, t=70, b=45),
    )
    fig.update_yaxes(title_text="probability")
    return fig
