import plotly.graph_objects as go
import numpy as np


def dens_hist(a, edges, outside):
    count, edges = np.histogram(a, edges)
    if outside[0]:
        count = np.append(len(a[a < edges[0]]), count)
        edges = np.append(edges[0] - (edges[1] - edges[0]), edges)
    if outside[1]:
        count = np.append(count, len(a[a > edges[-1]]))
    density = count * 100 / len(a)
    return density, edges


def format_hist(fig, bins, edges, density, title, outside, colors):
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=dict(
            text="<b>{}</b>".format(title),
            x=0.5,
            y=0.97,
            xanchor="center",
            yanchor="top",
        ),
        titlefont=dict(size=30),
        autosize=False,
        width=380,
        height=400,
        xaxis=dict(
            linecolor="black",
            linewidth=2,
            mirror=True,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            domain=[0.15, 1],
        ),
        yaxis=dict(
            linecolor="black",
            linewidth=2,
            mirror=True,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        margin=dict(l=210, r=1, t=50, b=2),
        plot_bgcolor=colors[0],
        paper_bgcolor=colors[1],
        barmode="overlay",
        showlegend=False,
    )
    annotations = []

    scale = np.max(np.abs(edges))

    if scale < 1e-2:
        fmt = "{:.2e}"
    elif scale < 1e-1:
        fmt = "{:.3f}"
    elif scale < 1e0:
        fmt = "{:.2f}"
    elif scale < 1e1:
        fmt = "{:.1f}"
    elif scale < 1e3:
        fmt = "{:.0f}"
    else:
        fmt = "{:.2e}"

    for k, xd in enumerate(density):
        yd = edges[k]
        if outside[0] and k == 0:
            label = "< {}".format(fmt).format(bins[0])
        elif outside[1] and k == len(density) - 1:
            label = "> {}".format(fmt).format(bins[-1])
        else:
            label = "{0} - {0}".format(fmt).format(edges[k], edges[k + 1])
        annotations.append(
            dict(
                xref="paper",
                yref="y",
                x=-1.2,
                y=yd,
                xanchor="left",
                text="{}".format(label),
                font=dict(size=24),
                showarrow=False,
                align="left",
            )
        )
        # labeling the bar net worth
        annotations.append(
            dict(
                xref="paper",
                yref="y1",
                y=yd,
                x=0.14,
                xanchor="right",
                text="{:.1f}".format(xd) + "%",
                font=dict(size=24),
                showarrow=False,
                align="right",
            )
        )
    fig.update_layout(annotations=annotations)


def draw_hist(edges, density):
    fig = go.Figure(
        data=go.Bar(
            y=edges,
            x=density,
            orientation="h",
            marker=dict(color="black", line=dict(color="white", width=1)),
        )
    )

    return fig


def fig_hist(
    da, bins, title, condi=None, outside=(False, True), colors=("#8ecbad", "#b3ffd9")
):
    density, edges = dens_hist(da, bins, outside)

    if condi is None:
        fig = draw_hist(edges, density)
    else:
        density2, edges2 = dens_hist(da[condi], bins, outside)

        bar1 = go.Bar(
            y=edges2,
            x=density2,
            orientation="h",
            marker=dict(color="black", line=dict(color="black")),
        )
        bar2 = go.Bar(
            y=edges,
            x=density,
            orientation="h",
            marker=dict(color="rgba(0,0,0,0)", line=dict(color="white", width=1)),
        )
        fig = go.Figure(data=[bar1, bar2])

        density = density2
        edges = edges2

    format_hist(fig, bins, edges, density, title, outside, colors)

    return fig


def add_border(fig):
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=-0.325,
        y0=-0.125,
        x1=1.023,
        y1=1.16,
        line=dict(
            color="black",
            width=2,
        ),
    )
