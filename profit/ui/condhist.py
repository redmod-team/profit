import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from textwrap import dedent as d

import pandas as pd

# df = pd.read_hdf(r'..\examples\algae\MC\mc_out_allyears_sigma1.h5')
df = pd.read_csv(
    r"..\examples\algae\MC\mc_out_allyears_sigma1.dat", delim_whitespace=True
)
data = df.values
param = data[:, 0:8]


def generate_table():
    data = dict()
    for l in range(4):
        for k in range(2):
            data[k + 2 * l] = [{"x": df.iloc[::100, k + 2 * l], "type": "histogram"}]
    return html.Table(
        [
            html.Tr(
                [
                    html.Td(
                        dcc.Graph(
                            id="hist{}".format(k + 2 * l),
                            figure={"data": data[k + 2 * l], "layout": {}},
                        ),
                        style={"width": "500px"},
                    )
                    for k in range(2)
                ],
                style={"height": "300px"},
            )
            for l in range(4)
        ],
    )


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    children=[
        html.H1(children="Conditional probability distributions"),
        html.Div(
            [
                dcc.Markdown(
                    d(
                        """
            **Click** on bars in one histogram to plot conditional
            probability distribution for other variables.
        """
                    )
                ),
                html.Button("Reset", id="reset", n_clicks_timestamp=0),
                # html.Pre(id='click-data'),
            ]
        ),
        generate_table(),
    ]
)

# @app.callback(
#    Output('click-data', 'children'),
#    [Input('hist0', 'clickData')])
# def display_click_data(clickData):
# return json.dumps(clickData['points'][0], indent=2)
#    try:
#        return json.dumps(clickData['points'][0]['binNumber'])
#    except:
#        return 0


def gen_callback(k):
    from numpy import array

    def update_figure(*args):
        defaultdata = {"data": [{"x": df.iloc[::100, k], "type": "histogram"}]}

        # Find out who triggered the callback,
        # see https://github.com/plotly/dash/issues/291
        ctx = dash.callback_context

        # Update was not triggered at all
        if not ctx.triggered:
            return defaultdata
        trigger = ctx.triggered[0]

        # Reset button triggered update
        if trigger["prop_id"] == "reset.n_clicks":
            return defaultdata

        # Other component triggered update
        try:
            reducedset = array(trigger["value"]["points"][0]["pointNumbers"])
        except:
            return defaultdata
        return {"data": [{"x": df.iloc[reducedset, k], "type": "histogram"}]}

    return update_figure


for k in range(8):
    app.callback(
        output=Output("hist{}".format(k), "figure"),
        inputs=(
            [Input("hist{}".format(l), "clickData") for l in range(8)]
            + [Input("reset", "n_clicks")]
        ),
    )(gen_callback(k))


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")
