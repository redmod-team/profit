import dash
import dash_table
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go


# Read data from a csv
z_data = pd.read_csv(
    'https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = False

tab_selected_style = {
    'backgroundColor': '#119DFF',
    'color': 'white'
}

indata = pd.read_csv('input.txt', delim_whitespace=True, escapechar='#')
outdata = pd.read_csv('output.txt', delim_whitespace=True, escapechar='#')
data = pd.concat([indata, outdata], 1)

app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Tabs(
                    id='tabs',
                    children=[
                        dcc.Tab(
                            label='Graphs',
                            selected_style=tab_selected_style,
                            children=[html.Div(dcc.Graph(id="graph1"))]
                        ),
                        dcc.Tab(
                            label='Tables',
                            selected_style=tab_selected_style,
                            children=[
                                dash_table.DataTable(
                                    id='table',
                                    columns=[{"name": i, "id": i}
                                             for i in data.columns],
                                    data=data.to_dict('records'),
                                    page_size=20,
                                )
                            ]
                        ),
                    ]
                )
            ]
        ),
        html.Div(id='editor')
    ])


@app.callback(
    dash.dependencies.Output("graph1", "figure"),
    [dash.dependencies.Input("tabs", "value")]
)
def update_figure(selected):
    fig = go.Figure(data=[
        # go.Scatter(x=indata.values[:, 0],
        #           y=outdata.values[:, 0], mode='markers'),
        go.Scatter3d(x=indata.values[:, 0],
                     y=indata.values[:, 1],
                     z=outdata.values[:, 0], mode='markers')
    ])
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
