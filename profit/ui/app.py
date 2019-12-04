import dash
import dash_table
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np

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
                    value='tabs',
                    children=[
                        dcc.Tab(
                            label='Tables',
                            selected_style=tab_selected_style,
                            children=[
                                dash_table.DataTable(
                                    id='table',
                                    columns=[{"name": i, "id": i} for i in data.columns],
                                    data=data.to_dict('records'),
                            )
                            ]
                        ),
                        dcc.Tab(
                            label='Graphs',
                            selected_style=tab_selected_style,
                            children=[html.Div(dcc.Graph(id="my-graph"))]
                        )
                    ]
                )
            ]
        ),
        html.Div(id='editor')
    ])

@app.callback(
    dash.dependencies.Output("my-graph", "figure"),[]
)
def update_figure(selected):
    pd.options.mode.chained_assignment = None
    dff = df[df['State FIPS Code'] == selected]
    dff["county"] = (dff["County Name/State Abbreviation"].str.split(",", expand=True))[0]
    df1 = dff.loc[:, ["county", 'Labor Force', 'Employed', 'Unemployed']]
    df1.loc['Labor Force'] = pd.to_numeric(df1['Labor Force'], errors='ignore')
    df1.loc['Employed'] = pd.to_numeric(df1['Employed'], errors='ignore')
    df1.loc['Unemployed'] = pd.to_numeric(df1['Unemployed'], errors='ignore')
    trace = [go.Surface(y=df1.county.values, x=df1.Employed.values, z=df1.values, colorscale="YlGnBu", opacity=0.8,
                        colorbar={"title": "Number", "len": 0.5, "thickness": 15}, )]
    fig = go.Figure(data=trace,
                    layout=go.Layout(title=f'Annual Average of Labor Force Data for {us.states.lookup(str(selected))}',
                                     autosize=True, height=800,
                                     scene={"xaxis": {'title': "Annual Average of Employed  (number)",
                                                      "tickfont": {"size": 10}, 'type': "linear"},
                                            "yaxis": {"title": f"County in {us.states.lookup(str(selected))} ",
                                                      "tickfont": {"size": 10}, "tickangle": 1},
                                            "zaxis": {
                                                'title': "         Annual Average of : <br>Labour Force,Employed,Unemployed  ",
                                                "tickfont": {"size": 10}},
                                            "camera": {"eye": {"x": 2, "y": 1, "z": 1.25}}, "aspectmode": "cube", }))
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
