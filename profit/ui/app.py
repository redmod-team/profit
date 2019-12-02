import dash
import dash_table
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = False

tab_selected_style = {
    'backgroundColor': '#119DFF',
    'color': 'white'
}

indata = pd.read_csv('input.txt')

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
                                    columns=[{"name": i, "id": i} for i in indata.columns],
                                    data=indata.to_dict('records'),
                            )
                            ]
                        ),
                        dcc.Tab(
                            label='Graphs',
                            selected_style=tab_selected_style
                        )
                    ]
                )
            ]
        ),
        html.Div(id='editor')
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
