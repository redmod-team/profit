import dash
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

with open('app.py', 'r') as f:
    text = f.read()

app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Tabs(
                    id='tabs',
                    value='tabs',
                    children=[
                        dcc.Tab(
                            label='Tab1',
                            selected_style=tab_selected_style,
                            children=[
                                dcc.Textarea(
                                    placeholder='Enter a value...',
                                    value=text,
                                    style={'width': '100%'}
                                )  
                            ]
                        ),
                        dcc.Tab(
                            label='Tab2',
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
