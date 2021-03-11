import dash
import dash_table
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State, MATCH, ALL
from math import log10, floor
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
indata.rename(columns=lambda x: x.strip(), inplace=True)
outdata = pd.read_csv('output.txt', delim_whitespace=True, escapechar='#')
outdata.rename(columns=lambda x: x.strip(), inplace=True)

data = pd.concat([indata, outdata], 1)

invars = indata.keys()
invars_list = invars.to_list()
dropdown_opts = [{'label': invar, 'value': invar} for invar in invars]
# print(dropdown_opts)

app.layout = html.Div([
    html.Div([
        dcc.Tabs(id='tabs', children=[
            dcc.Tab(
                label='1D Graphs',
                selected_style=tab_selected_style,
                children=[
                    html.Div(dcc.Graph(id='graph1')),
                    html.Div(dcc.Dropdown(
                        id='invar',
                        options=dropdown_opts,
                        value=invars[0], )
                    ),
                    html.Div(html.Table(html.Tr([
                        html.Td(html.Button("Add Filter", id='add-filter', n_clicks=0)),
                        html.Td(html.Button("Clear Filter", id='clear-filter', n_clicks=0)),
                        html.Td(html.Button("Clear all Filter", id='clear-all-filter', n_clicks=0)),
                    ]))),
                    html.Div(html.Table(id='param-table', children=[
                        html.Thead(id='param-table-head', children=[
                            html.Tr(children=[
                                html.Th('Parameter', style={'width': 150}),
                                html.Th('Slider', style={'width': 300}),
                                html.Th('Range (min/max)'),
                                html.Th('center/span'),
                            ]),
                        ]),
                        html.Tbody(id='param-table-body', children=[
                            html.Tr(children=[
                                html.Td(html.Div(id='param-text-div', children=[])),
                                html.Td(html.Div(id='param-slider-div', children=[])),
                                html.Td(html.Div(id='param-range-div', children=[])),
                                html.Td(html.Div(id='param-center-div', children=[])),
                            ]),
                        ]),
                    ])),
                    # html.Div(['Specify range with slider or via text input (min/max or center/range) - #digits',
                    #     dcc.Input(id='num-digits', type='number', min=0, value=3),
                    #     html.Button(id='reset-button', children='reset slider', n_clicks=0)
                    # ]),
                    # html.Div(children=[
                    #     html.B('Range min/max: '),
                    #     html.I('Data-range: ('), html.I(id='graph1_min'), ' - ', html.I(id='graph1_max'),
                    #     html.I(')'),
                    #     html.Br(),
                    #     dcc.Input(id='range-min', type='number', placeholder='range min', step=0.001),
                    #     dcc.Input(id='range-max', type='number', placeholder='range max', step=0.001),
                    #     html.B('  center & span:'),
                    #     dcc.Input(id='center', type='number', placeholder='center', step=0.001),
                    #     dcc.Input(id='span', type='number', placeholder='span', step=0.001),
                    # ]),
                ]
            ),
            dcc.Tab(
                label='2D Graphs',
                selected_style=tab_selected_style,
                children=[
                    html.Div([
                        dcc.Dropdown(
                            id='invar1',
                            options=dropdown_opts,
                            value=invars[0]
                        ),
                        dcc.Dropdown(
                            id='invar2',
                            options=dropdown_opts,
                            value=invars[1]
                        ),
                    ]
                    ),
                    html.Div(dcc.Graph(id='graph2', style={'height': 600}))
                ]
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
        ])
    ]),
    html.Div(id='editor')
])


@app.callback(
    [Output('param-text-div', 'children'),
     Output('param-slider-div', 'children'),
     Output('param-range-div', 'children'),
     Output('param-center-div', 'children'),],
    [Input('add-filter', 'n_clicks'),
     Input('clear-filter', 'n_clicks'),
     Input('clear-all-filter', 'n_clicks')],
    [State('invar', 'value'),
     State('param-text-div', 'children'),
     State('param-slider-div', 'children'),
     State('param-range-div', 'children'),
     State('param-center-div', 'children'), ],
)
def add_filterrow(n_clicks, clear, clear_all, invar, text, slider, range_div, center_div):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == 'clear-all-filter':
        return [], [], [], []
    elif trigger_id == 'clear-filter':
        for i, element in enumerate(text):  # TODO: better names
            if text[i]['props']['children'][0] == invar:
                text.pop(i)
                slider.pop(i)
                range_div.pop(i)
                center_div.pop(i)
    elif trigger_id == 'add-filter':  # TODO: avoid double usage of filter
        for i, element in enumerate(text):
            if text[i]['props']['children'][0] == invar:
                return text, slider, range_div, center_div
        ind = invars.to_list().index(invar)
        txt = invar
        new_text = html.Div(id={'type': 'dyn-text', 'index': ind}, children=[txt], style={'height': 50})
        new_slider = html.Div(id={'type': 'dyn-slider', 'index': ind}, style={'height': 50}, children=[
            create_slider(txt)], )
        new_range = html.Div(id={'type': 'dyn-range', 'index': ind}, style={'height': 50}, children=[
            dcc.Input(id={'type': 'param-range-min', 'index': ind}, type='number', placeholder='range min'),
            dcc.Input(id={'type': 'param-range-max', 'index': ind}, type='number', placeholder='range max'),
        ], )
        new_center = html.Div(id={'type': 'dyn-center', 'index': ind}, style={'height': 50}, children=[
            dcc.Input(id={'type': 'param-center', 'index': ind}, type='number', placeholder='center'),
            dcc.Input(id={'type': 'param-span', 'index': ind}, type='number', placeholder='span'),
        ], )
        text.append(new_text)
        slider.append(new_slider)
        range_div.append(new_range)
        center_div.append(new_center)
    return text, slider, range_div, center_div


@app.callback(
    [Output({'type': 'param-range-min', 'index': MATCH}, 'step'),
     Output({'type': 'param-range-max', 'index': MATCH}, 'step'),
     Output({'type': 'param-center', 'index': MATCH}, 'step'),
     Output({'type': 'param-span', 'index': MATCH}, 'step'), ],
    Input({'type': 'param-slider', 'index': MATCH}, 'step')
)
def update_step(step):
    return step, step, step, step


@app.callback(
    [Output({'type': 'param-range-min', 'index': MATCH}, 'value'),
     Output({'type': 'param-range-max', 'index': MATCH}, 'value'),
     Output({'type': 'param-slider', 'index': MATCH}, 'value'),
     Output({'type': 'param-center', 'index': MATCH}, 'value'),
     Output({'type': 'param-span', 'index': MATCH}, 'value'), ],
    [Input({'type': 'param-range-min', 'index': MATCH}, 'value'),
     Input({'type': 'param-range-max', 'index': MATCH}, 'value'),
     Input({'type': 'param-slider', 'index': MATCH}, 'value'),
     Input({'type': 'param-center', 'index': MATCH}, 'value'),
     Input({'type': 'param-span', 'index': MATCH}, 'value'), ],
    State({'type': 'param-slider', 'index': MATCH}, 'step')
)
def update_dyn_slider_range(dyn_min, dyn_max, slider_val, center, span, step):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split('}')[0].split(',')[1].split(':')[1]
    # TODO: search in str instead of split
    if trigger_id == '"param-center"' or trigger_id == '"param-span"' and (center and span):
        # print('center')
        dyn_min = center - span
        dyn_max = center + span
        slider_val = [dyn_min, dyn_max]
    elif (trigger_id == '"param-range-min"' or trigger_id == '"param-range-max"') and (dyn_min is not None and dyn_max is not None):
        # print('range')
        # print('min:', dyn_min, 'max:', dyn_max)
        slider_val = [dyn_min, dyn_max]
        span = (slider_val[1] - slider_val[0]) / 2
        center = slider_val[0] + span
    elif slider_val:
        # print('else')
        dyn_min = slider_val[0]
        dyn_max = slider_val[1]
        span = (slider_val[1] - slider_val[0])/2
        center = slider_val[0] + span
    # rounding based on stepsize of slider
    dig = int(-log10(step))
    slider_val = [round(slider_val[0], dig), round(slider_val[1], dig)]
    return round(dyn_min, dig), round(dyn_max, dig), slider_val, round(center, dig), round(span, dig)


def create_slider(dd_value):
    ind = invars.to_list().index(dd_value)
    slider_min = indata[dd_value].min()
    slider_max = indata[dd_value].max()
    step_exponent = floor(log10((slider_max - slider_min) / 100))
    while slider_max / (10**step_exponent) > 1000:
        step_exponent = step_exponent+1
    while (slider_max - slider_min) / (10 ** step_exponent) < 20:  # minimum of 20 steps per slider
        step_exponent = step_exponent - 1
    new_slider = dcc.RangeSlider(
        id={'type': 'param-slider', 'index': ind},
        step=10 ** step_exponent,  # floor and log10 from package `math`
        min=slider_min,
        max=slider_max,
        value=[slider_min, slider_max],
        marks={slider_min: str(round(slider_min, -step_exponent)),
               slider_max: str(round(slider_max, -step_exponent))},
    )
    return new_slider


@app.callback(
    Output('graph1', 'figure'),
    [Input('tabs', 'value'),
     Input('invar', 'value'),
     Input({'type': 'param-slider', 'index': ALL}, 'value'), ],
    State({'type': 'param-slider', 'index': ALL}, 'id'),
)
def update_figure(tab, invar, param_slider, id):
    if invar is None:
        return go.Figure()
    sel_y = np.full((len(outdata), ), True)
    for iteration, values in enumerate(param_slider):
        dds_value = invars_list[id[iteration]['index']]
        # filter for minimum
        sel_y_min = np.array(indata[dds_value] >= param_slider[iteration][0])
        # filter for maximum
        sel_y_max = np.array(indata[dds_value] <= param_slider[iteration][1])
        sel_y = sel_y_min & sel_y_max & sel_y
    fig = go.Figure(
        data=[go.Scatter(
            x=indata[invar].iloc[sel_y],
            y=outdata.iloc[sel_y, 0],
            mode='markers',
        )],
        layout=go.Layout(scene=dict(xaxis_title=invar), )
    )
    fig.update_xaxes(rangeslider=dict(visible=True, ), title=invar, )
    return fig


@app.callback(
    dash.dependencies.Output('graph2', 'figure'),
    [dash.dependencies.Input('tabs', 'value'),
     dash.dependencies.Input('invar1', 'value'),
     dash.dependencies.Input('invar2', 'value'), ]
)
def update_figure2(tab, invar1, invar2):
    if invar1 is None or invar2 is None:
        return go.Figure()
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=indata[invar1],
                y=indata[invar2],
                z=outdata.values[:, 0],
                mode='markers'
            )
        ],
        layout=go.Layout(
            scene=dict(
                xaxis_title=invar1,
                yaxis_title=invar2
            )
        )
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
