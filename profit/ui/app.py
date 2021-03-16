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

app.layout = html.Div(children=[
    html.Div(dcc.Graph(id='graph1')),
    html.Div(dcc.RadioItems(
        id='graph-type',
        options=[{'label': i, 'value': i} for i in ['1D scatter',
                                                    '2D scatter',
                                                    '2D contour']
                 ],
        value='1D scatter',
        labelStyle={'display': 'inline-block'})),
    html.Div(id='invar-1-div', style={'display': 'flex', 'align-items': 'baseline'}, children=[
        html.B('x: ', style={'width': 100}),
        dcc.Dropdown(
            id='invar',
            options=dropdown_opts,
            value=invars[0],
            style={'width': 700},
        ),
    ]),
    html.Div(id='invar-2-div', style={'display': 'flex', 'align-items': 'baseline'}, children=[
        html.B('y: ', style={'width': 100}),
        dcc.Dropdown(
            id='invar_2',
            options=dropdown_opts,
            value=invars[0],
            style={'width': 700},
        ),
    ]),
    html.Div(id='color-div', style={'display': 'flex', 'align-items': 'baseline'}, children=[
        html.B("color: ", style={'width': 100}),
        dcc.Dropdown(
            id='color-dropdown',
            options=dropdown_opts,
            value=invars[0],
            style={'width': 700},
        ),
        html.B("use color:", style={'width': 100, 'text-align': 'center'}),
        dcc.RadioItems(
            id='color-use',
            value='false',
            options=[
                {'label': 'True', 'value': 'true'},
                {'label': 'False', 'value': 'false'}
            ],
            labelStyle={'display': 'inline-block'},
        ),
        html.I("(for scatter plot only)", style={'width': 150, 'text-align': 'right'})
    ]),
    html.Div(html.Table(html.Tr([
        html.Td(html.Button("Add Filter", id='add-filter', n_clicks=0)),
        html.Td(html.Button("Clear Filter", id='clear-filter', n_clicks=0)),
        html.Td(html.Button("Clear all Filter", id='clear-all-filter', n_clicks=0)),
        html.Td(dcc.Slider(id='scale-slider',
                           min=-0.5, max=0.5,
                           value=0, step=0.01,
                           marks={-1: '-100%',
                                  -0.75: '-75%',
                                  -0.5: '-50%',
                                  -0.25: '-25%',
                                  0: '0%',
                                  0.25: '25%',
                                  0.5: '50%',
                                  0.75: '75%',
                                  1: '100%'}
                           ),
                style={'width': 500}),
        html.Td(html.Button("Scale", id='scale', n_clicks=0)),
    ]))),
    html.Div(html.Table(id='param-table', children=[
        html.Thead(id='param-table-head', children=[
            html.Tr(children=[
                html.Th("Parameter", style={'width': 150}),
                html.Th("Slider", style={'width': 300}),
                html.Th("Range (min/max)"),
                html.Th("center/span"),
                html.Th("filter active"),
            ]),
        ]),
        html.Tbody(id='param-table-body', children=[
            html.Tr(children=[
                html.Td(html.Div(id='param-text-div', children=[])),
                html.Td(html.Div(id='param-slider-div', children=[])),
                html.Td(html.Div(id='param-range-div', children=[])),
                html.Td(html.Div(id='param-center-div', children=[])),
                html.Td(html.Div(id='param-active-div', children=[])),
            ]),
        ]),
    ])),
    html.Div([
        html.Div(children=[
            html.B("Show table of data:"),
            html.Button("show table", id='show-table', n_clicks=0),
            html.Button("hide table", id='hide-table', n_clicks=0),
        ]),
        html.Div(id='data-table-div', style={'visibility': 'hidden'}, children=[
            dash_table.DataTable(
                id='data-table',
                columns=[{"name": i, "id": i} for i in data.columns],
                data=data.to_dict('records'),
                page_size=20,
            )
        ])
    ]),
])


@app.callback(
    [Output('param-text-div', 'children'),
     Output('param-slider-div', 'children'),
     Output('param-range-div', 'children'),
     Output('param-center-div', 'children'),
     Output('param-active-div', 'children'), ],
    [Input('add-filter', 'n_clicks'),
     Input('clear-filter', 'n_clicks'),
     Input('clear-all-filter', 'n_clicks')],
    [State('invar', 'value'),
     State('param-text-div', 'children'),
     State('param-slider-div', 'children'),
     State('param-range-div', 'children'),
     State('param-center-div', 'children'),
     State('param-active-div', 'children'), ],
)
def add_filterrow(n_clicks, clear, clear_all, invar, text, slider, range_div, center_div, active_div):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == 'clear-all-filter':
        return [], [], [], [], []
    elif trigger_id == 'clear-filter':
        for i, element in enumerate(text):  # TODO: better names
            if text[i]['props']['children'][0] == invar:
                text.pop(i)
                slider.pop(i)
                range_div.pop(i)
                center_div.pop(i)
                active_div.pop(i)
    elif trigger_id == 'add-filter':  # TODO: avoid double usage of filter
        for i, element in enumerate(text):
            if text[i]['props']['children'][0] == invar:
                return text, slider, range_div, center_div, active_div
        ind = invars.to_list().index(invar)
        txt = invar
        new_text = html.Div(id={'type': 'dyn-text', 'index': ind}, children=[txt], style={'height': 40})
        new_slider = html.Div(id={'type': 'dyn-slider', 'index': ind}, style={'height': 40}, children=[
            create_slider(txt)], )
        new_range = html.Div(id={'type': 'dyn-range', 'index': ind}, style={'height': 40}, children=[
            dcc.Input(id={'type': 'param-range-min', 'index': ind}, type='number', placeholder='range min'),
            dcc.Input(id={'type': 'param-range-max', 'index': ind}, type='number', placeholder='range max'),
        ], )
        new_center = html.Div(id={'type': 'dyn-center', 'index': ind}, style={'height': 40}, children=[
            dcc.Input(id={'type': 'param-center', 'index': ind}, type='number', placeholder='center'),
            dcc.Input(id={'type': 'param-span', 'index': ind}, type='number', placeholder='span'),
        ], )
        new_active = html.Div(id={'type': 'dyn-active', 'index': ind},
                              style={'height': 40, 'text-align': 'center'},
                              children=[
                                  dcc.Checklist(id={'type': 'param-active', 'index': ind},
                                                options=[{'label': '', 'value': 'act'}],
                                                value=['act'],
                                                )
                              ])
        text.append(new_text)
        slider.append(new_slider)
        range_div.append(new_range)
        center_div.append(new_center)
        active_div.append(new_active)
    return text, slider, range_div, center_div, active_div


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
     Input({'type': 'param-span', 'index': MATCH}, 'value'),
     Input('scale', 'n_clicks'), ],
    [State({'type': 'param-slider', 'index': MATCH}, 'step'),
     State('scale-slider', 'value'), ]
)
def update_dyn_slider_range(dyn_min, dyn_max, slider_val, center, span, scale, step, scale_slider):
    ctx = dash.callback_context
    # print(ctx.triggered[0]["prop_id"])
    if ctx.triggered[0]["prop_id"] == "scale.n_clicks":
        span = span * (1 + scale_slider)
        dyn_min = center - span
        dyn_max = center + span
        slider_val = [dyn_min, dyn_max]
    else:
        trigger_id = ctx.triggered[0]["prop_id"].split('}')[0].split(',')[1].split(':')[1]
        # TODO: search in str instead of split
        if trigger_id == '"param-center"' or trigger_id == '"param-span"' and (center and span):
            # print('center')
            dyn_min = center - span
            dyn_max = center + span
            slider_val = [dyn_min, dyn_max]
        elif (trigger_id == '"param-range-min"' or trigger_id == '"param-range-max"') and (
                dyn_min is not None and dyn_max is not None):
            # print('range')
            # print('min:', dyn_min, 'max:', dyn_max)
            slider_val = [dyn_min, dyn_max]
            span = (slider_val[1] - slider_val[0]) / 2
            center = slider_val[0] + span
        elif slider_val:
            # print('else')
            dyn_min = slider_val[0]
            dyn_max = slider_val[1]
            span = (slider_val[1] - slider_val[0]) / 2
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
    while slider_max / (10 ** step_exponent) > 1000:
        step_exponent = step_exponent + 1
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
    [Input('invar', 'value'),
     Input('invar_2', 'value'),
     Input({'type': 'param-slider', 'index': ALL}, 'value'),
     Input('graph-type', 'value'),
     Input('color-use', 'value'),
     Input('color-dropdown', 'value'),
     Input({'type': 'param-active', 'index': ALL}, 'value'), ],
    [State({'type': 'param-slider', 'index': ALL}, 'id'), ],
)
def update_figure(invar, invar_2, param_slider, graph_type, color_use, color_dd, filter_active, id_type):
    if invar is None:
        return go.Figure()
    sel_y = np.full((len(outdata),), True)
    for iteration, values in enumerate(param_slider):
        dds_value = invars_list[id_type[iteration]['index']]
        # filter for minimum
        sel_y_min = np.array(indata[dds_value] >= param_slider[iteration][0])
        # filter for maximum
        sel_y_max = np.array(indata[dds_value] <= param_slider[iteration][1])
        # print('iter ', iteration, 'filer', filter_active[iteration][0])
        if filter_active != [[]]:
            if filter_active[iteration] == ['act']:
                sel_y = sel_y_min & sel_y_max & sel_y
    if graph_type == '1D scatter':
        fig = go.Figure(
            data=[go.Scatter(
                x=indata[invar].iloc[sel_y],
                y=outdata.iloc[sel_y, 0],
                mode='markers',
            )],
            layout=go.Layout(scene=dict(xaxis_title=invar), )
        )
        fig.update_xaxes(rangeslider=dict(visible=True, ), title=invar, )
        # fig.update_traces()
    elif graph_type == '2D scatter':
        fig = go.Figure(
            data=[go.Scatter3d(
                x=indata[invar].iloc[sel_y],
                y=indata[invar_2].iloc[sel_y],
                z=outdata.iloc[sel_y, 0],
                mode='markers',
            )],
            layout=go.Layout(scene=dict(xaxis_title=invar, yaxis_title=invar_2), )
        )
    elif graph_type == '2D contour':
        fig = go.Figure(
            data=go.Contour(
                x=indata[invar].iloc[sel_y],
                y=indata[invar_2].iloc[sel_y],
                z=outdata.iloc[sel_y, 0],
            ),
        )
        fig.update_xaxes(title=invar)
        fig.update_yaxes(title=invar_2)
    else:
        fig = go.Figure(
            data=go.Surface(
                x=indata[invar].iloc[sel_y],
                y=indata[invar_2].iloc[sel_y],
                z=outdata.iloc[sel_y, 0],
            ),
        )
        fig.update_xaxes(title=invar)
        fig.update_yaxes(title=invar_2)
    if color_use == 'true':
        fig.update_traces(marker=dict(
            color=indata[color_dd].iloc[sel_y],
            colorscale='Viridis',
            colorbar=dict(thickness=20, title=color_dd),
        ))
    return fig


@app.callback(
    Output('data-table-div', 'style'),
    [Input('show-table', 'n_clicks'),
     Input('hide-table', 'n_clicks'), ]
)
def show_table(show, hide):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    print(trigger_id)
    if trigger_id == 'show-table':
        return {'visibility': 'visible'}
    else:
        return {'visibility': 'hidden'}




if __name__ == '__main__':
    app.run_server(debug=True)
