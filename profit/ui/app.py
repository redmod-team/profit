import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State, MATCH, ALL
from math import log10, floor
import numpy as np
from profit.util import load
from profit.sur import Surrogate
from matplotlib import cm as colormaps
from matplotlib.colors import to_hex as color2hex

def init_app(config):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    external_scripts = ['https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=external_scripts)
    server = app.server
    app.config.suppress_callback_exceptions = False

    indata = load(config['files']['input']).flatten()
    outdata = load(config['files']['output']).flatten()

    # data = pd.concat([indata, outdata], 1) # TODO: data in table

    invars = indata.dtype.names
    outvars = outdata.dtype.names
    dropdown_opts_in = [{'label': invar, 'value': invar} for invar in invars]
    dropdown_opts_out = [{'label': outvar, 'value': outvar} for outvar in outvars]
    dropdown_style = {'width': 500}
    axis_options_text_style = {'width':100}
    axis_options_log_style = {'width': 50}
    axis_options_div_style = {'display': 'flex', 'align-items': 'baseline'}
    fit_options_text_style = {'width': 150}

    def colormap(cmin, cmax, c):
        if cmin == cmax:
            c_scal = 0.5
        else:
            c_scal = (c-cmin)/(cmax-cmin)
        return color2hex(colormaps.viridis(c_scal))

    app.layout = html.Div(children=[
        html.Div(dcc.Graph(id='graph1')),
        html.Div(dcc.RadioItems(
            id='graph-type',
            options=[{'label': i, 'value': i} for i in ['1D',
                                                        '2D',
                                                        '2D contour',
                                                        '3D']
                     ],
            value='1D',
            labelStyle={'display': 'inline-block'})),
        html.Table(children=[html.Tr(children=[
            html.Td(id='axis-options', style={'width': 650}, children=[
                html.Div(id='invar-1-div', style=axis_options_div_style, children=[
                    html.B('x: ', style=axis_options_text_style),
                    dcc.Dropdown(
                        id='invar',
                        options=dropdown_opts_in,
                        value=invars[0],
                        style=dropdown_style,
                    ),
                    dcc.Checklist(
                        id='invar-1-log',
                        options=[{'label': 'log', 'value': 'log'}],
                        style=axis_options_log_style,
                    ),
                ]),
                html.Div(id='invar-2-div', style=axis_options_div_style, children=[
                    html.B('y: ', style=axis_options_text_style),
                    dcc.Dropdown(
                        id='invar_2',
                        options=dropdown_opts_in,
                        value=invars[1] if len(invars) > 1 else invars[0],
                        style=dropdown_style,
                    ),
                    dcc.Checklist(
                        id='invar-2-log',
                        options=[{'label': 'log', 'value': 'log'}],
                        style=axis_options_log_style,
                    ),
                ]),
                html.Div(id='invar-3-div', style=axis_options_div_style, children=[
                    html.B('z: ', style=axis_options_text_style),
                    dcc.Dropdown(
                        id='invar_3',
                        options=dropdown_opts_in,
                        value=invars[2] if len(invars) > 2 else invars[0],
                        style=dropdown_style,
                    ),
                    dcc.Checklist(
                        id='invar-3-log',
                        options=[{'label': 'log', 'value': 'log'}],
                        style=axis_options_log_style,
                    ),
                ]),
                html.Div(id='outvar-div', style=axis_options_div_style, children=[
                    html.B('output: ', style=axis_options_text_style),
                    dcc.Dropdown(
                        id='outvar',
                        options=dropdown_opts_out,
                        value=outvars[0],
                        style=dropdown_style,
                    ),
                    dcc.Checklist(
                        id='outvar-log',
                        options=[{'label': 'log', 'value': 'log'}],
                        style=axis_options_log_style,
                    ),
                ]),
                html.Div(id='color-div', style=axis_options_div_style, children=[
                    html.B("color: ", style={'width': 50}),
                    dcc.Checklist(
                        id='color-use',
                        options=[{'label': '', 'value': 'true'}],
                        style={'width': 50},
                    ),
                    dcc.Dropdown(
                        id='color-dropdown',
                        options=dropdown_opts_in + dropdown_opts_out,
                        value=invars[2] if len(invars) > 2 else invars[0],
                        style=dropdown_style,
                    ),
                ]),
            ]),
            html.Td(id='fit-options', style={'vertical-align': 'top'}, children=[
                html.Div(id='fit-use-div', style=axis_options_div_style, children=[
                    html.B("display fit:", style=fit_options_text_style),
                    dcc.Checklist(
                        id='fit-use',
                        options=[{'label': '', 'value': 'show'}],
                        labelStyle={'display': 'inline-block'},
                    ),
                ]),
                html.Div(id='fit-multiinput-div', style=axis_options_div_style, children=[
                    html.B("variable for multi-fit:", style=fit_options_text_style),
                    dcc.Dropdown(
                        id='fit-multiinput-dropdown',
                        options=dropdown_opts_in,
                        value=invars[1] if len(invars) > 1 else invars[0],
                        style=dropdown_style,
                    ),
                ]),
                html.Div(id='fit-number-div', style=axis_options_div_style, children=[
                    html.B("number of fits:", style=fit_options_text_style),
                    dcc.Input(id='fit-number', type='number', value=1),
                ]),
                html.Div(id='fit-conf-div', style=axis_options_div_style, children=[
                    html.B("\u03c3-confidence:", style=fit_options_text_style),
                    dcc.Input(id='fit-conf', type='number', value=2, min=0),
                ]),
            ]),
        ])]),
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
            html.Td(html.Button("Scale Filter span", id='scale', n_clicks=0)),
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
                # dash_table.DataTable(
                #     id='data-table',
                #     columns=[{"name": i, "id": i} for i in invars],
                #     data=indata.to_dict('records'),
                #     page_size=20,
                # ) # TODO: fix table
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
            ind = invars.index(invar)
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
        ind = invars.index(dd_value)
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
        [Output('invar-3-div', 'style'),
         Output('color-div', 'style'),
         Output('fit-multiinput-div', 'style'),
         Output('fit-conf-div', 'style'), ],
        Input('graph-type', 'value')
    )
    def div_visibility(graph_type):
        hidden_div_style = axis_options_div_style.copy()
        show_div_style = axis_options_div_style.copy()
        if graph_type == '3D':
            hidden_div_style['visibility'] = 'hidden'
            show_div_style['visibility'] = 'visible'
        else:
            hidden_div_style['visibility'] = 'visible'
            show_div_style['visibility'] = 'hidden'
        return show_div_style, hidden_div_style, hidden_div_style, hidden_div_style


    @app.callback(
        Output('graph1', 'figure'),
        [Input('invar', 'value'),
         Input('invar_2', 'value'),
         Input('invar_3', 'value'),
         Input('outvar', 'value'),
         Input('invar-1-log', 'value'),
         Input('invar-2-log', 'value'),
         Input('invar-3-log', 'value'),
         Input('outvar-log', 'value'),
         Input({'type': 'param-slider', 'index': ALL}, 'value'),
         Input('graph-type', 'value'),
         Input('color-use', 'value'),
         Input('color-dropdown', 'value'),
         Input({'type': 'param-active', 'index': ALL}, 'value'),
         Input('fit-use', 'value'),
         Input('fit-multiinput-dropdown', 'value'),
         Input('fit-number', 'value'),
         Input('fit-conf', 'value'), ],
        [State({'type': 'param-slider', 'index': ALL}, 'id'),
         State({'type': 'param-center', 'index': ALL}, 'value')],
    )
    def update_figure(invar, invar_2, invar_3, outvar, invar1_log, invar2_log, invar3_log, outvar_log, param_slider, graph_type, color_use, color_dd, filter_active, fit_use, fit_dd, fit_num, fit_conf, id_type, param_center):
        if invar is None:
            return go.Figure()
        sel_y = np.full((len(outdata),), True)
        for iteration, values in enumerate(param_slider):
            dds_value = invars[id_type[iteration]['index']]
            # filter for minimum
            sel_y_min = np.array(indata[dds_value] >= param_slider[iteration][0])
            # filter for maximum
            sel_y_max = np.array(indata[dds_value] <= param_slider[iteration][1])
            # print('iter ', iteration, 'filer', filter_active[iteration][0])
            if filter_active != [[]]:
                if filter_active[iteration] == ['act']:
                    sel_y = sel_y_min & sel_y_max & sel_y
        if graph_type == '1D':
            fig = go.Figure(
                data=[go.Scatter(
                    x=indata[invar][sel_y],
                    y=outdata[outvar][sel_y],
                    mode='markers',
                    name='data',
                )],
                layout=go.Layout(height=600, xaxis=dict(title=invar, rangeslider=dict(visible=True)), yaxis=dict(title=outvar))
            )
            if fit_use == ['show']:
                num_samples = 20
                mesh_in, mesh_out, mesh_out_std, fit_dd_values = mesh_fit(param_slider, id_type, fit_dd, fit_num,
                                                                          param_center, [invar], outvar, num_samples)
                for i in range(len(fit_dd_values)):
                    fig.add_trace(go.Scatter(
                        x=mesh_in[i][0],
                        y=mesh_out[i],
                        mode='lines',
                        name=f'fit: {fit_dd}={fit_dd_values[i]:.2f}',
                        line_color=colormap(fit_dd_values[0], fit_dd_values[-1], fit_dd_values[i]),
                    ))
                    fig.add_trace(go.Scatter(
                        x=np.hstack((mesh_in[i][0], mesh_in[i][0][::-1])),
                        y=np.hstack((mesh_out[i] + fit_conf * mesh_out_std[i], mesh_out[i][::-1] - fit_conf * mesh_out_std[i][::-1])),
                        showlegend=False,
                        fill='toself',
                        line_color=colormap(fit_dd_values[0], fit_dd_values[-1], fit_dd_values[i]),
                    ))
        elif graph_type == '2D':
            fig = go.Figure(
                data=[go.Scatter3d(
                    x=indata[invar][sel_y],
                    y=indata[invar_2][sel_y],
                    z=outdata[outvar][sel_y],
                    mode='markers',
                    name='Data',
                )],
                layout=go.Layout(scene=dict(xaxis_title=invar, yaxis_title=invar_2, zaxis_title=outvar))
            )
            fig.update_layout(height=700)
            if fit_use == ['show'] and invar != invar_2:
                num_samples = 20
                mesh_in, mesh_out, mesh_out_std, fit_dd_values = mesh_fit(param_slider, id_type, fit_dd, fit_num,
                                                                          param_center, [invar, invar_2], outvar,
                                                                          num_samples)
                for i in range(len(fit_dd_values)):
                    fig.add_trace(go.Surface(
                        x=mesh_in[i][0].reshape((num_samples, num_samples)),
                        y=mesh_in[i][1].reshape((num_samples, num_samples)),
                        z=mesh_out[i].reshape((num_samples, num_samples)),
                        name=f'fit: {fit_dd}={fit_dd_values[i]:.2f}',
                        coloraxis="coloraxis",
                    ))
                    if fit_conf > 0:
                        fig.add_trace(go.Surface(
                            x=mesh_in[i][0].reshape((num_samples, num_samples)),
                            y=mesh_in[i][1].reshape((num_samples, num_samples)),
                            z=mesh_out[i].reshape((num_samples, num_samples)) + fit_conf * mesh_out_std[i].reshape((num_samples, num_samples)),
                            showlegend=False,
                            name=f'fit+var: {fit_dd}={fit_dd_values[i]:.2f}',
                            opacity=0.25,
                            coloraxis="coloraxis",
                        ))
                        fig.add_trace(go.Surface(
                            x=mesh_in[i][0].reshape((num_samples, num_samples)),
                            y=mesh_in[i][1].reshape((num_samples, num_samples)),
                            z=mesh_out[i].reshape((num_samples, num_samples)) - fit_conf * mesh_out_std[i].reshape((num_samples, num_samples)),
                            showlegend=False,
                            name=f'fit-var: {fit_dd}={fit_dd_values[i]:.2f}',
                            opacity=0.5,
                            coloraxis="coloraxis",
                        ))
        elif graph_type == '2D contour':
            fig = go.Figure(
                data=go.Contour(
                    x=indata[invar][sel_y],
                    y=indata[invar_2][sel_y],
                    z=outdata[outvar][sel_y],
                ),
                layout=go.Layout(xaxis=dict(title=invar), yaxis=dict(title=invar_2))
            )
        elif graph_type == '3D':
            fig = go.Figure(
                data=go.Scatter3d(
                    x=indata[invar][sel_y],
                    y=indata[invar_2][sel_y],
                    z=indata[invar_3][sel_y],
                    mode='markers',
                    marker=dict(
                            color=outdata[outvar][sel_y],
                            coloraxis="coloraxis",
                        ),
                ),
                layout=go.Layout(scene=dict(xaxis_title=invar, yaxis_title=invar_2, zaxis_title=invar_3)),
            )
            if fit_use == ['show'] and len({invar, invar_2, invar_3}) == 3:
                num_samples = 20
                mesh_in, mesh_out, mesh_out_std, fit_dd_values = mesh_fit(param_slider, id_type, fit_dd, fit_num,
                                                                          param_center, [invar, invar_2, invar_3], outvar,
                                                                          num_samples)
                for i in range(len(fit_dd_values)):
                    print(mesh_in.shape)
                    print(mesh_in[0][0].shape)
                    print(mesh_in[0][0].flatten().shape)
                    print(type(mesh_out[i]))
                    print('min', mesh_out[i].min())
                    fig.add_trace(
                        go.Isosurface(
                            x=mesh_in[i][0],
                            y=mesh_in[i][1],
                            z=mesh_in[i][2],
                            value=mesh_out[i],
                            surface_count=fit_num,
                            coloraxis="coloraxis",
                            isomin=mesh_out[i].min() * 1.1,
                            isomax=mesh_out[i].max() * 0.9,
                            caps=dict(x_show=False, y_show=False, z_show=False),
                        ),
                    )
        else:
            fig = go.Figure()
        if color_use == ['true']: # TODO: trigger-detection no new fig just update
            fig.update_traces(
                marker=dict(
                    color=indata[color_dd][sel_y] if color_dd in indata.dtype.names else outdata[color_dd][sel_y],
                    coloraxis="coloraxis" if graph_type == '2D' and color_dd == outvar else None,
                    colorbar=dict(thickness=20, title=color_dd),
                ),
                selector=dict(mode='markers'),
            )
        # log scale
        log_dict = {'1D': (invar1_log, outvar_log),
                    '2D': (invar1_log, invar2_log, outvar_log),
                    '2D contour': (invar1_log, invar2_log),
                    '3D': (invar1_log, invar2_log, invar3_log),}
        log_list = ['linear' if log is None or len(log) == 0 else log[0] for log in log_dict[graph_type]]
        log_key = ['xaxis', 'yaxis', 'zaxis']
        comb_dict = dict(zip(log_key, [{'type': log} for log in log_list]))
        if len(log_list) < 3 :
            fig.update_layout(**comb_dict)
        else:
            fig.update_scenes(**comb_dict)
        return fig


    def mesh_fit(param_slider, id_type, fit_dd, fit_num, param_center, invar_list, outvar, num_samples):
        try:  # collecting min/max of slider in filter section
            fit_dd_min, fit_dd_max = param_slider[[i['index'] for i in id_type].index(invars.index(fit_dd))]
        except ValueError:
            fit_dd_min = min(indata[fit_dd])
            fit_dd_max = max(indata[fit_dd])
        if fit_num == 1:
            fit_dd_values = np.array([(fit_dd_max + fit_dd_min) / 2])
        else:
            fit_dd_values = np.linspace(fit_dd_min, fit_dd_max, fit_num)
        mesh_out = tuple()
        mesh_out_std = tuple()
        for iteration, fit_dd_value in enumerate(fit_dd_values):
            fit_params = [(max(indata[var_invar]) + min(indata[var_invar])) / 2 for var_invar in invars]
            for iteration, center_values in enumerate(param_center):
                ind = id_type[iteration]['index']
                fit_params[ind] = param_center[iteration]
            fit_params[invars.index(fit_dd)] = fit_dd_value
            for invar in invar_list:
                fit_params[invars.index(invar)] = np.linspace(min(indata[invar]), max(indata[invar]), num_samples)
            grid = np.meshgrid(*fit_params)
            x_pred = np.vstack([g.flatten() for g in grid]).T  # extract vector for predict
            sur = Surrogate.load_model(config['fit']['save']) # load surrogate
            fit_data, fit_var = sur.predict(x_pred)
            # generated data
            new_mesh_in = np.array([[grid[invars.index(invar)].flatten() for invar in invars]])
            new_mesh_out = np.array([fit_data[:, outvars.index(outvar)]])
            new_mesh_out_std = np.array([np.sqrt(fit_var[:, outvars.index(outvar)])])
            if iteration == 0:
                mesh_in = new_mesh_in
                mesh_out = new_mesh_out
                mesh_out_std = new_mesh_out_std
            else:
                mesh_in = np.vstack((mesh_in, new_mesh_in))
                mesh_out = np.vstack((mesh_out, new_mesh_out))
                mesh_out_std = np.vstack((mesh_out_std, new_mesh_out_std))
        return mesh_in, mesh_out, mesh_out_std, fit_dd_values

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

    return app
