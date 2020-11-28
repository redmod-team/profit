using Dash
using DashHtmlComponents
using DashCoreComponents

app = dash(external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"])

app.layout = html_div() do
        html_h1("Hello Dash"),
        html_div("Dash.jl: Julia interface for Dash"),
        dcc_graph(
            id = "example-graph",
            figure = (
                data = [
                    (x = [1, 2, 3], y = [100, 1, 2], type = "bar", name = "SF"),
                    (x = [1, 2, 3], y = [2, 4, 5], type = "bar", name = "Montréal"),
                ],
                layout = (title = "Dash Data Visualization 2",)
            )
        )
    end

run_server(app, "0.0.0.0", 8080)
