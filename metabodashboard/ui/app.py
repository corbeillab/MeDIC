from dash import html
from dash.dependencies import Input, Output
import dash
import dash_bootstrap_components as dbc
import os
from ..service import Utils

from .tabs import *
from ..domain import MetaboController


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)
app.title = "MeDIC"
server = app.server
# app.scripts.config.serve_locally = False
app.css.config.serve_locally = False
app.config.suppress_callback_exceptions = True
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

metabo_controller = MetaboController()
infoTab = InfoTab(app, metabo_controller)
splitsTab = SplitsTab(app, metabo_controller)
mLTab = MLTab(app, metabo_controller)
resultsTab = ResultsTab(app, metabo_controller)
resultsSummaryTab = ResultsSummaryTab(app, metabo_controller)
interpretTab = InterpretTab(app, metabo_controller)


app.layout = html.Div(
    id="page",
    children=[
        html.Div(id="dataCache", children=[], style={"display": "none"}),
        html.Div(
            id="title_container",
            className="row",
            children=[
                html.Div(
                    children=[
                        html.H1(id="title", children="MeDIC"),
                        html.Div(
                            children=[
                                html.H6(
                                    "Metabolomics Dashboard", style={"color": "white"}
                                ),
                                html.H6(
                                    "for Interpretable Classification",
                                    style={"color": "white"},
                                ),
                            ],
                            id="acronym",
                        ),
                    ],
                    id="title_bg",
                )
            ],
        ),
        html.Div(
            id="main-content",
            children=[
                dbc.Tabs(
                    id="custom_big_tabs",
                    persistence=True,
                    className="global_tabs_container",
                    children=[
                        infoTab.getLayout(),
                        splitsTab.getLayout(),
                        mLTab.getLayout(),
                        resultsTab.getLayout(),
                        resultsSummaryTab.getLayout()
                        # interpretTab.getLayout()
                    ],
                )
            ],
        ),
    ],
)
