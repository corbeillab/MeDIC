import dash_bootstrap_components as dbc
from dash import html

from .MetaTab import MetaTab


class InterpretTab(MetaTab):
    def getLayout(self) -> dbc.Tab:
        return dbc.Tab(className="global_tab",
                       # tab_style={"margin-left": "auto"},
                       label="Model Interpretation",
                       children=[html.P("A tab to allow model interpretation with model-agnostic methods.")])

    def _registerCallbacks(self) -> None:
        pass
    