from abc import abstractmethod
import dash_bootstrap_components as dbc

from dash import Dash

from ...domain import MetaboController


class MetaTab:
    def __init__(self, app: Dash, metabo_controller: MetaboController):
        self.metabo_controller = metabo_controller
        self.app = app
        self._registerCallbacks()

    @abstractmethod
    def getLayout(self) -> dbc.Tab:
        pass

    @abstractmethod
    def _registerCallbacks(self) -> None:
        pass
