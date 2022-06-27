import glob
import json
import os
import time
from collections import Counter

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import html, dcc, Output, Input, State, dash, Dash
import plotly.graph_objs as go
import pickle as pkl
from matplotlib import pyplot as plt
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from .MetaTab import MetaTab
from ...service import Plots
from ...domain import MetaboController


class ResultsSummaryTab(MetaTab):
    def __init__(self, app: Dash, metabo_controller: MetaboController):
        super().__init__(app, metabo_controller)
        self.r = self.metabo_controller.get_all_results()
        self._plots = Plots("blues")

    def getLayout(self) -> dbc.Tab:
        _resultsMenuDropdowns = dbc.Card(className="results_menu_dropdowns", children=[
            dbc.CardBody([
                html.Div(className="dropdowns", children=[
                    html.H6("Experimental Design : "),
                    dbc.Select(id="design_dropdown_summary",
                               className="form_select",
                               options=[{"label": "None", "value": "None"}],
                               value="None",
                               )]
                         ),

                dbc.Button("Load", color="primary", id="load_results_button",
                           className="custom_buttons", n_clicks=0),
                html.Div(id="output_button_load_results"),
            ],
                id="all_algo_results")

        ])
        _heatmapSamplesAlwaysWrong = html.Div(className="umap_plot_and_title",
                                 children=[
                                     html.Div(className="title_and_help",
                                              children=[
                                                  html.H6("Errors on samples"),
                                                  dbc.Button("[?]",
                                                             className="text-muted btn-secondary popover_btn",
                                                             id="help_heatmapSamples"),
                                                  dbc.Popover(
                                                      children=[
                                                          dbc.PopoverBody(
                                                              "Blablabla wout wout")
                                                      ],
                                                      id="pop_help_heatmapSamples",
                                                      is_open=False,
                                                      target="help_heatmapSamples")
                                              ]),

                                     dcc.Loading(dcc.Graph(id="heatmapSamples"),
                                                 type="dot", color="#13BD00"),
                                     # dcc.Slider(min=0, max=3, step=1, value=0, marks={0: "10", 1: "40", 2: "100", 3: "All"},
                                     #            id="features_stripChart_dropdown")

                                 ])


        return dbc.Tab(className="global_tab",

                       label="Results Summary", children=[
                html.Div(className="fig_group", children=[
                    html.Div(className="column_content",
                             # WARNING !! : _infoFigure is not with the card, it's in a separate column
                             children=[_resultsMenuDropdowns, _heatmapSamplesAlwaysWrong])])])

    def _registerCallbacks(self) -> None:

        @self.app.callback(
            [Output("design_dropdown_summary", "options"),
             Output("design_dropdown_summary", "value")],
            [Input("custom_big_tabs", "active_tab")]
        )
        def update_results_dropdown_design(active):
            if active == "tab-4":
                self.r = self.metabo_controller.get_all_results()
                a = list(self.r.keys())
                return [{"label": i, "value": i} for i in a], a[0]
            else:
                return dash.no_update



        @self.app.callback(
            Output("heatmapSamples", "figure"),
            [Input("load_results_button", "n_clicks")],
            State("design_dropdown_summary", "value")
        )
        def show_heatmap_samples_always_wrong(n_clicks, design):
            if n_clicks >= 1:
                algos = list(self.r[design].keys())

                data_train = []
                data_test = []
                all_samples = []

                for i, a in enumerate(algos):
                    data_train.append([])
                    data_test.append([])
                    train = []
                    test = []
                    for j, s in enumerate(self.r[design][a].splits_number):
                        train_d, test_d = self.r[design][a].results[s]["failed_samples"]
                        train.append(train_d)
                        test.append(test_d)

                    counter_train = Counter()
                    for d in train:
                        counter_train.update(d)

                    counter_test = Counter()
                    for d in test:
                        counter_test.update(d)

                    all_samples = list(counter_train.keys()) + list(counter_test.keys())
                    for s in all_samples:
                        if s in counter_train.keys():
                            data_train[i].append(counter_train[s])
                        else:
                            data_train[i].append(0)

                        if s in counter_test.keys():
                            data_test[i].append(counter_test[s])
                        else:
                            data_test[i].append(0)

                data_train = np.array(data_train).T
                data_test = np.array(data_test).T

                fig = self._plots.show_heatmap_wrong_samples(data_train, data_test, all_samples, algos)

                return fig
            else:
                return dash.no_update