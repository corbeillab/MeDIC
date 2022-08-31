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

CONFIG = {
    "toImageButtonOptions": {
        "format": "svg",  # one of png, svg, jpeg, webp
        "height": None,
        "width": None,
        "scale": 1,  # Multiply title/legend/axis/canvas sizes by this factor
    }
}


class ResultsSummaryTab(MetaTab):
    def __init__(self, app: Dash, metabo_controller: MetaboController):
        super().__init__(app, metabo_controller)
        self.r = self.metabo_controller.get_all_results()
        self._plots = Plots("blues")

    def getLayout(self) -> dbc.Tab:
        _resultsMenuDropdowns = dbc.Card(
            className="results_menu_dropdowns",
            children=[
                dbc.CardBody(
                    [
                        html.Div(
                            className="dropdowns",
                            children=[
                                html.H6("Experimental Design : "),
                                dbc.Select(
                                    id="design_dropdown_summary",
                                    className="form_select",
                                    options=[{"label": "None", "value": "None"}],
                                    value="None",
                                ),
                            ],
                        ),
                        dbc.Button(
                            "Load",
                            color="primary",
                            id="load_results_button",
                            className="custom_buttons",
                            n_clicks=0,
                        ),
                        html.Div(id="output_button_load_results"),
                    ],
                    id="all_algo_results",
                )
            ],
        )
        _nonRandomHeatmapUsedFeatures = html.Div(
            className="umap_plot_and_title",
            children=[
                html.Div(
                    className="title_and_help",
                    children=[
                        html.H6("Features Usage"),
                        dbc.Button(
                            "[?]",
                            className="text-muted btn-secondary popover_btn",
                            id="help_nonrandomHeatmapFeatures",
                        ),
                        dbc.Popover(
                            children=[dbc.PopoverBody("Blablabla wout wout")],
                            id="pop_help_nonrandomHeatmapFeatures",
                            is_open=False,
                            target="help_nonrandomHeatmapFeatures",
                        ),
                    ],
                ),
                dcc.Loading(
                    dcc.Graph(id="nonRandomHeatmapFeatures", config=CONFIG),
                    type="dot",
                    color="#13BD00",
                ),
            ],
        )
        _randomHeatmapUsedFeatures = html.Div(
            className="umap_plot_and_title",
            children=[
                html.Div(
                    className="title_and_help",
                    children=[
                        html.H6("Features Usage"),
                        dbc.Button(
                            "[?]",
                            className="text-muted btn-secondary popover_btn",
                            id="help_randomheatmapFeatures",
                        ),
                        dbc.Popover(
                            children=[dbc.PopoverBody("Blablabla wout wout")],
                            id="pop_help_randomheatmapFeatures",
                            is_open=False,
                            target="help_randomheatmapFeatures",
                        ),
                    ],
                ),
                dcc.Loading(
                    dcc.Graph(id="randomHeatmapFeatures", config=CONFIG),
                    type="dot",
                    color="#13BD00",
                ),
            ],
        )

        _heatmapSamplesAlwaysWrong = html.Div(
            className="umap_plot_and_title",
            children=[
                html.Div(
                    className="title_and_help",
                    children=[
                        html.H6("Errors on samples in test"),
                        dbc.Button(
                            "[?]",
                            className="text-muted btn-secondary popover_btn",
                            id="help_heatmapSamples",
                        ),
                        dbc.Popover(
                            children=[dbc.PopoverBody("Blablabla wout wout")],
                            id="pop_help_heatmapSamples",
                            is_open=False,
                            target="help_heatmapSamples",
                        ),
                    ],
                ),
                dcc.Loading(
                    dcc.Graph(id="heatmapSamples", config=CONFIG),
                    type="dot",
                    color="#13BD00",
                ),
                # dcc.Slider(min=0, max=3, step=1, value=0, marks={0: "10", 1: "40", 2: "100", 3: "All"},
                #            id="features_stripChart_dropdown")
            ],
        )

        return dbc.Tab(
            className="global_tab",
            label="Results aggregated",
            children=[
                _resultsMenuDropdowns,
                html.Div(
                    className="fig_group",
                    children=[
                        _nonRandomHeatmapUsedFeatures,
                        _randomHeatmapUsedFeatures,
                    ],
                ),
                html.Div(className="fig_group", children=[_heatmapSamplesAlwaysWrong]),
            ],
        )
        # html.Div(className="column_content",
        #          # WARNING !! : _infoFigure is not with the card, it's in a separate column
        #          children=[_heatmapSamplesAlwaysWrong])])])

    def _registerCallbacks(self) -> None:
        @self.app.callback(
            [
                Output("design_dropdown_summary", "options"),
                Output("design_dropdown_summary", "value"),
            ],
            [Input("custom_big_tabs", "active_tab")],
        )
        def update_results_dropdown_design(active):
            if active == "tab-4":
                try:
                    self.r = self.metabo_controller.get_all_results()
                    a = list(self.r.keys())
                    return [{"label": i, "value": i} for i in a], a[0]
                except:  # TODO: wrong practice ???
                    return dash.no_update
            else:
                return dash.no_update

        @self.app.callback(
            [
                Output("randomHeatmapFeatures", "figure"),
                Output("nonRandomHeatmapFeatures", "figure"),
            ],
            [Input("load_results_button", "n_clicks")],
            State("design_dropdown_summary", "value"),
        )
        def show_heatmap_features_usage(n_clicks, design):
            if n_clicks >= 1:
                algos = list(self.r[design].keys())
                global_df = None
                for a in algos:
                    if global_df is None:
                        print("glob df is none")
                        global_df = self.r[design][a].results["features_table"]
                        global_df = global_df.loc[
                            :, ("features", "importance_usage")
                        ]  # reduce dataframe to 2 columns
                        global_df.rename(
                            columns={"importance_usage": a}, inplace=True
                        )  # rename column to identify algorithm
                    else:
                        print("glob df not none, algo :", a)
                        df = self.r[design][a].results[
                            "features_table"
                        ]  # retrieve features table of algo a
                        df = df.loc[
                            :, ("features", "importance_usage")
                        ]  # reduce dataframe to 2 columns
                        df.rename(
                            columns={"importance_usage": a}, inplace=True
                        )  # rename column to identify algorithm
                        global_df = global_df.merge(
                            df, how="outer", on="features"
                        )  # join data with global dataset

                global_df = global_df.set_index("features")
                global_df = global_df.fillna(0)

                random_df = global_df.loc[:, ("RandomForest", "RandomSCM")]
                random_df = random_df[
                    (random_df["RandomForest"] > 0.001)
                    | (random_df["RandomSCM"] > 0.001)
                ]

                non_random_df = global_df.loc[:, ("DecisionTree", "SCM")]
                non_random_df = non_random_df[
                    (non_random_df["DecisionTree"] > 0.01)
                    | (non_random_df["SCM"] > 0.01)
                ]

                random_fig = self._plots.show_heatmap_features_usage(random_df)
                non_random_fig = self._plots.show_heatmap_features_usage(non_random_df)

                return random_fig, non_random_fig
            else:
                return dash.no_update

        @self.app.callback(
            Output("heatmapSamples", "figure"),
            [Input("load_results_button", "n_clicks")],
            State("design_dropdown_summary", "value"),
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

                fig = self._plots.show_heatmap_wrong_samples(
                    data_train, data_test, all_samples, algos
                )

                return fig
            else:
                return dash.no_update
