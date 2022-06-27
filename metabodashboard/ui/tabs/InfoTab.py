import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input, State, callback_context

from .MetaTab import MetaTab
from ...service import decode_pickle_from_base64, Utils


class InfoTab(MetaTab):
    def getLayout(self) -> dbc.Tab:
        _loadExpe = dbc.Card(className="cards_info", children=[
            dbc.CardHeader("Load MetaboExperiment"),
            dbc.CardBody(
                [
                    html.P(
                        "You can load the file of a previous experiment and resume your analysis."
                    ),
                    dcc.Upload(id="load_expe", style={"width": "fit-content"},
                               children=[dbc.Button("Select File",
                                                    id="load_expe_button",
                                                    className="custom_buttons",
                                                    color="primary")]),
                ]
            ),
        ])

        _splitsInfo = dbc.Card(className="cards_info", children=[
            dbc.CardHeader("Splits"),
            dbc.CardBody(
                [
                    html.P(
                        "In the Splits tab, you create a setting file with all info necessary to run a machine learning experiment. "
                        "This file will even contain path to a copy of the data to avoid broken paths (after some times, "
                        "files might be moved or deleted and the path pointing to their location will then be not valid). "
                    ),
                    html.P(
                        "It is at this step that you can decide the (potential) multiple design experiment that you wish to explore. "
                        "Indeed, in the case where you have 3 or more different labels (eg: healthy, sick, extremely sick or diet1, diet2, diet3, diet4), "
                        "you will be able to combine them in two groups (two classes) to confront the different conditions in different ways. "
                        "For example, can the algorithms differentiate/discriminate between diet1 vs all the others, or diet1 and diet2 vs diet3 and diet4, etc.",
                    ),
                ]
            ),
        ])

        _MLInfo = dbc.Card(className="cards_info", children=[
            dbc.CardHeader("Machine Learning"),
            dbc.CardBody(
                [
                    html.P(
                        "In this tab, you select the cross-validation parameters and the algorithms you wish to apply on your data."
                        "The cross-validation step is usefull to optimize the algorithm : it runs several time the algorithm on small"
                        "part of the dataset with different parameters, and will keep the best parameters/model which will be applied "
                        "to your real dataset and gives you the final results and analyses."

                    ),
                    html.P(
                        "For the algorithms selection, there is some already implemented by default in the tool, so you can simply "
                        "select them. But there is also the possibility to import manually other algorithms from Scikit-Learn, in this"
                        "case you need to provide several information about the algorithm you wish to add so it can be integrated in the "
                        "analysis."
                    ),
                    html.P(
                        "The possibility to add a completely custom algorithm will eventually also be available. But it will require "
                        "modifications directly in the code files, and thus? is meant for people with more informatics abilities."
                    ),
                ]
            ),
        ])

        _resultInfo = dbc.Card(className="cards_info", children=[
            dbc.CardHeader("Results"),
            dbc.CardBody(
                [
                    html.P("In this tab, ",
                           className="card-text"),
                ]
            ),
        ])

        # _interpretInfo = dbc.Card(className="cards_info", children=[
        #     dbc.CardHeader("Model interpretation"),
        #     dbc.CardBody(
        #         [
        #             html.P("Blablabla",
        #                    className="card-text"),
        #         ]
        #     ),
        # ])

        _infoFigure = html.Div(className="column_content", children=[
            dbc.Card(className="card_body_fig", children=[
                # dbc.Card("Amazing figure here", className="card_body_fig", body=True),
                dbc.CardImg(src="/assets/Figure_home_wider.png", bottom=True)
            ])
        ])
        # TODO : add the filename
        _modal = dbc.Modal(children=[
            dbc.ModalHeader(style={"padding": "2rem 3rem"},
                            children=[html.H6("Warning: Saved local files does not match")]),
            dbc.ModalBody(style={"padding": "2em"}, children=[
                html.P("The data and/or metadata used in the MetaboExperiment file are not the same as the "
                       "local ones."),
                html.P("Please restore the correct data and/or metadata if you want "
                       "to continue the same experiment. (Full restore)"),
                html.P("Otherwise, you can use the same parameters with new data and/or metadata (Partial restore) "
                       "but the sample column name, target column name and experimental designs won't be "
                       "restored."),
                html.P("If you only want to see the results, it will be available (Load results) but metadata and data "
                       "matrix will be reset, as well as the experimental designs."),
                dcc.Upload(id="upload_datatable_modal",
                           children=[dbc.Button("Upload Data Matrix",
                                                id="upload_datatable_modal_button",
                                                # className="custom_buttons",
                                                color="outline-primary")]),
                dcc.Upload(id="upload_metadata_modal",
                           children=[dbc.Button("Upload Metadata",
                                                id="upload_metadata_modal_button",
                                                # className="custom_buttons",
                                                color="outline-primary")]),
            ]),
            dbc.ModalFooter(
                style={"padding-left": "1em"},
                children=[
                    dbc.Button(
                        "Close", id="close", className="custom_buttons", n_clicks=0
                    ),
                    # Show diff
                    dbc.Button(
                        "Load results", id="loadAnyway", className="custom_buttons push", n_clicks=0
                    ),
                    # NO FILES
                    dbc.Button(
                        "Partial restore", id="partialRestore", className="custom_buttons", n_clicks=0
                    ),
                    # require files
                    dbc.Button(
                        "Full restore", id="fullRestore", className="custom_buttons", n_clicks=0
                    ),
                ]),
        ],
            id="warning-not-match",
            size="lg",
            is_open=False,

        )

        _hidden_div = html.Div(id="hidden_div", style={"display": "none"})

        return dbc.Tab(className="global_tab",
                       tab_style={"margin-left": "auto"},
                       label="Home", children=[_modal,
                                               html.Div(className="fig_group", children=[
                                                   html.Div(className="column_content",
                                                            # WARNING !! : _infoFigure is not with the card, it's in a separate column
                                                            children=[_loadExpe, _splitsInfo, _MLInfo, _resultInfo, _hidden_div]),
                                                   _infoFigure])])  # _interpretInfo

    def _registerCallbacks(self) -> None:
        @self.app.callback(
            [Output("warning-not-match", "is_open"),
             Output("hidden_div", "children")],
            [Input("close", "n_clicks"),
             Input("loadAnyway", "n_clicks"),
             Input("partialRestore", "n_clicks"),
             Input("fullRestore", "n_clicks"),
             Input("load_expe", "contents"),
             Input("upload_datatable_modal", "contents"),
             Input("upload_metadata_modal", "contents"),
             Input("upload_datatable_modal", "filename"),
             Input("upload_metadata_modal", "filename")],
            [State("warning-not-match", "is_open"),
             State("load_expe", "filename")]
        )
        def toggle_modal(close, load_anyway, partial_restore, full_restore, file, data, metadata, data_name, metadata_name, is_open, filename):
            if full_restore:
                metabo_exp_dto = decode_pickle_from_base64(file)
                if Utils.are_files_corresponding(data, metadata, metabo_exp_dto):
                    self.metabo_controller.full_restore(metabo_exp_dto)
                    return False, dcc.Location(href="/home", id="someid_doesnt_matter")
                else:
                    return True, ""
            if partial_restore:
                if data and metadata:
                    metabo_exp_dto = decode_pickle_from_base64(file)
                    self.metabo_controller.partial_restore(metabo_exp_dto, data_name, metadata_name, data=data, metadata=metadata)
                    print("partial restore")
                    print(self.metabo_controller.get_features())
                    return False, dcc.Location(href="/home", id="someid_doesnt_matter")
                else:
                    return True, ""
            if load_anyway:
                metabo_exp_dto = decode_pickle_from_base64(file)
                self.metabo_controller.load_results(metabo_exp_dto)
                return False, dcc.Location(href="/home", id="someid_doesnt_matter")
            if close:
                return False, dash.no_update
            if filename is not None:
                metabo_exp_dto = decode_pickle_from_base64(file)
                if self.metabo_controller.is_save_safe(metabo_exp_dto):
                    self.metabo_controller.full_restore(metabo_exp_dto)
                    return False, dcc.Location(href="/home", id="someid_doesnt_matter")
                else:
                    return True, dash.no_update
            return is_open, dash.no_update

