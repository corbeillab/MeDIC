import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input, State, callback_context

from .MetaTab import MetaTab
from ...service import decode_pickle_from_base64, Utils


class InfoTab(MetaTab):
    def getLayout(self) -> dbc.Tab:
        _docLink = dbc.Card(
            className="cards_info",
            children=[
                dbc.CardHeader("Documentation link"),
                dbc.CardBody(
                    [
                        "You can find the official documentation at this ",
                        html.A(
                            href="https://elinaff.github.io/MetaboDashboard/",
                            target="_blank",
                            rel="noreferrer noopener",
                            children="link",
                        ),
                        ".",
                    ]
                ),
            ],
        )

        _splitsInfo = dbc.Card(
            className="cards_info",
            children=[
                dbc.CardHeader("Splits"),
                dbc.CardBody(
                    [
                        html.P(
                            "In the Splits tab, you create a setting file with all info necessary to run a machine learning experiment. "
                            "There is a hash mecanism in place to ensure that the locally saved data fits the experiment file "
                            "that might be loaded in the futur. This mecanism can be compared to a lock and key mecanism where the key "
                            "to check a file will only fit this particular file."
                        ),
                        html.P(
                            "It is at this step that you can decide the (potential) multiple design experiment that you wish to explore. "
                            "Indeed, in the case where you have 3 or more different labels (eg: healthy, sick, extremely sick or diet1, diet2, diet3, diet4), "
                            "you will be able to combine them in two groups (two classes) to confront the different conditions in different ways. "
                            "For example, can the algorithms differentiate/discriminate between diet1 vs all the others, or diet1 and diet2 vs diet3 and diet4, etc.",
                        ),
                    ]
                ),
            ],
        )

        _MLInfo = dbc.Card(
            className="cards_info",
            children=[
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
                            "select them. But there is also the possibility to import manually other algorithms from Scikit-Learn. In this "
                            "case you need to provide several information about the algorithm you wish to add so it can be integrated in the "
                            "analysis."
                        ),
                        html.P(
                            "The possibility to add a completely custom algorithm will eventually also be available. But it will require "
                            "modifications directly in the code files. It is meant for people with more programming abilities."
                        ),
                    ]
                ),
            ],
        )

        _resultInfo = dbc.Card(
            className="cards_info",
            children=[
                dbc.CardHeader("Results"),
                dbc.CardBody(
                    [
                        html.P(
                            "The entire section of Results is based on the perspective of analysing the features selected "
                            "by the algorithms. For one experimental design of classes, multiple algorithms can be run. "
                            "Then, the results and performances of each algorithm can be explored one by one to ensure that "
                            "the prediction and therefore the selection of the features is valid. At the end, there is a "
                            "section that aggregates the results of all the algorithms in several figures to compare which "
                            "features are selected by which algorithm and check for redundancies. The repeated use of a "
                            "metabolite across different algorithm is a good indicator of the relevance of this molecule.",
                            className="card-text",
                        ),
                    ]
                ),
            ],
        )

        # _interpretInfo = dbc.Card(className="cards_info", children=[
        #     dbc.CardHeader("Model interpretation"),
        #     dbc.CardBody(
        #         [
        #             html.P("Blablabla",
        #                    className="card-text"),
        #         ]
        #     ),
        # ])

        _loadExpe = dbc.Card(
            className="cards_info",
            style={"margin-left": "2em"},
            children=[
                dbc.CardHeader("Load MetaboExperiment"),
                dbc.CardBody(
                    [
                        html.P(
                            "You can load the file of a previous experiment and resume your analysis."
                        ),
                        dcc.Upload(
                            id="load_expe",
                            style={"width": "fit-content"},
                            children=[
                                dbc.Button(
                                    "Select File",
                                    id="load_expe_button",
                                    className="custom_buttons",
                                    color="primary",
                                )
                            ],
                        ),
                    ]
                ),
            ],
        )

        _infoFigure = dbc.Card(
            className="card_body_fig",
            children=[
                # dbc.Card("Amazing figure here", className="card_body_fig", body=True),
                dbc.CardImg(src="/assets/update_figure_steps_MeDIC_4.svg", bottom=True)
            ],
        )
        # TODO : add the filename
        _modal = dbc.Modal(
            children=[
                dbc.ModalHeader(
                    style={"padding": "2rem 3rem"},
                    children=[html.H6("Warning: Saved local files does not match")],
                ),
                dbc.ModalBody(
                    style={"padding": "2em"},
                    children=[
                        html.P(
                            "The data and/or metadata used in the MetaboExperiment file are not the same as the "
                            "local ones."
                        ),
                        html.P(
                            "Please restore the correct data and/or metadata if you want "
                            "to continue the same experiment. (Full restore)"
                        ),
                        html.P(
                            "Otherwise, you can use the same parameters with new data and/or metadata (Partial restore) "
                            "but the sample column name, target column name and experimental designs won't be "
                            "restored."
                        ),
                        html.P(
                            "If you only want to see the results, it will be available (Load results) but metadata and data "
                            "matrix will be reset, as well as the experimental designs."
                        ),
                        html.Div(
                            children=[
                                dcc.Upload(
                                    id="upload_datatable_modal",
                                    children=[
                                        dbc.Button(
                                            "Upload Data Matrix",
                                            id="upload_datatable_modal_button",
                                            # className="custom_buttons",
                                            color="outline-primary",
                                        )
                                    ],
                                ),
                                dcc.Loading(
                                    id="upload_datatable_modal_loading",
                                    type="dot",
                                    color="#13BD00",
                                    children=[
                                        html.Div(id="upload_datatable_modal_output"),
                                    ],
                                ),
                            ],
                            style={"display": "flex", "align-items": "center"},
                        ),
                        html.Div(
                            children=[
                                dcc.Upload(
                                    id="upload_metadata_modal",
                                    children=[
                                        dbc.Button(
                                            "Upload Metadata",
                                            id="upload_metadata_modal_button",
                                            # className="custom_buttons",
                                            color="outline-primary",
                                        )
                                    ],
                                ),
                                dcc.Loading(
                                    id="upload_metadata_modal_loading",
                                    type="dot",
                                    color="#13BD00",
                                    children=[
                                        html.Div(id="upload_metadata_modal_output"),
                                    ],
                                ),
                            ],
                            style={"display": "flex", "align-items": "center"},
                        ),
                        html.Div(
                            id="upload_datatable_modal_error_output",
                            style={"color": "red"},
                        ),
                    ],
                ),
                dbc.ModalFooter(
                    style={"padding-left": "1em"},
                    children=[
                        dbc.Button(
                            "Close", id="close", className="custom_buttons", n_clicks=0
                        ),
                        # NO FILES
                        dbc.Button(
                            "Load results",
                            id="loadAnyway",
                            className="custom_buttons push",
                            n_clicks=0,
                        ),
                        # require files
                        dbc.Button(
                            "Partial restore",
                            id="partialRestore",
                            className="custom_buttons",
                            n_clicks=0,
                        ),
                        # require files
                        dbc.Button(
                            "Full restore",
                            id="fullRestore",
                            className="custom_buttons",
                            n_clicks=0,
                        ),
                    ],
                ),
            ],
            id="warning-not-match",
            size="lg",
            is_open=False,
        )

        _hidden_div = html.Div(id="hidden_div", style={"display": "none"})

        return dbc.Tab(
            className="global_tab",
            tab_style={"margin-left": "auto"},
            label="Home",
            children=[
                _modal,
                html.Div(
                    className="fig_group",
                    children=[
                        html.Div(
                            className="column_content",
                            # WARNING !! : _infoFigure is not with the card, it's in a separate column
                            children=[
                                _docLink,
                                _splitsInfo,
                                _MLInfo,
                                _resultInfo,
                                _hidden_div,
                            ],
                        ),
                        html.Div(
                            className="column_content",
                            children=[
                                _loadExpe,
                                _infoFigure,
                            ],
                        ),
                    ],
                ),
            ],
        )  # _interpretInfo

    def _registerCallbacks(self) -> None:
        @self.app.callback(
            [
                Output("upload_datatable_modal_output", "children"),
                Output("upload_datatable_modal_output", "style"),
            ],
            [Input("upload_datatable_modal", "contents")],
            [State("load_expe", "contents")],
        )
        def set_data_matrix_in_modal(contents, dto_contents):
            if contents is not None:
                metabo_experiment_dto = decode_pickle_from_base64(dto_contents)
                if Utils.is_data_the_same(contents, metabo_experiment_dto):
                    return "Data matrix uploaded successfully", {"color": "green"}
                else:
                    return (
                        "Data matrix uploaded but not corresponding to the local one",
                        {"color": "yellow"},
                    )
            else:
                return dash.no_update, dash.no_update

        @self.app.callback(
            [
                Output("upload_metadata_modal_output", "children"),
                Output("upload_metadata_modal_output", "style"),
            ],
            [Input("upload_metadata_modal", "contents")],
            [State("load_expe", "contents")],
        )
        def set_metadata_in_modal(contents, dto_contents):
            if contents is not None:
                metabo_experiment_dto = decode_pickle_from_base64(dto_contents)
                if Utils.is_metadata_the_same(contents, metabo_experiment_dto):
                    return "Metadata uploaded successfully", {"color": "green"}
                else:
                    return "Metadata uploaded but not corresponding to the local one", {
                        "color": "yellow"
                    }
            else:
                return dash.no_update, dash.no_update

        @self.app.callback(
            [
                Output("warning-not-match", "is_open"),
                Output("hidden_div", "children"),
                Output("upload_datatable_modal_error_output", "children"),
            ],
            [
                Input("close", "n_clicks"),
                Input("loadAnyway", "n_clicks"),
                Input("partialRestore", "n_clicks"),
                Input("fullRestore", "n_clicks"),
                Input("load_expe", "filename"),
            ],
            [
                State("load_expe", "contents"),
                State("upload_datatable_modal", "contents"),
                State("upload_metadata_modal", "contents"),
                State("upload_datatable_modal", "filename"),
                State("upload_metadata_modal", "filename"),
            ],
        )
        def toggle_modal(
            close,
            load_anyway,
            partial_restore,
            full_restore,
            filename_loaded,
            contents_loaded,
            new_data,
            new_metadata,
            new_data_name,
            new_metadata_name,
        ):
            triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]

            if triggered_id == "close":
                return False, dash.no_update
            else:
                if triggered_id == "load_expe":
                    metabo_exp_dto = decode_pickle_from_base64(contents_loaded)
                    if self.metabo_controller.is_save_safe(metabo_exp_dto):
                        self.metabo_controller.full_restore(metabo_exp_dto)
                        return (
                            False,
                            "reload",
                            "",
                        )
                    else:
                        return True, dash.no_update, dash.no_update

                elif triggered_id == "loadAnyway":
                    metabo_exp_dto = decode_pickle_from_base64(contents_loaded)
                    self.metabo_controller.load_results(metabo_exp_dto)
                    return (
                        False,
                        "",
                        "",
                    )  # dcc.Location(href="/home", id="someid_doesnt_matter")

                elif triggered_id == "partialRestore":
                    metabo_exp_dto = decode_pickle_from_base64(contents_loaded)
                    if new_data and new_metadata:
                        try:
                            self.metabo_controller.partial_restore(
                                metabo_exp_dto,
                                new_data_name,
                                new_metadata_name,
                                data=new_data,
                                metadata=new_metadata,
                            )
                        except ValueError as ve:
                            return True, dash.no_update, str(ve)
                        return (
                            False,
                            "",
                            "",
                        )
                    else:
                        return (
                            True,
                            "",
                            "You need to upload both data and metadata to do a partial restore",
                        )

                elif triggered_id == "fullRestore":
                    metabo_exp_dto = decode_pickle_from_base64(contents_loaded)
                    if (
                        new_data is not None
                        and new_metadata is not None
                        and Utils.are_files_corresponding_to_dto(
                            new_data, new_metadata, metabo_exp_dto
                        )
                    ):
                        self.metabo_controller.full_restore(metabo_exp_dto)
                        return (
                            False,
                            "",
                            "",
                        )
                    else:
                        return (
                            True,
                            "",
                            "You need to restore original data matrix and metadata to do a full restore",
                        )

                return False, dash.no_update, dash.no_update
