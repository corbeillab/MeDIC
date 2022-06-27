import dash_bootstrap_components as dbc
import pandas
from dash import html, Output, Input, dash, State, dcc, callback_context
from dash.dcc import send_file

from .MetaTab import MetaTab
from ...domain import MetaboController
from ...service import Utils

EXP_NAME = []


class SplitsTab(MetaTab):

    def __init__(self, app: dash.Dash, metabo_controller: MetaboController):
        super().__init__(app, metabo_controller)

    def getLayout(self) -> dbc.Tab:
        _introductionNotice = html.Div(className="fig_group_all_width", children=[
            dbc.Card(children=[
                dbc.CardBody(
                    [
                        html.Div(
                            "In this tab, you will create a setting file with all info necessary "
                            "to run a machine learning experiment. An * besides the name of the field means a value "
                            "is required, all other fields can be left untouched and will use default values."),
                    ]
                    # This file will even contain a copy of the data "
                    # "to avoid broken paths (after some times, files might be moved or deleted and the path pointing "
                    # "to their location will then be not valid).
                ),
            ]),
        ])

        __dataFile = html.Div(
            [
                dbc.Label("Data file(s) *", className="form_labels"),
                html.Div([
                    dcc.Upload(id="upload_datatable",
                               children=[dbc.Button("Upload File",
                                                    id="upload_datatable_button",
                                                    # className="custom_buttons",
                                                    color="outline-primary")]),
                    dcc.Loading(id="upload_datatable_loading", children=[
                        html.Div(id="upload_datatable_output", style={"color": "green"})
                    ], style={"width": "100%"}, type="dot", color="#13BD00"),
                ], style={"display": "flex", "align-items": "center"}),
                dbc.FormText(
                    "You can give a Progenesis abundance file, or a matrix with samples as lines and features as "
                    "columns.",
                ),
            ],
            className="form_field",
            id="datatable-section",
            style={"display": "none"}
        )

        __metaDataFile = html.Div(
            [
                dbc.Label(className="form_labels", children=[html.Span("Metadata file  "), html.Span(
                    "(optionnal if Progenesis matrix given)", style={"font-size": "0.9em", "text-transform": "none"})]),
                html.Div([
                    dcc.Upload(id="upload_metadata",
                               children=[dbc.Button("Upload File",
                                                    id="upload_metadata_button",
                                                    # className="custom_buttons",
                                                    color="outline-primary")]),
                    dcc.Loading(id="upload_metadata_loading", children=[
                        html.Div(id="upload_metadata_output", style={"color": "green"})
                    ], style={"width": "100%"}, type="dot", color="#13BD00"),
                ], style={"display": "flex", "align-items": "center"}),
                dbc.FormText(
                    "The metadata file should at least contain : one column with samples name corresponding to names in the"
                    "data file, and one column of target/class/condition.",
                )
            ],
            className="form_field",
            id="metadata-section",
            style={"display": "none"}
        )

        __useRawData = html.Div(
            [
                dbc.Label("DATA NORMALIZATION FOR PROGENESIS",
                          className="form_labels"),
                dbc.FormText(
                    "If there is normalized and raw data in your file, you can choose to use raw by selecting raw. "
                    "Default is normalised.",
                ),
                dbc.RadioItems(
                    id="in_use_raw",

                    options=[
                        {"label": "Raw", "value": 'raw'},
                        {"label": "Normalized", "value": 'normalized'},
                        {"label": "Not Progenesis", "value": "nap"}
                    ],
                    labelCheckedStyle={"color": "#13BD00"},
                ),
                html.Div(id="error_data_normalization", style={"color": "red"})

            ],
            className="form_field"
        )

        _file = html.Div(className="title_and_form", children=[
            html.H4(id="CreateSplits_paths_title", children="A) Files"),
            dbc.Form(children=[
                dbc.Col(children=[__useRawData, __dataFile, __metaDataFile
                                  ]),

            ]),

        ])

        __typeGroupLink = dbc.Card([
            html.Div(
                [
                    dbc.Label(
                        "Name of the targets column"),
                    dbc.RadioItems(
                        id="in_target_col_name",
                        options=Utils.format_list_for_checklist(self.metabo_controller.get_features()),
                        value=self.metabo_controller.get_target_column(),
                        inline=True),
                ],
                className="form_field"
            ),
            html.Div(
                [
                    dbc.Label(
                        "Name of the unique id column"),
                    dbc.RadioItems(
                        id="in_ID_col_name",

                        options=Utils.format_list_for_checklist(self.metabo_controller.get_features()),
                        value=self.metabo_controller.get_id_column(),
                        inline=True),
                ],
                className="form_field"
            ),
            html.Div(
                id="info_progenesis_loaded",
                style={"color": "grey", "padding-left": "2em", "font-style": "italic"}
            )],
            body=True
        )

        __labelDefinition = dbc.Card(id="",
                                     children=[
                                         html.Div(
                                             [
                                                 dbc.Label(
                                                     "Type of classification"),
                                                 dbc.RadioItems(
                                                     id="in_classification_type",

                                                     value=0,
                                                     inline=True,
                                                     options=[
                                                         {
                                                             "label": "Binary",
                                                             "value": 0},
                                                         {
                                                             "label": "Multiclass",
                                                             "value": 1,
                                                             "disabled": True},
                                                     ]),
                                             ],
                                             className="form_field"
                                         ),
                                         html.Div(
                                             [
                                                 dbc.Label(
                                                     "Labels"),
                                                 html.Div(
                                                     className="fig_group_mini",
                                                     id="define_classes_desgn_exp",
                                                     children=[
                                                         dbc.Input(
                                                             id="class1_name",
                                                             type="text",
                                                         ),
                                                         dbc.Checklist(
                                                             id="possible_groups_for_class1",
                                                         ),
                                                         dbc.Input(
                                                             id="class2_name",
                                                         ),
                                                         dbc.Checklist(
                                                             id="possible_groups_for_class2",
                                                         )
                                                     ])
                                             ],
                                             className="form_field"
                                         ),
                                         dbc.Button(
                                             "Add",
                                             id="btn_add_design_exp",
                                             color="primary",
                                             className="custom_buttons",
                                             n_clicks=0),
                                         html.Div(
                                             id="output_btn_add_desgn_exp")
                                     ], body=True)

        _experimentalDesigns = html.Div(className="title_and_form",
                                        children=[
                                            html.H4(id="Exp_desg_title",
                                                    children="B) Define Experimental designs"),
                                            dbc.Form(
                                                children=[
                                                    dbc.Col(children=[

                                                        dbc.FormText(
                                                            "Link each sample to its target/class."
                                                        ),
                                                        __typeGroupLink
                                                        ,
                                                        html.Br(),
                                                        dbc.FormText(
                                                            "Experimental Designs."
                                                        ),
                                                        dbc.Card(id="setted_classes_container",
                                                                 children=self._get_wrapped_experimental_designs(),
                                                                 style={"display": "block", "padding": "1em"}),
                                                        dbc.FormText(
                                                            "Define labels and filter out samples."
                                                        ),
                                                        __labelDefinition
                                                        ,
                                                    ]),

                                                ]),
                                        ])

        __posNegPairing = html.Div(
            [
                dbc.Checklist(
                    id="in_pairing_pos_neg",

                    options=[
                        {"label": "Pos and Neg pairing",
                         "value": 0},
                    ],
                    labelCheckedStyle={"color": "#13BD00"},
                )
            ],
        )

        __posPattern = html.Div(
            [
                dbc.Input(id="distinct_id_pos_samples",

                          className="form_input_text",
                          placeholder="Pattern for positive samples"),
            ],
        )

        __negPattern = html.Div(
            [
                dbc.Input(id="distinct_id_neg_samples",

                          className="form_input_text",
                          placeholder="Pattern for negative samples"),
            ],
        )

        __otherPairing = html.Div(
            [
                dbc.Checklist(
                    id="in_pairing_samples",

                    options=[
                        {"label": "Other pairing", "value": 0},
                    ],
                    labelCheckedStyle={"color": "#13BD00"},
                )
            ],
        )

        _type1Pattern = html.Div(
            [
                dbc.Input(id="distinct_id_1_samples",

                          className="form_input_text",
                          placeholder="Pattern for type 1 of samples"),
            ],
        )

        _type2Pattern = html.Div(
            [
                dbc.Input(id="distinct_id_2_samples",

                          className="form_input_text",
                          placeholder="Pattern for type 2 of samples"),
            ],
        )

        _dataFusion = html.Div(className="title_and_form", children=[
            html.H4(id="sep_samples_title", children="C) Sample pairing"),
            dbc.Form(children=[
                dbc.Col(children=[
                    dbc.FormText(
                        "A pattern for a positive file could be '_pos_' if the file name was"
                        " : sample1_pos_JH35.lcs it means that all positive files would have"
                        " the pattern in the middle of their name."),
                    dbc.FormText(
                        "We consider that the name of a pos file is in all point identical"
                        " to the name of the neg file corresponding, except for the pos/neg"
                        " pattern. It is the same consideration for the other potential"
                        " pairing."
                    ),
                    html.Br(),
                    __posNegPairing,
                    html.Div(id="div_pair_pn", children=[
                        __posPattern,
                        __negPattern
                    ], style={'display': 'none'}),
                    __otherPairing,
                    html.Div(id="div_pair_12", children=[
                        _type1Pattern,
                        _type2Pattern
                    ], style={"display": "none"}),

                ])
            ])
        ])

        __sampleProportion = html.Div([
            dbc.Label(
                "Proportion of samples in test"),
            dbc.Input(
                id="in_percent_samples_in_test",

                value=self.metabo_controller.get_train_test_proportion(),
                type="number",
                min=0, max=1,
                step=0.01,
                size="5")
        ], className="form_field")

        __splitsNumber = html.Div([
            dbc.Label(
                "Number of splits"),
            dbc.Input(
                id="in_nbr_splits",

                value=self.metabo_controller.get_number_of_splits(),
                type="number",
                min=1,
                size="5"),
        ],
            className="form_field")

        _splitDefinition = html.Div(className="title_and_form",
                                    children=[
                                        html.H4(
                                            id="Define_split_title",
                                            children="D) Define splits"),
                                        dbc.Form(children=[
                                            dbc.Col(children=[__sampleProportion,
                                                              __splitsNumber
                                                              ]),
                                        ])
                                    ])
        __LDTDDataType = html.Div(
            [
                dbc.Label(
                    "Processing according to data type"),
                dbc.FormText(
                    "LDTD1 means the preprocessing will be done on all samples in one time. "
                    "LDTD2 means the preprocessing will be done seperatly for each split."),
                dbc.RadioItems(
                    id="in_type_of_data",

                    value="none",
                    inline=True,
                    options=[
                        {
                            "label": "None",
                            "value": "none"},
                        {
                            "label": "LDTD 1",
                            "value": "LDTD1"},
                        {
                            "label": "LDTD 2",
                            "value": "LDTD2"},
                    ]),
            ], className="form_field")

        __LDTDPeakPicking = html.Div(
            [
                dbc.Label(
                    "Perform peak picking"),
                dbc.RadioItems(
                    id="in_peak_picking",

                    value=0,
                    inline=True,
                    options=[
                        {
                            "label": "No",
                            "value": 0,
                            "disabled": True},
                        {
                            "label": "Yes",
                            "value": 1,
                            "disabled": True},
                    ]),
            ], className="form_field")

        __LDTDAlignment = html.Div(
            [
                dbc.Label(
                    "Perform alignment"),
                dbc.RadioItems(
                    id="in_alignment",

                    value=0,
                    inline=True,
                    options=[
                        {
                            "label": "No",
                            "value": 0,
                            "disabled": True},
                        {
                            "label": "Yes",
                            "value": 1,
                            "disabled": True},
                    ]),
            ],
            className="form_field"
        )

        __LDTDNormalization = html.Div(
            [
                dbc.Label(
                    "Perform normalization"),
                dbc.RadioItems(
                    id="in_normalization",

                    value=0,
                    inline=True,
                    options=[
                        {
                            "label": "No",
                            "value": 0,
                            "disabled": True},
                        {
                            "label": "Yes",
                            "value": 1,
                            "disabled": True},
                    ]),
            ], className="form_field")

        __peakThreshold = html.Div([
            dbc.Label(
                "Peak Threshold"),
            dbc.Input(
                id="in_peak_threshold_value",

                value="500",
                type="number",
                min=1,
                size="5")
        ], className="form_field")

        __autoOptimizeNumber = html.Div([
            dbc.Label(
                "AutoOptimize number"),
            dbc.Input(
                id="in_autoOptimize_value",

                value="20",
                type="number",
                min=1,
                size="5")
        ], className="form_field")

        # TODO: Not displayed
        _otherProcessing = html.Div(className="title_and_form",
                                    children=[
                                        html.H4(id="preprocess_title",
                                                children="E) Other Preprocessing"),
                                        dbc.Form(children=[
                                            dbc.Col(children=[
                                                dbc.FormText(
                                                    "Options in case of LDTD data that needs to be preprocess"),
                                                dbc.Collapse(
                                                    dbc.Card(
                                                        dbc.CardBody(
                                                            children=[__LDTDDataType,
                                                                      __LDTDPeakPicking,
                                                                      __LDTDAlignment,
                                                                      __LDTDNormalization,
                                                                      __peakThreshold,
                                                                      __autoOptimizeNumber
                                                                      ]
                                                        )
                                                    ),
                                                    id="collapse_preprocessing",
                                                ),
                                                dbc.Button(
                                                    "Open",
                                                    id="collapse_preprocessing_button",
                                                    className="custom_buttons",
                                                    color="primary",
                                                    n_clicks=0
                                                ),

                                            ])
                                        ])
                                    ],
                                    style={"visibility": "hidden"})

        _generateFile = html.Div(className="title_and_form",
                                 children=[
                                     html.H4(
                                         id="create_split_title",
                                         children="E) Generate file"),
                                     dbc.Form(children=[
                                         dbc.Col(children=[
                                             html.Div(
                                                 id="output_button_split_file"),
                                             html.Div(
                                                 className="button_box",
                                                 children=[
                                                     html.Div(
                                                         "Before clicking on the Create button, make shure all field with an * are correctly filled."),
                                                     dbc.Button(
                                                         "Create",
                                                         color="primary",
                                                         id="split_dataset_button",
                                                         className="custom_buttons",
                                                         n_clicks=0),
                                                     html.Div(
                                                         id="output_button_split",
                                                         children="",
                                                         style={
                                                             'display': 'none'}),
                                                 ]),
                                         ])
                                     ])
                                 ])

        return dbc.Tab(className="global_tab",
                       label="Splits",
                       children=[_introductionNotice,
                                 html.Div(className="fig_group",
                                          children=[_file,
                                                    _experimentalDesigns,
                                                    ]),

                                 html.Div(className="fig_group",
                                          children=[_dataFusion,
                                                    _splitDefinition
                                                    ]),

                                 html.Div(className="fig_group",
                                          children=[_otherProcessing,
                                                    _generateFile
                                                    ]),
                                 dcc.Download(id="download-save-file-split")
                                 ])

    def _registerCallbacks(self) -> None:
        @self.app.callback(
            [Output("datatable-section", "style"),
             Output("metadata-section", "style")],
            [Input("in_use_raw", "value")]
        )
        def normalization_selection(value):
            if value is not None:
                return {"display": "block"}, {"display": "block"}
            return dash.no_update, dash.no_update

        @self.app.callback(
            [Output('info_progenesis_loaded', 'children'),
             Output('upload_datatable_output', 'children'),
             Output('upload_datatable_output', 'style'),
             Output("error_data_normalization", "children")],
            [Input('upload_datatable', 'contents')],
            [State('upload_datatable', 'filename'),
             State("in_use_raw", "value")
             ]
        )
        def upload_data(list_of_contents, list_of_names, normalization):
            if list_of_contents is not None:
                if normalization is None:
                    return dash.no_update, dash.no_update, dash.no_update, [
                        html.P("You must select a normalization before adding the data file(s)")]
                else:
                    if normalization == 'raw':
                        use_raw = True
                    else:
                        use_raw = False

                try:
                    self.metabo_controller.set_data_matrix_from_path(list_of_names,
                                                                     data=list_of_contents,
                                                                     use_raw=use_raw)
                except TypeError as err:
                    return dash.no_update, [html.P(str(err))], {"color": "red"}, ""
                except pandas.errors.ParserError as err:
                    return dash.no_update, [html.P("Rows must have an equal number of columns")], {
                        "color": "red"}, ""
                self.metabo_controller.reset_experimental_designs()

                if self.metabo_controller.is_progenesis_data():
                    # trigger the update of possible targets
                    return "Info: Selection not needed, handled by Progenesis.", [
                        html.P(f"\"{list_of_names}\" has successfully been uploaded !")], {"color": "green"}, ""
                return "", [html.P(f"\"{list_of_names}\" has successfully been uploaded !")], {
                    "color": "green"}, ""
            else:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        @self.app.callback(
            [Output("div_pair_pn", "style"),
             Output("div_pair_12", "style")],
            [Input("in_pairing_pos_neg", "value"),
             Input("in_pairing_samples", "value")]
        )
        def pairing_show_hide_name_fields(pair_pn, pair_12):
            if pair_pn == [0] and pair_12 == [0]:
                return {"display": "block"}, {"display": "block"}
            elif pair_pn == [0] and pair_12 != [0]:
                return {"display": "block"}, {"display": "none"}
            elif pair_pn != [0] and pair_12 == [0]:
                return {"display": "none"}, {"display": "block"}
            else:
                return {"display": "none"}, {"display": "none"}

        @self.app.callback([Output("in_target_col_name", "options"),
                            Output("in_ID_col_name", "options"),
                            Output("upload_metadata_output", "children"),
                            Output("upload_metadata_output", "style")],
                           [Input('upload_metadata', 'contents')],
                           [State('upload_metadata', 'filename')]
                           )
        def get_metadata_cols_names_to_choose_from(list_of_contents, list_of_names):
            if list_of_contents is not None:
                try:
                    self.metabo_controller.set_metadata(list_of_names, data=list_of_contents)
                except TypeError as err:
                    return [], [], html.P(str(err)), {"color": "red"}
                formatted_columns = Utils.format_list_for_checklist(self.metabo_controller.get_features())
                self.metabo_controller.reset_experimental_designs()
                return formatted_columns, formatted_columns, html.P(
                    f"\"{list_of_names}\" has successfully been uploaded !"), {"color": "green"}
            else:
                return dash.no_update

        @self.app.callback(
            Output("define_classes_desgn_exp", "children"),
            [Input("in_classification_type", "value")]
        )
        def define_classes_for_experiment_design(t):
            """
            if the classification type is binary (0), certain options will be available
            if it is multiclass (1), other options wil be shown
            :param t:
            :return:
            """
            if t == 0:
                return [
                    html.Div(className="title_and_form_mini", children=[
                        dbc.Form(children=[
                            dbc.Col(children=[
                                html.Div(
                                    [
                                        dbc.Label("Label 1"),
                                        dbc.Input(id="class1_name",
                                                  placeholder="Enter name",
                                                  debounce=True,
                                                  className="form_input_text"),

                                    ],
                                    className="form_field"
                                ),
                                html.Div(
                                    [
                                        dbc.Label("Class(es)"),
                                        dbc.Checklist(
                                            id="possible_groups_for_class1")
                                    ]
                                )
                            ])
                        ])
                    ]),
                    html.Div(className="title_and_form_mini", children=[
                        dbc.Form(children=[
                            dbc.Col(children=[
                                html.Div(
                                    [
                                        dbc.Label("Label 2"),
                                        dbc.Input(id="class2_name",
                                                  placeholder="Enter name",
                                                  debounce=True,
                                                  className="form_input_text"),

                                    ],
                                    className="form_field"
                                ),
                                html.Div(
                                    [
                                        dbc.Label("Class(es)"),
                                        dbc.Checklist(
                                            id="possible_groups_for_class2")
                                    ]
                                )
                            ])
                        ])
                    ])
                ]

        @self.app.callback(
            [Output("possible_groups_for_class1", "options"),
             Output("possible_groups_for_class2", "options"),
             Output("output_btn_add_desgn_exp", "children"),
             Output("in_target_col_name", "value")],
            [Input("in_target_col_name", "value"),
             Input('info_progenesis_loaded', 'children')],
        )
        def update_possible_classes_exp_design(target_col, children):
            triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]

            if triggered_id == "in_target_col_name":
                self.metabo_controller.set_target_column(target_col)
                formatted_possible_targets = Utils.format_list_for_checklist(
                    self.metabo_controller.get_unique_targets()
                )
                return formatted_possible_targets, formatted_possible_targets, "", target_col
            elif triggered_id == "info_progenesis_loaded":
                formatted_possible_targets = Utils.format_list_for_checklist(
                    self.metabo_controller.get_unique_targets()
                )
                return formatted_possible_targets, formatted_possible_targets, "", None
            else:
                return [], [], "", self.metabo_controller.get_target_column()

        @self.app.callback(
            [Output("class1_name", "value"),
             Output("possible_groups_for_class1", "value"),
             Output("class2_name", "value"),
             Output("possible_groups_for_class2", "value"),
             Output("setted_classes_container", "children"),
             Output("setted_classes_container", "style")],
            [Input("btn_add_design_exp", "n_clicks"),
             Input("remove_experimental_design_button", "n_clicks"),
             Input("in_target_col_name", "value"),
             Input("info_progenesis_loaded", "children")],
            [State("class1_name", "value"),
             State("possible_groups_for_class1", "value"),
             State("class2_name", "value"),
             State("possible_groups_for_class2", "value")],
        )
        def add_n_reset_classes_exp_design(n_add, n_remove, target_col, children, c1, g1, c2, g2):
            triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]

            if triggered_id == "remove_experimental_design_button" or triggered_id == "in_target_col_name" or \
                    triggered_id == "info_progenesis_loaded":
                self.metabo_controller.reset_experimental_designs()
            elif triggered_id == "btn_add_design_exp":
                self.metabo_controller.add_experimental_design({c1: g1, c2: g2})

            return "", 0, "", 0, self._get_wrapped_experimental_designs(), {"display": "block", "padding": "1em"}

        @self.app.callback(
            Output("collapse_preprocessing", "is_open"),
            [Input("collapse_preprocessing_button", "n_clicks")],
            [State("collapse_preprocessing", "is_open")],
        )
        def toggle_collapse_preprocessing(n, is_open):
            if n:
                return not is_open
            return is_open

        @self.app.callback(
            Output("in_ID_col_name", "value"),
            [Input("in_ID_col_name", "value")]
        )
        def update_ID_col_name(new_value):
            if new_value is not None:
                self.metabo_controller.set_id_column(new_value)
            return self.metabo_controller.get_id_column()

        @self.app.callback(
            Output("in_nbr_splits", "value"),
            [Input("in_nbr_splits", "value")]
        )
        def update_nbr_splits(new_value):
            if callback_context.triggered[0]["prop_id"] == ".":
                return self.metabo_controller.get_number_of_splits()
            try:
                casted_value = int(new_value)
            except (ValueError, TypeError):
                return new_value
            self.metabo_controller.set_number_of_splits(int(casted_value))
            return casted_value

        @self.app.callback(
            Output("in_percent_samples_in_test", "value"),
            [Input("in_percent_samples_in_test", "value")]
        )
        def update_percent_samples_in_test(new_value):
            if callback_context.triggered[0]["prop_id"] == ".":
                return self.metabo_controller.get_train_test_proportion()
            try:
                casted_value = float(new_value)
            except (ValueError, TypeError):
                return new_value
            self.metabo_controller.set_train_test_proportion(casted_value)
            return casted_value

        @self.app.callback(
            [Output('output_button_split_file', 'children'),
             Output("download-save-file-split", "data")],
            [Input('split_dataset_button', 'n_clicks')],
            [State("in_use_raw", "value"),
             State('in_nbr_splits', 'value'),
             State('in_nbr_processes', 'value'),
             # State("path_to_data_file", "value"),
             State('in_peak_threshold_value', 'value'),
             State('in_percent_samples_in_test', 'value'),
             State('in_autoOptimize_value', 'value'),
             # State('in_path_to_metadata', 'value'),
             State('in_ID_col_name', 'value'),
             State('in_target_col_name', 'value'),
             State("in_type_of_data", "value"),
             State("in_peak_picking", "value"),
             State("in_alignment", "value"),
             State("in_normalization", "value"),
             State("in_pairing_pos_neg", "value"),
             State("distinct_id_pos_samples", "value"),
             State("distinct_id_neg_samples", "value"),
             State("in_pairing_samples", "value"),
             State("distinct_id_1_samples", "value"),
             State("distinct_id_2_samples", "value"),
             ]
        )
        def saving_params_of_splits_batch(n, use_raw, nbr_splits, nbr_processes,  # path_data_files,
                                          peakT, percent_in_test, autoOpt, ID_col_name,  # path_to_metadata,
                                          targets_col_name,
                                          type_of_processing, peak_pick, align, normalize, pair_pn, pair_id_pos,
                                          pair_id_neg, pair_12, pair_id_1, pair_id_2):
            """
            Create the file (json) which will contains all info about the split creation / data experiment.
            """
            if n >= 1:
                self.metabo_controller.create_splits()
                Utils.dump_metabo_expe(self.metabo_controller.generate_save())

                return "The parameters file is created, the splits's creation should start shortly...", \
                       send_file(Utils.get_metabo_experiment_path())
            else:
                return dash.no_update, dash.no_update

    def _get_wrapped_experimental_designs(self):
        children_container = [html.Div("Experimental design")]
        all_experimental_designs = self.metabo_controller.get_all_experimental_designs_names()

        if len(all_experimental_designs) == 0:
            button = html.Div(
                dbc.Button("Reset", className="custom_buttons", id="remove_experimental_design_button"),
                style={"display": "none"}
            )
            return html.Div([html.P("No experimental design setted yet."), button])

        for _, full_name in all_experimental_designs:
            children_container.append(html.Div(children=["- " + full_name],
                                               style={"display": "flex",
                                                      "justify-content": "space-between", "align-items": "center"})
                                      )
        button = html.Div(
            dbc.Button("Reset", className="custom_buttons", id="remove_experimental_design_button"),
            style={"textAlign": "right"}
        )
        children_container.append(button)
        return children_container
