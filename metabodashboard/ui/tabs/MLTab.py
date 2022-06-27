import dash_bootstrap_components as dbc
from dash import html, State, Input, Output, dash, dcc, callback_context

from .MetaTab import MetaTab
from ...service import Utils


class MLTab(MetaTab):
    def getLayout(self) -> dbc.Tab:
        __splitConfigFile = html.Div(
            [
                dbc.Label("Select CV search type",
                          className="form_labels"),
                dbc.RadioItems(
                    options=Utils.format_list_for_checklist(self.metabo_controller.get_cv_types()),
                    value=self.metabo_controller.get_selected_cv_type(),
                    id="radio_cv_types"),
            ],
            className="form_field"
        )

        __CVConfig = html.Div([
            dbc.Label("Number of Cross Validation folds", className="form_labels"),
            dbc.Input(id="in_nbr_CV_folds", value="5", type="number", min=1,
                      size="5")
        ], className="form_field")

        __processNumberConfig = html.Div([
            dbc.Label("Number of processes"),
            dbc.Input(id="in_nbr_processes", value="2", type="number",
                      min=1,
                      size="5")
        ], className="form_field")

        _definitionLearningConfig = html.Div(className="title_and_form", children=[
            html.H4(id="Learn_conf_title", children="Define Learning configs"),
            dbc.Form(children=[
                dbc.Col(children=[__splitConfigFile,
                                  __CVConfig,
                                  __processNumberConfig
                                  ],
                        )
            ])
        ])

        __availableAlgorithms = html.Div(
            [
                dbc.Label("Available Algorithms", className="form_labels"),
                dbc.Checklist(id="in_algo_ML",
                              # inline=True
                              ),
            ],
            className="form_field"
        )

        __addCustomAlgorithm = html.Div(
            [
                dbc.Label("Add Sklearn Algorithms", className="form_labels"),
                dbc.Label("from sklearn.A import B"),
                dbc.Input(id="import_new_algo", placeholder="Complete import (A)",
                          className="form_input_text"),
                dbc.Input(id="name_new_algo", placeholder="Enter Name (B)",
                          className="form_input_text"),
                dbc.Label("Specify parameters to explore by gridsearch"),
                dbc.Input(id="name_param", placeholder="Name of parameter",
                          className="form_input_text"),
                dbc.Input(id="values_param", placeholder="Values to explore",
                          className="form_input_text"),
                html.Div(children=[
                    html.Br(),
                    html.P("You can set the grid search parameters as followed:"),
                    html.P("Name of parameter: 'param1, param2'"),
                    html.P("Values to explore: '[val1A, val1B, val1C], [val2A, val2B, val2C]'")
                ]),
                dbc.Button("Add", color="success",
                           id="add_n_refresh_sklearn_algo_button",
                           className="custom_buttons", n_clicks=0),

            ],
            className="form_field"
        )

        __validationButton = html.Div(className="button_box", children=[
            html.Div(
                "Before clicking on the Learn button, make shure all field with an * are correctly filled."),
            dbc.Button("Learn", color="primary", id="start_learning_button",
                       className="custom_buttons", n_clicks=0),
            html.Div(id="output_button_ml", children="",
                     style={'display': 'none'}),

        ])

        _definitionLearningAlgorithm = html.Div(className="title_and_form", children=[
            html.H4(id="learn_algo_title", children="Define Learning Algorithms"),
            dbc.Form(children=[
                dbc.Col(children=[__availableAlgorithms, __addCustomAlgorithm, __validationButton
                                  ],
                        )
            ])
        ])

        return dbc.Tab(className="global_tab", label="Machine Learning",
                       children=[
                           html.Div(className="fig_group",
                                    children=[_definitionLearningConfig,
                                              _definitionLearningAlgorithm
                                              ]),
                           dcc.Download(id="download-save-file-ml")
                       ])

    def _registerCallbacks(self) -> None:
        @self.app.callback(
            [Output("in_algo_ML", "options"),
             Output("in_algo_ML", "value"),
             Output("import_new_algo", "value"),
             Output("name_new_algo", "value"),
             Output("name_param", "value"),
             Output("values_param", "value")],
            [Input("add_n_refresh_sklearn_algo_button", "n_clicks"),
             Input("in_algo_ML", "value")],
            [State("import_new_algo", "value"),
             State("name_new_algo", "value"),
             State("name_param", "value"),
             State("values_param", "value")]
        )
        def add_refresh_available_sklearn_algorithms(n, value, import_new, name_new, name_param, values_param):
            if n >= 1:
                new_algo_name = name_new
                new_algo_params = {"function": name_new, "ParamGrid": {name_param: values_param},
                                   "importing": import_new}

                self.metabo_controller.add_custom_model(new_algo_name, import_new, new_algo_params)
            if callback_context.triggered[0]["prop_id"] == "in_algo_ML.value":
                self.metabo_controller.set_selected_models(value)
            return Utils.format_list_for_checklist(
                self.metabo_controller.get_all_algos_names()), \
                self.metabo_controller.get_selected_models(), "", "", "", ""

        @self.app.callback(
            [Output("output_button_ml", "children"),
             Output("download-save-file-ml", "data")],
            [Input("start_learning_button", "n_clicks")],
            [State("in_algo_ML", "value"),
             State("in_nbr_CV_folds", "value"),
             State("in_nbr_processes", "value")]
        )
        def start_machine_learning(n, selected_models, cv_folds, nbr_process):
            if n >= 1:
                print("in")
                print(self.metabo_controller.get_selected_models())
                self.metabo_controller.learn(int(cv_folds))

                Utils.dump_metabo_expe(self.metabo_controller.generate_save())

                return "Done!", dcc.send_file(Utils.get_metabo_experiment_path())
            else:
                return dash.no_update

        @self.app.callback(
            Output("radio_cv_types", "value"),
            [Input("radio_cv_types", "value")]
        )
        def set_cv_type(value):
            if callback_context.triggered[0]["prop_id"] == "radio_cv_types.value":
                self.metabo_controller.set_cv_type(value)

            return self.metabo_controller.get_selected_cv_type()
