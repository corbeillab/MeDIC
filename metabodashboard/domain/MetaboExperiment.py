from typing import Generator, Tuple, List, Dict

import sklearn

from . import ExperimentalDesign
from . import MetaData, MetaboModel
from .DataMatrix import DataMatrix
from .MetaboExperimentDTO import MetaboExperimentDTO
from .ModelFactory import ModelFactory
from ..conf.SupportedCV import CV_ALGORITHMS
from ..service import Utils

X_TRAIN_INDEX = 0
X_TEST_INDEX = 1
y_TRAIN_INDEX = 2
y_TEST_INDEX = 3


class MetaboExperiment:
    def __init__(self):
        self._model_factory = ModelFactory()

        self._data_matrix = DataMatrix()
        self._is_progenesis_data = False
        self._metadata = MetaData()

        self._number_of_splits = 5
        self._train_test_proportion = 0.2
        self._pairing_group_column = ""
        self._cv_folds = 5
        self._number_of_processes_for_cv = 2

        self.experimental_designs = {}

        self._supported_models = self._model_factory.create_supported_models()
        self._custom_models = {}
        self._selected_models = []
        self._cv_algorithms = CV_ALGORITHMS
        self._selected_cv_type = list(self._cv_algorithms.keys())[0]

    def init_metadata(self):
        self._metadata = MetaData()

    def get_metadata(self) -> MetaData:
        return self._metadata

    def init_data_matrix(self):
        self._data_matrix = DataMatrix()

    def set_metadata_with_dataframe(self, filename, data=None, from_base64=True):
        self.init_metadata()
        self._metadata.read_format_and_store_metadata(
            filename, data=data, from_base64=from_base64
        )

    def set_data_matrix(
        self, path_data_matrix: str, data=None, from_base64: bool = True
    ):
        self._data_matrix.reset_file()
        metadata_df = self._data_matrix.read_format_and_store_data(
            path_data_matrix, data=data, from_base64=from_base64
        )
        if metadata_df is not None:
            self._metadata = MetaData(metadata_df)
            self._metadata.set_id_column("sample_names")
            self._metadata.set_target_column("labels")
            self._is_progenesis_data = True
        else:
            self._is_progenesis_data = False

    def get_data_matrix(self) -> DataMatrix:
        return self._data_matrix

    def get_train_test_proportion(self) -> float:
        return self._train_test_proportion

    def get_number_of_splits(self) -> int:
        return self._number_of_splits

    def set_number_of_splits(self, number_of_splits: int):
        self._number_of_splits = number_of_splits

    def set_train_test_proportion(self, train_test_proportion: float):
        self._train_test_proportion = train_test_proportion

    def create_splits(self):
        if self._number_of_splits is None:
            raise ValueError("Number of splits not set")
        if self._train_test_proportion is None:
            raise ValueError("Train test proportion not set")
        if self._pairing_group_column is None:
            raise ValueError("Pairing group column not set")
        if self._metadata is None:
            raise ValueError("Metadata not set")
        for _, experimental_design in self.experimental_designs.items():
            experimental_design.set_split_parameter_and_compute_splits(
                self._train_test_proportion,
                self._number_of_splits,
                self._metadata,
                self._pairing_group_column,
            )

    def get_pairing_group_column(self) -> str:
        return self._pairing_group_column

    def set_pairing_group_column(self, pairing_group_column: str):
        if pairing_group_column not in self._metadata.get_columns():
            raise RuntimeError(
                "Column {} is not in the metadata".format(pairing_group_column)
            )
        self._pairing_group_column = pairing_group_column

    def get_experimental_designs(self) -> Dict[str, ExperimentalDesign]:
        return self.experimental_designs

    def _raise_if_classes_design_is_not_valid(self, classes_design: dict) -> None:
        if "" in list(classes_design.keys()):
            raise ValueError("Empty label(s) is not allowed")
        if len(classes_design) < 2:
            raise ValueError("Labels must have different names")
        items = [inner_val for val in classes_design.values() for inner_val in val]
        if 0 in items:
            raise ValueError(
                "You need to select at least one class. If no class appears, "
                "please upload a data matrix and, if necessary, a metadata file"
            )
        if len(set(items)) != len(items):
            raise ValueError("Duplicate class name is not allowed")

    def add_experimental_design(self, classes_design: dict):
        self._raise_if_classes_design_is_not_valid(classes_design)
        experimental_design = ExperimentalDesign(classes_design)
        self.experimental_designs[experimental_design.get_name()] = experimental_design

    def remove_experimental_design(self, name: str):
        self.experimental_designs.pop(name)

    def add_custom_model(
        self,
        model_name: str,
        needed_imports: str,
        params: List[str],
        values_to_explore: List[List[str]],
    ):
        self._custom_models[model_name] = self._model_factory.create_custom_model(
            model_name, needed_imports, params, values_to_explore
        )

    def get_custom_models(self) -> dict:
        return self._custom_models

    def set_selected_models(self, selected_models: list):
        if self.experimental_designs == {}:
            raise ValueError(
                "You must define at least one experimental design before selecting models."
            )
        self._selected_models = selected_models
        for _, experimental_design in self.experimental_designs.items():
            experimental_design.set_selected_models_name(selected_models)

    def update_experimental_designs_with_selected_models(self):
        for _, experimental_design in self.experimental_designs.items():
            experimental_design.set_selected_models_name(self._selected_models)

    def get_selected_models(self) -> list:
        return self._selected_models

    def get_metadata_columns(self) -> list:
        if self._metadata is None:
            raise RuntimeError("Metadata is not set.")
        return self._metadata.get_columns()

    def set_target_column(self, target_column: str):
        if self._metadata is None:
            raise RuntimeError("Metadata is not set.")
        self._metadata.set_target_column(target_column)

    def set_id_column(self, id_column: str):
        if self._metadata is None:
            raise RuntimeError("Metadata is not set.")
        self._metadata.set_id_column(id_column)

    def get_unique_targets(self) -> list:
        try:
            return self._metadata.get_unique_targets()
        except RuntimeError:
            return []

    def get_model_from_name(self, model_name: str) -> MetaboModel:
        if model_name in self._supported_models.keys():
            return self._supported_models[model_name]
        elif model_name in self._custom_models.keys():
            return self._custom_models[model_name]
        else:
            raise RuntimeError(
                "The model '"
                + model_name
                + "' has not been found neither in supported and custom lists."
            )

    def _check_experimental_design(self):
        error_message = "Train test proportion, number of splits and metadata need to be set before start learning: "
        if self._number_of_splits is None:
            raise RuntimeError(error_message + "missing number of splits")
        if self._train_test_proportion is None:
            raise RuntimeError(error_message + "missing train test proportion")
        if self._metadata is None:
            raise RuntimeError(error_message + "missing metadata")

    def all_experimental_designs_names(self) -> Generator[Tuple[str, str], None, None]:
        for name, experimental_design in self.experimental_designs.items():
            yield name, experimental_design.get_full_name()

    def reset_experimental_designs(self):
        self.experimental_designs = {}

    def _raise_if_value_for_learning_not_setted(self):
        if self._data_matrix is None:
            raise RuntimeError("Data matrix not set")
        if self._metadata is None:
            raise RuntimeError("Metadata not set")
        if self._pairing_group_column is None:
            raise RuntimeError("Pairing group column not set")
        if self._train_test_proportion is None:
            raise RuntimeError("Train test proportion not set")
        if self._number_of_splits is None:
            raise RuntimeError("Number of splits not set")
        if self._selected_models is None:
            raise RuntimeError("Selected models not set")
        if self._cv_folds is None:
            raise RuntimeError("CV folds not set")
        if self.experimental_designs == {}:
            raise RuntimeError(
                "You must define at least one experimental design before learning."
            )

    def learn(self):
        self._raise_if_value_for_learning_not_setted()
        cv_algorithm = self.get_cv_algorithm()
        self._check_experimental_design()
        self._data_matrix.load_data()
        for _, experimental_design in self.experimental_designs.items():
            results = experimental_design.get_results()
            selected_targets_name = experimental_design.get_selected_targets_name()
            (
                selected_targets,
                selected_ids,
            ) = self._metadata.get_selected_targets_and_ids(selected_targets_name)
            classes = Utils.load_classes_from_targets(
                experimental_design.get_classes_design(), selected_targets
            )
            for split_index, split in experimental_design.all_splits():
                x_train = self._data_matrix.load_samples_corresponding_to_IDs_in_splits(
                    split[X_TRAIN_INDEX]
                )
                x_test = self._data_matrix.load_samples_corresponding_to_IDs_in_splits(
                    split[X_TEST_INDEX]
                )
                for model_name in self._selected_models:
                    results[model_name].set_feature_names(x_train)
                    results[model_name].design_name = experimental_design.get_name()
                    metabo_model = self.get_model_from_name(model_name)
                    best_model = metabo_model.train(
                        self._cv_folds,
                        x_train,
                        split[y_TRAIN_INDEX],
                        cv_algorithm,
                        self._number_of_processes_for_cv,
                    )
                    y_train_pred = best_model.predict(x_train)
                    y_test_pred = best_model.predict(x_test)
                    results[model_name].add_results_from_one_algo_on_one_split(
                        best_model,
                        self._data_matrix.get_scaled_data(selected_ids),
                        classes,
                        split[y_TRAIN_INDEX],
                        y_train_pred,
                        split[y_TEST_INDEX],
                        y_test_pred,
                        str(split_index),
                        split[X_TRAIN_INDEX],
                        split[X_TEST_INDEX],
                    )
                experimental_design.set_is_done(True)
        self._data_matrix.unload_data()

    def get_results(self, classes_design: str, algo_name) -> dict:
        return self.experimental_designs[classes_design].get_results()[algo_name]

    def get_all_updated_results(self) -> dict:
        results = {}
        for name in self.experimental_designs:
            if self.experimental_designs[name].get_is_done():
                results[name] = self.experimental_designs[name].get_results()
        return results

    def get_all_algos_names(self) -> list:
        return list(self._supported_models.keys()) + list(self._custom_models.keys())

    def set_cv_type(self, cv_type: str):
        if cv_type not in self._cv_algorithms:
            raise ValueError("CV type '" + cv_type + "' is not supported.")
        self._selected_cv_type = cv_type

    def get_selected_cv_type(self) -> str:
        return self._selected_cv_type

    def get_cv_algorithm(self) -> sklearn.model_selection:
        return self._cv_algorithms[self._selected_cv_type]

    def get_cv_types(self) -> List[str]:
        return list(self._cv_algorithms.keys())

    def generate_save(self) -> MetaboExperimentDTO:
        return MetaboExperimentDTO(self)

    def full_restore(self, saved_metabo_experiment_dto: MetaboExperimentDTO):
        if self.is_save_safe(saved_metabo_experiment_dto):
            raise ValueError(
                "The save is not safe : either the data matrix or the metadata are not the same."
            )
        self._metadata = saved_metabo_experiment_dto.metadata
        self._data_matrix = saved_metabo_experiment_dto.data_matrix
        self._static_restore_for_partial(saved_metabo_experiment_dto)

    def _static_restore_for_partial(
        self, saved_metabo_experiment_dto: MetaboExperimentDTO
    ):
        self._number_of_splits = saved_metabo_experiment_dto.number_of_splits
        self._train_test_proportion = saved_metabo_experiment_dto.train_test_proportion
        self.experimental_designs = saved_metabo_experiment_dto.experimental_designs
        self._custom_models = saved_metabo_experiment_dto.custom_models
        self._selected_models = saved_metabo_experiment_dto.selected_models
        self._selected_cv_type = saved_metabo_experiment_dto.selected_cv_type

    def partial_restore(
        self,
        saved_metabo_experiment_dto: MetaboExperimentDTO,
        filename_data: str,
        filename_metadata: str,
        data=None,
        from_base64_data: bool = True,
        metadata=None,
        from_base64_metadata=True,
    ):
        self._data_matrix.set_raw_use(saved_metabo_experiment_dto.data_matrix.is_raw())
        self._data_matrix.set_remove_rt(
            saved_metabo_experiment_dto.data_matrix.get_remove_rt()
        )
        self.set_data_matrix(
            filename_data,
            data=data,
            from_base64=from_base64_data,
        )
        self.set_metadata_with_dataframe(
            filename_metadata, data=metadata, from_base64=from_base64_metadata
        )
        self._static_restore_for_partial(saved_metabo_experiment_dto)

    def load_results(self, saved_metabo_experiment_dto: MetaboExperimentDTO):
        self.init_metadata()
        self.init_data_matrix()
        self._static_restore_for_partial(saved_metabo_experiment_dto)

    def is_save_safe(self, saved_metabo_experiment_dto: MetaboExperimentDTO) -> bool:
        return (
            self._metadata.get_hash() == saved_metabo_experiment_dto.metadata.get_hash()
            and self._data_matrix.get_hash()
            == saved_metabo_experiment_dto.data_matrix.get_hash()
        )

    def is_the_data_matrix_corresponding(self, data: str) -> bool:
        return self._data_matrix.get_hash() == Utils.compute_hash(data)

    def is_the_metadata_corresponding(self, metadata: str) -> bool:
        return self._metadata.get_hash() == Utils.compute_hash(metadata)

    def get_target_column(self) -> str:
        return self._metadata.get_target_column()

    def get_id_column(self) -> str:
        return self._metadata.get_id_column()

    def is_progenesis_data(self) -> bool:
        return self._is_progenesis_data

    def is_data_raw(self) -> bool:
        print("is_data_raw", self._data_matrix.is_raw())
        return self._data_matrix.is_raw()

    def set_raw_use_for_data(self, use_raw: bool):
        self._data_matrix.set_raw_use(use_raw)

    def get_data_matrix_remove_rt(self) -> bool:
        return self._data_matrix.get_remove_rt()

    def set_data_matrix_remove_rt(self, remove_rt: bool):
        self._data_matrix.set_remove_rt(remove_rt)

    def get_cv_folds(self) -> int:
        return self._cv_folds

    def set_cv_folds(self, cv_folds: int):
        if cv_folds < 2:
            raise ValueError("CV folds must be greater than or equal to 2.")
        self._cv_folds = cv_folds

    def get_number_of_processes_for_cv(self) -> int:
        return self._number_of_processes_for_cv

    def set_number_of_processes_for_cv(self, number_of_processes: int):
        self._number_of_processes_for_cv = number_of_processes


# TODO: print current algo when training
