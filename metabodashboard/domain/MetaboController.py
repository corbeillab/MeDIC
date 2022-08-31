import os
import pickle
from typing import Generator, Tuple

import pandas as pd

from . import MetaboExperiment
from .MetaboExperimentDTO import MetaboExperimentDTO
from .Results import *

ROOT_PATH = os.path.dirname(__file__)
DUMP_PATH = os.path.join(ROOT_PATH, os.path.join("dumps", "splits"))


class MetaboController:
    def __init__(self, metaboExp: MetaboExperiment = None):
        if metaboExp is None:
            self._metabo_experiment = MetaboExperiment()
        else:
            self._metabo_experiment = metaboExp

    def set_metadata(self, filename: str, data=None, from_base64=True) -> None:
        self._metabo_experiment.set_metadata_with_dataframe(
            filename=filename, data=data, from_base64=from_base64
        )

    def set_data_matrix_from_path(
        self,
        path_data_matrix,
        data=None,
        from_base64=True,
    ):
        return self._metabo_experiment.set_data_matrix(
            path_data_matrix,
            data=data,
            from_base64=from_base64,
        )

    def get_metadata_columns(self) -> list:
        return self._metabo_experiment.get_metadata_columns()

    def get_unique_targets(self) -> list:
        return self._metabo_experiment.get_unique_targets()

    def add_experimental_design(self, classes_design: dict):
        self._metabo_experiment.add_experimental_design(classes_design)

    def get_experimental_designs(self):
        return self._metabo_experiment.get_experimental_designs()

    def all_experimental_designs_names(self) -> Generator[Tuple[str, str], None, None]:
        return self._metabo_experiment.all_experimental_designs_names()

    def get_all_experimental_designs_names(self) -> List[Tuple[str, str]]:
        return list(self._metabo_experiment.all_experimental_designs_names())

    def reset_experimental_designs(self):
        self._metabo_experiment.reset_experimental_designs()

    def remove_experimental_design(self, name: str):
        self._metabo_experiment.remove_experimental_design(name)

    def get_samples_id_from_splits(self, nbr_split_list, design):
        samples_list = []
        for s in nbr_split_list:
            with open(
                os.path.join(DUMP_PATH, design + "_split_{}.p".format(s)), "rb"
            ) as split_file:
                samples_list.append(
                    pickle.load(split_file)[:2]
                )  # append list of X_train & X_test samples names
        return samples_list

    def set_target_column(self, target_column: str):
        self._metabo_experiment.set_target_column(target_column)

    def set_id_column(self, id_column: str):
        self._metabo_experiment.set_id_column(id_column)

    def set_selected_models(self, selected_models: list):
        self._metabo_experiment.set_selected_models(selected_models)

    def learn(self):
        self._metabo_experiment.learn()

    def get_results(self, design_name: str, algo: str):
        return (
            self._metabo_experiment.experimental_designs[design_name]
            .results[algo]
            .results
        )

    def get_all_results(self):
        return self._metabo_experiment.get_all_updated_results()

    def add_custom_model(
        self,
        model_name: str,
        needed_imports: str,
        params: List[str],
        values_to_explore: List[List[str]],
    ):
        self._metabo_experiment.add_custom_model(
            model_name, needed_imports, params, values_to_explore
        )

    def get_all_algos_names(self) -> list:
        return self._metabo_experiment.get_all_algos_names()

    def set_cv_type(self, cv_type: str):
        self._metabo_experiment.set_cv_type(cv_type)

    def get_cv_types(self) -> List[str]:
        return self._metabo_experiment.get_cv_types()

    def get_selected_cv_type(self) -> str:
        return self._metabo_experiment.get_selected_cv_type()

    def generate_save(self) -> MetaboExperimentDTO:
        return self._metabo_experiment.generate_save()

    def is_save_safe(self, saved_metabo_experiment_dto: MetaboExperimentDTO) -> bool:
        return self._metabo_experiment.is_save_safe(saved_metabo_experiment_dto)

    def full_restore(self, saved_metabo_experiment_dto: MetaboExperimentDTO):
        self._metabo_experiment.full_restore(saved_metabo_experiment_dto)

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
        self._metabo_experiment.partial_restore(
            saved_metabo_experiment_dto,
            filename_data,
            filename_metadata,
            data,
            from_base64_data,
            metadata,
            from_base64_metadata,
        )

    def load_results(self, saved_metabo_experiment_dto: MetaboExperimentDTO):
        self._metabo_experiment.load_results(saved_metabo_experiment_dto)

    def get_target_column(self) -> str:
        return self._metabo_experiment.get_target_column()

    def get_id_column(self) -> str:
        return self._metabo_experiment.get_id_column()

    def set_number_of_splits(self, number_of_splits: int):
        self._metabo_experiment.set_number_of_splits(number_of_splits)

    def get_number_of_splits(self) -> int:
        return self._metabo_experiment.get_number_of_splits()

    def set_train_test_proportion(self, train_test_proportion: float):
        self._metabo_experiment.set_train_test_proportion(train_test_proportion)

    def get_train_test_proportion(self) -> float:
        return self._metabo_experiment.get_train_test_proportion()

    def create_splits(self):
        self._metabo_experiment.create_splits()

    def get_selected_models(self) -> List[str]:
        return self._metabo_experiment.get_selected_models()

    def is_progenesis_data(self) -> bool:
        return self._metabo_experiment.is_progenesis_data()

    def get_pairing_group_column(self) -> str:
        return self._metabo_experiment.get_pairing_group_column()

    def set_pairing_group_column(self, pairing_group_column: str):
        self._metabo_experiment.set_pairing_group_column(pairing_group_column)

    def is_data_raw(self) -> bool:
        return self._metabo_experiment.is_data_raw()

    def set_raw_use_for_data(self, use_raw_data: bool):
        self._metabo_experiment.set_raw_use_for_data(use_raw_data)

    def get_data_matrix_remove_rt(self) -> bool:
        return self._metabo_experiment.get_data_matrix_remove_rt()

    def set_data_matrix_remove_rt(self, remove_rt: bool):
        self._metabo_experiment.set_data_matrix_remove_rt(remove_rt)

    def get_cv_folds(self) -> int:
        return self._metabo_experiment.get_cv_folds()

    def set_cv_folds(self, cv_folds: int):
        self._metabo_experiment.set_cv_folds(cv_folds)

    def get_number_of_processes_for_cv(self) -> int:
        return self._metabo_experiment.get_number_of_processes_for_cv()

    def set_number_of_processes_for_cv(self, number_of_processes: int):
        self._metabo_experiment.set_number_of_processes_for_cv(number_of_processes)

    def update_experimental_designs_with_selected_models(self):
        self._metabo_experiment.update_experimental_designs_with_selected_models()

    def is_the_data_matrix_corresponding(self, data: str) -> bool:
        return self._metabo_experiment.is_the_data_matrix_corresponding(data)

    def is_the_metadata_corresponding(self, metadata: str) -> bool:
        return self._metabo_experiment.is_the_metadata_corresponding(metadata)
