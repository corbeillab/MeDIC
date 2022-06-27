import base64
import hashlib
import random
from math import isclose
from typing import List
from unittest.mock import Mock

import numpy as np
import pandas as pd
from pyscm import SetCoveringMachineClassifier
from randomscm.randomscm import RandomScmClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ...metabodashboard.conf.SupportedModels import LEARN_CONFIG


def _get_samples_id(size: int) -> List[str]:
    return ["patient-" + str(patient_id) for patient_id in range(size)]


def _get_random_data(size: int, number_of_columns: int):
    data = np.random.rand(size, number_of_columns)
    column = ["feature-" + str(index) for index in range(number_of_columns)]
    return pd.DataFrame(data, columns=column)


def _get_targets_and_classes(size: int, classes_design: dict):
    target_list = [target
                   for _, target_list in classes_design.items()
                   for target in target_list]
    number_of_target = len(target_list) - 1
    reversed_classes_design = {target: class_
                               for class_, target_list in classes_design.items()
                               for target in target_list}
    targets = []
    classes = []
    for index in range(size):
        target_index = random.randint(0, number_of_target)
        target = target_list[target_index]
        targets.append(target)
        classes.append(reversed_classes_design[target])

    return targets, classes


def _get_splits(number_of_splits: int, train_test_proportion: float, samples_id: list, classes: list) -> List[
    List[str]]:
    splits = []
    for split_index in range(number_of_splits):
        X_train, X_test, y_train, y_test = train_test_split(samples_id, classes, test_size=train_test_proportion,
                                                            random_state=split_index)
        splits.append([X_train, X_test, y_train, y_test])
    return splits


def base64_encode_dataframe(data: pd.DataFrame) -> str:
    return "data:application/vnd.ms-excel;base64," + base64.b64encode(data.to_csv(index=True).encode("utf-8")).decode(
        "utf-8")


def base64_encode_metadata(data: pd.DataFrame) -> str:
    return "data:application/vnd.ms-excel;base64," + base64.b64encode(data.to_csv(index=False).encode("utf-8")).decode(
        "utf-8")


def compute_hash(data: pd.DataFrame) -> str:
    return hashlib.sha256(data.to_csv(index=False).encode('utf-8')).hexdigest()


def assert_dataframe_approximately_equal(data1: pd.DataFrame, data2: pd.DataFrame, epsilon: float = 1e-15):
    for col, item in data2.items():
        for index, value in item.items():
            assert isclose(data1.loc[index, col], value, abs_tol=epsilon)


def _get_random_results(number_of_split: int):
    results = {str(s): {} for s in range(number_of_split)}
    for split_number in range(number_of_split):
        for metric in ["train_accuracy", "test_accuracy", "balanced_train_accuracy", "balanced_test_accuracy",
                       "train_precision", "test_precision", "train_recall", "test_recall", "train_f1", "test_f1",
                       "train_roc_auc", "test_roc_auc"]:
            results[str(split_number)][metric] = random.random()

    return results



SIZE = 100
COLUMNS = 1000

EXPERIMENT_NAME = "sick_vs_healthy"
EXPERIMENT_FULL_NAME = "sick (sick, ill) versus healthy (healthy)"

CLASSES_DESIGN = {"sick": ["sick", "ill"], "healthy": ["healthy"]}

NUMBER_OF_SPLITS = 10
TRAIN_TEST_PROPORTION = 0.75

SAMPLES_ID = _get_samples_id(SIZE)
TARGETS, CLASSES = _get_targets_and_classes(SIZE, CLASSES_DESIGN)
DATA = _get_random_data(SIZE, COLUMNS)
SCALED_DATA = pd.DataFrame(StandardScaler().fit_transform(DATA), columns=DATA.columns)

SAMPLES_ID_COLUMN = "samples_id"
TARGETS_COLUMN = "target"

METADATA_DATAFRAME = pd.DataFrame({SAMPLES_ID_COLUMN: SAMPLES_ID, TARGETS_COLUMN: TARGETS})
ENCODED_METADATA_DATAFRAME = base64_encode_metadata(METADATA_DATAFRAME)
METADATA_DATAFRAME_HASH = compute_hash(METADATA_DATAFRAME)

DATAMATRIX_DATAFRAME = pd.concat([pd.DataFrame({SAMPLES_ID_COLUMN: SAMPLES_ID}), DATA], axis=1)
DATAMATRIX_DATAFRAME.set_index(SAMPLES_ID_COLUMN, inplace=True)
ENCODED_DATAMATRIX_DATAFRAME = base64_encode_dataframe(DATAMATRIX_DATAFRAME)
DATAMATRIX_DATAFRAME_HASH = compute_hash(DATAMATRIX_DATAFRAME)

DIFFERENT_DATAMATRIX_DATAFRAMES = pd.concat(
    [pd.DataFrame({SAMPLES_ID_COLUMN: SAMPLES_ID}), _get_random_data(SIZE, COLUMNS)], axis=1)
DIFFERENT_DATAMATRIX_DATAFRAMES.set_index(SAMPLES_ID_COLUMN, inplace=True)
ENCODED_DIFFERENT_DATAMATRIX_DATAFRAMES = base64_encode_dataframe(DIFFERENT_DATAMATRIX_DATAFRAMES)

MOCKED_METADATA_CLASS = Mock(name="MockedMetadata")
MOCKED_METADATA = MOCKED_METADATA_CLASS.return_value
MOCKED_METADATA.get_metadata.return_value = METADATA_DATAFRAME
MOCKED_METADATA.get_samples_id.return_value = SAMPLES_ID
MOCKED_METADATA.get_targets.return_value = TARGETS
MOCKED_METADATA.get_hash.return_value = METADATA_DATAFRAME_HASH

MOCKED_DATAMATRIX_CLASS = Mock(name="MockedDatamatrix")
MOCKED_DATAMATRIX = MOCKED_DATAMATRIX_CLASS.return_value
MOCKED_DATAMATRIX.load_data.return_value = DATAMATRIX_DATAFRAME
MOCKED_DATAMATRIX.get_hash.return_value = DATAMATRIX_DATAFRAME_HASH

RESULTS = _get_random_results(NUMBER_OF_SPLITS)

MOCKED_RESULT_CLASS = Mock(name="MockedResult")
MOCKED_RESULT = MOCKED_RESULT_CLASS.return_value

EXP_RESULTS = {"DecisionTree": MOCKED_RESULT, "RandomForest": MOCKED_RESULT, "LogisticRegression": MOCKED_RESULT}

MOCKED_EXPERIMENTAL_DESIGN_CLASS = Mock(name="MockedExperimentalDesign")
MOCKED_EXPERIMENTAL_DESIGN = MOCKED_EXPERIMENTAL_DESIGN_CLASS.return_value
MOCKED_EXPERIMENTAL_DESIGN.get_classes_design.return_value = CLASSES_DESIGN
MOCKED_EXPERIMENTAL_DESIGN.get_name.return_value = EXPERIMENT_NAME
MOCKED_EXPERIMENTAL_DESIGN.get_full_name.return_value = EXPERIMENT_FULL_NAME
MOCKED_EXPERIMENTAL_DESIGN.get_results.return_value = EXP_RESULTS

EXPERIMENT_DESIGNS = {EXPERIMENT_NAME: MOCKED_EXPERIMENTAL_DESIGN}

MOCKED_CUSTOM_MODEL_CLASS = Mock(name="MockedCustomModel")
MOCKED_CUSTOM_MODEL = MOCKED_CUSTOM_MODEL_CLASS.return_value

CUSTOM_MODELS = {
    "Custom model": MOCKED_CUSTOM_MODEL,
}

SELECTED_MODELS = ["RandomForestClassifier", "DecisionTreeClassifier"]
CV_TYPE = "GridSearchCV"

MOCKED_METABOEXPERIMENT_CLASS = Mock(name="MockedMetaboExperiment")
MOCKED_METABOEXPERIMENT = MOCKED_METABOEXPERIMENT_CLASS.return_value
MOCKED_METABOEXPERIMENT.get_metadata.return_value = MOCKED_METADATA
MOCKED_METABOEXPERIMENT.get_data_matrix.return_value = MOCKED_DATAMATRIX
MOCKED_METABOEXPERIMENT.get_number_of_splits.return_value = NUMBER_OF_SPLITS
MOCKED_METABOEXPERIMENT.get_train_test_proportion.return_value = TRAIN_TEST_PROPORTION
MOCKED_METABOEXPERIMENT.get_experimental_designs.return_value = EXPERIMENT_DESIGNS
MOCKED_METABOEXPERIMENT.get_custom_models.return_value = CUSTOM_MODELS
MOCKED_METABOEXPERIMENT.get_selected_models.return_value = SELECTED_MODELS
MOCKED_METABOEXPERIMENT.get_selected_cv_type.return_value = CV_TYPE

MOCKED_METABOEXPERIMENT_DTO_CLASS = Mock(name="MockedMetaboExperimentDTO")
MOCKED_METABOEXPERIMENT_DTO = MOCKED_METABOEXPERIMENT_DTO_CLASS.return_value
MOCKED_METABOEXPERIMENT_DTO.metadata = MOCKED_METADATA
MOCKED_METABOEXPERIMENT_DTO.data_matrix = MOCKED_DATAMATRIX
MOCKED_METABOEXPERIMENT_DTO.number_of_splits = NUMBER_OF_SPLITS
MOCKED_METABOEXPERIMENT_DTO.train_test_proportion = TRAIN_TEST_PROPORTION
MOCKED_METABOEXPERIMENT_DTO.experimental_designs = EXPERIMENT_DESIGNS
MOCKED_METABOEXPERIMENT_DTO.custom_models = CUSTOM_MODELS
MOCKED_METABOEXPERIMENT_DTO.selected_models = SELECTED_MODELS
MOCKED_METABOEXPERIMENT_DTO.selected_cv_type = CV_TYPE

SPLITS = _get_splits(NUMBER_OF_SPLITS, TRAIN_TEST_PROPORTION, SAMPLES_ID, CLASSES)

FOLDS = 5
PARAMETER_GRID = {
    'criterion': ['gini', 'entropy'],
    "max_depth": [1, 2, 3, 4, 5, 10]
}

SUPPORTED_MODEL = LEARN_CONFIG
