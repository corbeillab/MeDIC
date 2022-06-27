from unittest.mock import patch, Mock, mock_open

import pandas as pd
import pytest as pytest
from sklearn.model_selection import train_test_split

from ...metabodashboard.domain import ExperimentalDesign, MetaData, SplitGroup

from .TestsUtility import MOCKED_METADATA, CLASSES_DESIGN, TRAIN_TEST_PROPORTION, NUMBER_OF_SPLITS, \
    EXPERIMENT_NAME, EXPERIMENT_FULL_NAME, SPLITS


@pytest.fixture
def input_experimental_design():
    experimental_design = ExperimentalDesign(CLASSES_DESIGN)
    return experimental_design


@patch('builtins.open', new_callable=mock_open())
def test_givenAnExperimentalDesign_whenGetNumberOfSplit_thenNumberOfSplitsIsCorrect(open_mock, input_experimental_design):
    input_experimental_design.set_split_parameter(TRAIN_TEST_PROPORTION, NUMBER_OF_SPLITS, MOCKED_METADATA)
    assert input_experimental_design.get_number_of_splits() == NUMBER_OF_SPLITS


@patch('pickle.load', side_effect=SPLITS)
@patch('builtins.open', new_callable=mock_open())
def test_givenAnExperimentalDesign_whenGetAllSplit_thenTheSplitsAreReproducible(open_mock, pickle_mock, input_experimental_design):
    input_experimental_design.set_split_parameter(TRAIN_TEST_PROPORTION, NUMBER_OF_SPLITS, MOCKED_METADATA)
    for split_index, (real_split_index, actual_split) in enumerate(input_experimental_design.all_splits()):
        assert split_index == real_split_index
        real_X_train, real_X_test, real_y_train, real_y_test = actual_split
        X_train, X_test, y_train, y_test = SPLITS[split_index]
        assert real_X_train == X_train
        assert real_X_test == X_test
        assert real_y_train == y_train
        assert real_y_test == y_test


def test_givenAnExperimentalDesign_whenGetName_thenTheNameIsCorrect(input_experimental_design):
    assert input_experimental_design.get_name() == EXPERIMENT_NAME


def test_givenAnExperimentalDesign_whenGetFullName_thenTheFullNameIsCorrect(input_experimental_design):
    assert input_experimental_design.get_full_name() == EXPERIMENT_FULL_NAME


def test_givenAnExperimentalDesign_whenGetClassesDesign_thenClassesDesignIsCorrect(input_experimental_design):
    assert input_experimental_design.get_classes_design() == CLASSES_DESIGN


def test_givenAnExperimentalDesignWithNoSelectedModels_whenGetResults_thenRaiseRuntimeError(input_experimental_design):
    with pytest.raises(RuntimeError) as e_info:
        input_experimental_design.get_results()
        assert "Trying to set models before setting splits parameters" in str(e_info.value)
