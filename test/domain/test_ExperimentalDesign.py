from unittest.mock import patch, mock_open

import pytest as pytest

from ...metabodashboard.domain import ExperimentalDesign

from ..TestsUtility import (
    MOCKED_METADATA,
    CLASSES_DESIGN,
    TRAIN_TEST_PROPORTION,
    NUMBER_OF_SPLITS,
    EXPERIMENT_NAME,
    EXPERIMENT_FULL_NAME,
    SPLITS,
    SELECTED_TARGETS,
    PARTIAL_CLASSES_DESIGN,
)


@pytest.fixture
def input_experimental_design():
    experimental_design = ExperimentalDesign(CLASSES_DESIGN)
    return experimental_design


@pytest.fixture
def input_experimental_design_with_partial():
    experimental_design = ExperimentalDesign(PARTIAL_CLASSES_DESIGN)
    return experimental_design


def test_givenAnExperimentalDesign_whenGetNumberOfSplit_thenNumberOfSplitsIsCorrect(
    input_experimental_design,
):
    input_experimental_design.set_split_parameter_and_compute_splits(
        TRAIN_TEST_PROPORTION, NUMBER_OF_SPLITS, MOCKED_METADATA, ""
    )
    assert input_experimental_design.get_number_of_splits() == NUMBER_OF_SPLITS


def test_givenAnExperimentalDesign_whenGetAllSplit_thenTheSplitsAreReproducible(
    input_experimental_design,
):
    input_experimental_design.set_split_parameter_and_compute_splits(
        TRAIN_TEST_PROPORTION, NUMBER_OF_SPLITS, MOCKED_METADATA, ""
    )
    for split_index, (real_split_index, actual_split) in enumerate(
        input_experimental_design.all_splits()
    ):
        print("\n Split n{}".format(split_index))
        assert split_index == real_split_index
        real_X_train, real_X_test, real_y_train, real_y_test = actual_split
        X_train, X_test, y_train, y_test = SPLITS[split_index]
        assert real_X_train == X_train
        assert real_X_test == X_test
        assert real_y_train == y_train
        assert real_y_test == y_test


def test_givenAnExperimentalDesign_whenGetName_thenTheNameIsCorrect(
    input_experimental_design,
):
    assert input_experimental_design.get_name() == EXPERIMENT_NAME


def test_givenAnExperimentalDesign_whenGetFullName_thenTheFullNameIsCorrect(
    input_experimental_design,
):
    assert input_experimental_design.get_full_name() == EXPERIMENT_FULL_NAME


def test_givenAnExperimentalDesign_whenGetClassesDesign_thenClassesDesignIsCorrect(
    input_experimental_design,
):
    assert input_experimental_design.get_classes_design() == CLASSES_DESIGN


def test_givenAnExperimentalDesignWithNoSelectedModels_whenGetResults_thenRaiseRuntimeError(
    input_experimental_design,
):
    with pytest.raises(RuntimeError) as e_info:
        input_experimental_design.get_results()
        assert "Trying to set models before setting splits parameters" in str(
            e_info.value
        )


def test_givenAnExperimentalDesign_whenGetSelectedTargetsName_thenTheSelectedTargetsNameAreCorrect(
    input_experimental_design_with_partial,
):
    assert (
        input_experimental_design_with_partial.get_selected_targets_name()
        == SELECTED_TARGETS
    )
