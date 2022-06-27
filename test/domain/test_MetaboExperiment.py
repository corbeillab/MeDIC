from unittest.mock import patch, mock_open

import pytest as pytest
from sklearn.model_selection import RandomizedSearchCV

from ..TestsUtility import TRAIN_TEST_PROPORTION, NUMBER_OF_SPLITS, CLASSES_DESIGN, EXPERIMENT_NAME, MOCKED_METADATA, \
    MOCKED_METABOEXPERIMENT_DTO, MOCKED_DATAMATRIX, EXPERIMENT_DESIGNS, \
    CV_TYPE, ENCODED_DATAMATRIX_DATAFRAME, ENCODED_METADATA_DATAFRAME, METADATA_DATAFRAME, DATAMATRIX_DATAFRAME, \
    assert_dataframe_approximately_equal, EXP_RESULTS
from ...metabodashboard.domain import MetaboExperiment


@pytest.fixture
def input_metabo_experiment():
    metabo_experiment = MetaboExperiment()
    return metabo_experiment


def test_givenMetaboExperiment_whenAddExperimentalDesign_thenTheExperimentalDesignsAreCorrect(input_metabo_experiment):
    input_metabo_experiment.add_experimental_design(CLASSES_DESIGN)
    assert list(input_metabo_experiment.get_experimental_designs().keys()) == [EXPERIMENT_NAME]


def test_givenMetaboExperiment_whenGetSelectedCvType_thenTheCvTypeIsCorrect(input_metabo_experiment):
    input_metabo_experiment.set_cv_type('RandomizedSearchCV')
    assert input_metabo_experiment.get_selected_cv_type() == "RandomizedSearchCV"


def test_givenMetaboExperiment_whenChangeCvType_thenTheCvTypeIsCorrect(input_metabo_experiment):
    input_metabo_experiment.set_cv_type('RandomizedSearchCV')
    assert input_metabo_experiment.get_cv_algorithm() == RandomizedSearchCV


def test_givenMetaboExperiment_whenChangeCvTypeToIncorrect_thenRaiseValueError(input_metabo_experiment):
    with pytest.raises(ValueError):
        input_metabo_experiment.set_cv_type('alibaba')


def test_givenMetaboExperiment_whenFullRestore_thenMetaboExperimentIsUpdated(input_metabo_experiment):
    input_metabo_experiment.full_restore(MOCKED_METABOEXPERIMENT_DTO)
    assert input_metabo_experiment.get_metadata() == MOCKED_METADATA
    assert input_metabo_experiment.get_data_matrix() == MOCKED_DATAMATRIX
    assert input_metabo_experiment.get_number_of_splits() == NUMBER_OF_SPLITS
    assert input_metabo_experiment.get_train_test_proportion() == TRAIN_TEST_PROPORTION
    assert input_metabo_experiment.get_experimental_designs() == EXPERIMENT_DESIGNS
    assert input_metabo_experiment.get_experimental_designs()[EXPERIMENT_NAME].get_results() == EXP_RESULTS
    assert input_metabo_experiment.get_selected_cv_type() == CV_TYPE


@patch("builtins.open", new_callable=mock_open)
@patch("pickle.dump", return_value=None)
def test_givenMetaboExperiment_whenPartialRestore_thenMetaboExperimentIsUpdated(dump_patch, open_patch, input_metabo_experiment):
    input_metabo_experiment.partial_restore(MOCKED_METABOEXPERIMENT_DTO, "metadata.csv", "data_matrix.csv",
                                            data=ENCODED_DATAMATRIX_DATAFRAME, metadata=ENCODED_METADATA_DATAFRAME)
    dumped_data_matrix_dataframe = dump_patch.call_args_list[0][0][0]
    assert_dataframe_approximately_equal(dumped_data_matrix_dataframe, DATAMATRIX_DATAFRAME)
    assert input_metabo_experiment.get_metadata().get_metadata().equals(METADATA_DATAFRAME)
    assert input_metabo_experiment.get_number_of_splits() == NUMBER_OF_SPLITS
    assert input_metabo_experiment.get_train_test_proportion() == TRAIN_TEST_PROPORTION
    assert input_metabo_experiment.get_experimental_designs() == EXPERIMENT_DESIGNS
    assert input_metabo_experiment.get_experimental_designs()[EXPERIMENT_NAME].get_results() == EXP_RESULTS
    assert input_metabo_experiment.get_selected_cv_type() == CV_TYPE


def test_givenMetaboExperiment_whenLoadResults_thenResultsAreLoaded(input_metabo_experiment):
    input_metabo_experiment.load_results(MOCKED_METABOEXPERIMENT_DTO)
    assert input_metabo_experiment.get_metadata().get_hash() is None
    assert input_metabo_experiment.get_data_matrix().get_hash() is None
    assert input_metabo_experiment.get_number_of_splits() == NUMBER_OF_SPLITS
    assert input_metabo_experiment.get_train_test_proportion() == TRAIN_TEST_PROPORTION
    assert input_metabo_experiment.get_experimental_designs() == EXPERIMENT_DESIGNS
    assert input_metabo_experiment.get_experimental_designs()[EXPERIMENT_NAME].get_results() == EXP_RESULTS
    assert input_metabo_experiment.get_selected_cv_type() == CV_TYPE
