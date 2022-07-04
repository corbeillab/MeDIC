import pytest

from ...metabodashboard.domain import MetaboController
from ...metabodashboard.service import Utils
from ..TestsUtility import (
    ENCODED_DATAMATRIX_DATAFRAME,
    SAMPLES_ID_COLUMN,
    TARGETS_COLUMN,
    CLASSES_DESIGN,
    ENCODED_METADATA_DATAFRAME,
    SELECTED_MODELS,
)


@pytest.fixture
def input_controller():
    return MetaboController()


def test_givenIBDMDBDataset_whenLearning_thenNoThrow(input_controller):
    input_controller.set_metadata("metadata.csv", data=ENCODED_METADATA_DATAFRAME)
    input_controller.set_data_matrix_from_path(
        "data.csv", data=ENCODED_DATAMATRIX_DATAFRAME
    )

    input_controller.set_id_column(SAMPLES_ID_COLUMN)
    input_controller.set_target_column(TARGETS_COLUMN)
    input_controller.add_experimental_design(CLASSES_DESIGN)

    input_controller.set_train_test_proportion(0.2)
    input_controller.set_number_of_splits(2)
    input_controller.create_splits()
    input_controller.set_selected_models(SELECTED_MODELS)

    input_controller.learn(2)
