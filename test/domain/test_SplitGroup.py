from unittest.mock import patch, mock_open

import pytest

from ..TestsUtility import (
    MOCKED_METADATA,
    TRAIN_TEST_PROPORTION,
    NUMBER_OF_SPLITS,
    CLASSES_DESIGN,
    SPLITS,
    PAIRING_GROUP_COLUMN,
    GROUPED_ID,
    GROUPED_TARGETS,
    SAMPLES_ID,
    TARGETS,
    SELECTED_TARGETS,
    ALL_TARGETS,
)
from ...metabodashboard.domain import SplitGroup


@pytest.fixture
def input_splits():
    return SplitGroup(
        MOCKED_METADATA,
        ALL_TARGETS,
        TRAIN_TEST_PROPORTION,
        NUMBER_OF_SPLITS,
        CLASSES_DESIGN,
        "",
    )


def test_givenASplitGroup_whenLoadSplitWithIndex_thenTheSplitsAreReproducible(
    input_splits,
):
    for split_index in range(NUMBER_OF_SPLITS):
        assert input_splits.load_split_with_index(split_index) == SPLITS[split_index]


def test_givenASplitGroup_whenGetNumberOfSplit_thenNumberOfSplitsIsCorrect(
    input_splits,
):
    assert input_splits.get_number_of_splits() == NUMBER_OF_SPLITS


def test_givenASplitGroup_whenFilterSample_thenSamplesAreFiltered(input_splits):
    assert input_splits.filter_sample_with_pairing_group(PAIRING_GROUP_COLUMN) == (
        GROUPED_ID,
        GROUPED_TARGETS,
    )


# def test_givenASplitGroup_whenRestoreFilteredSamples_thenSamplesAreRestored(
#     input_splits,
# ):
#     (
#         restored_X_train,
#         restored_X_test,
#         restored_y_train,
#         restored_y_test,
#     ) = input_splits.restore_filtered_samples_from_pairing_group(
#         GROUPED_ID, GROUPED_TARGETS, PAIRING_GROUP_COLUMN, CLASSES_DESIGN
#     )
#
#     X = restored_X_train + restored_X_test
#     X.sort()
#     y = restored_y_train + restored_y_test
#     y.sort()
#
#     assert len(X) == len(SAMPLES_ID)
#     assert len(y) == len(SAMPLES_ID)
#     assert X == SAMPLES_ID
