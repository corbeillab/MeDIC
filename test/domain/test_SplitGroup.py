from unittest.mock import patch, mock_open

import pytest

from ...metabodashboard.domain import SplitGroup

from ..TestsUtility import MOCKED_METADATA, TRAIN_TEST_PROPORTION, NUMBER_OF_SPLITS, CLASSES_DESIGN, SPLITS, SELECTED_TARGETS


@pytest.fixture
def input_splits():
    with patch('builtins.open', new_callable=mock_open()):
        return SplitGroup(MOCKED_METADATA, SELECTED_TARGETS, TRAIN_TEST_PROPORTION, NUMBER_OF_SPLITS, CLASSES_DESIGN)


@patch('pickle.load', side_effect=SPLITS)
@patch('builtins.open', new_callable=mock_open())
def test_givenASplitGroup_whenLoadSplitWithIndex_thenTheSplitsAreReproducible(pickle_mock, open_mock, input_splits):
    for split_index in range(NUMBER_OF_SPLITS):
        assert input_splits.load_split_with_index(split_index) == SPLITS[split_index]


def test_givenASplitGroup_whenGetNumberOfSplit_thenNumberOfSplitsIsCorrect(input_splits):
    assert input_splits.get_number_of_splits() == NUMBER_OF_SPLITS

