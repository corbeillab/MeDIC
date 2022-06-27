from unittest.mock import patch, mock_open

import pytest

from ...metabodashboard.domain import DataMatrix

from ..TestsUtility import DATAMATRIX_DATAFRAME, SAMPLES_ID, ENCODED_DATAMATRIX_DATAFRAME, \
    assert_dataframe_approximately_equal, SCALED_DATA


@pytest.fixture
def input_data_matrix():
    data_matrix = DataMatrix()
    return data_matrix


@patch('pickle.load', return_value=DATAMATRIX_DATAFRAME)
@patch('builtins.open', new_callable=mock_open)
def test_givenData_whenLoadData_thenDataIsLoaded(open_mock, pickle_mock, input_data_matrix):
    input_data_matrix.load_data()
    assert input_data_matrix.data.equals(DATAMATRIX_DATAFRAME)


@patch('pickle.dump', return_value=None)
@patch('builtins.open', new_callable=mock_open)
def test_givenBase64Data_whenReadFormat_thenDataIsLoaded(open_mock, pickle_dump, input_data_matrix):
    input_data_matrix.read_format_and_store_data("test.csv", data=ENCODED_DATAMATRIX_DATAFRAME)
    assert_dataframe_approximately_equal(pickle_dump.call_args[0][0], DATAMATRIX_DATAFRAME)


def test_givenData_whenLoadSampleWithEmptyList_thenNoDataIsLoaded(input_data_matrix):
    input_data_matrix.data = DATAMATRIX_DATAFRAME
    assert input_data_matrix.load_samples_corresponding_to_IDs_in_splits([]).equals(DATAMATRIX_DATAFRAME.loc[[], :])


def test_givenData_whenLoadSampleWithIdList_thenTheDataIsLoaded(input_data_matrix):
    input_data_matrix.data = DATAMATRIX_DATAFRAME
    selected_samples = list(SAMPLES_ID[:5])
    assert input_data_matrix.load_samples_corresponding_to_IDs_in_splits(selected_samples).equals(
        DATAMATRIX_DATAFRAME.loc[selected_samples, :])


@patch('sklearn.preprocessing.StandardScaler.transform', return_value=SCALED_DATA)
def test_givenData_whenGettingScaleData_thenTheScaleDataIsLoaded(scaler_mock, input_data_matrix):
    input_data_matrix.data = DATAMATRIX_DATAFRAME
    assert input_data_matrix.get_scaled_data().equals(SCALED_DATA)


def test_givenNoData_whenLoadSampleWithIdList_thenThrowException(input_data_matrix):
    with pytest.raises(RuntimeError):
        input_data_matrix.load_samples_corresponding_to_IDs_in_splits(SAMPLES_ID)


def test_givenNoData_whenGettingScaledData_thenThrowException(input_data_matrix):
    with pytest.raises(RuntimeError):
        input_data_matrix.get_scaled_data()
