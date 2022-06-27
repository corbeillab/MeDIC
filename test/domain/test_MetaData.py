import pytest as pytest

from ...metabodashboard.domain.MetaData import MetaData

from .TestsUtility import METADATA_DATAFRAME, SAMPLES_ID_COLUMN, TARGETS_COLUMN, SAMPLES_ID, TARGETS


@pytest.fixture
def input_meta_data():
    return MetaData(METADATA_DATAFRAME)


def test_givenMetadata_whenGetMetadata_thenMetadataIsCorrect(input_meta_data):
    assert input_meta_data.get_metadata().equals(METADATA_DATAFRAME)


def test_givenSampleIDColumn_whenGetSamplesID_thenSamplesIDAreCorrect(input_meta_data):
    input_meta_data.set_id_column(SAMPLES_ID_COLUMN)
    assert input_meta_data.get_samples_id() == SAMPLES_ID


def test_givenNoSampleIDColumn_whenSamplesID_thenEmptyList(input_meta_data):
    assert input_meta_data.get_samples_id() == []


def test_givenTargetColumn_whenGetTargets_thenTargetsAreCorrect(input_meta_data):
    input_meta_data.set_target_column(TARGETS_COLUMN)
    assert input_meta_data.get_targets() == TARGETS


def test_givenNoTargetColumn_whenGetTargets_thenEmptyList(input_meta_data):
    assert input_meta_data.get_targets() == []

