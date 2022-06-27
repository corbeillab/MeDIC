import pytest

from ...metabodashboard.domain import ResultsDT
from .TestsUtility import NUMBER_OF_SPLITS, RESULTS


@pytest.fixture
def input_results():
    return ResultsDT(NUMBER_OF_SPLITS)


def test_givenResults_whenProduceMetricsTable_thenReturnTable(input_results):
    input_results.results = RESULTS
    metrics_table = input_results.produce_metrics_table()
    print(input_results.produce_metrics_table())
