import pytest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from ..TestsUtility import FOLDS, DATA, CLASSES, PARAMETER_GRID
from ...metabodashboard.domain.MetaboModel import MetaboModel


@pytest.fixture
def input_metabomodel():
    return MetaboModel(DecisionTreeClassifier, PARAMETER_GRID)


def test_givenModel_whenTuningWithGridSearch_thenReturnBestModel(input_metabomodel):
    best_model = input_metabomodel.train(FOLDS, DATA, CLASSES, GridSearchCV)
    real_model = GridSearchCV(
        DecisionTreeClassifier(random_state=42), PARAMETER_GRID, cv=FOLDS
    )
    real_model.fit(DATA, CLASSES)
    assert best_model.get_params() == real_model.best_estimator_.get_params()


def test_givenModel_whenTuningWithRandomizedSearch_thenReturnBestModel(
    input_metabomodel,
):
    real_model = RandomizedSearchCV(
        DecisionTreeClassifier(random_state=42),
        PARAMETER_GRID,
        cv=FOLDS,
        random_state=42,
    )
    real_model.fit(DATA, CLASSES)
    for i in range(10):
        print(f"Run {i}")
        best_model = input_metabomodel.train(FOLDS, DATA, CLASSES, RandomizedSearchCV)
        assert best_model.get_params() == real_model.best_estimator_.get_params()
