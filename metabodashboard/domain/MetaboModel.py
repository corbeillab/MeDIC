import pandas as pd
import sklearn
from sklearn.model_selection import RandomizedSearchCV


# TODO: implement randomSearch
# TODO : get_specific_results, retourne les attributs nécessaires de feat importance pour n'importe quel algo sklearn
# TODO : (suite) , faire un genre de moule d'algo, goulot d'étranglement de nom de méthode
class MetaboModel:
    def __init__(self, model: sklearn, grid_search_configuration: dict, seed: int = 42):
        self.grid_search_param = grid_search_configuration
        self.model = model
        self.seed = seed

    def train(
        self,
        folds: int,
        X_train: pd.DataFrame,
        y_train: list,
        cv_algorithms: sklearn.model_selection,
        number_of_processes: int,
    ) -> sklearn:
        if cv_algorithms == RandomizedSearchCV:
            search = cv_algorithms(
                self.model(random_state=self.seed),
                self.grid_search_param,
                cv=folds,
                random_state=self.seed,
                n_jobs=number_of_processes,
            )
        else:
            search = cv_algorithms(
                self.model(random_state=self.seed),
                self.grid_search_param,
                cv=folds,
                n_jobs=number_of_processes,
            )
        search.fit(X_train, y_train)
        return search.best_estimator_


# TODO: dump best model ?
