from typing import Generator, Tuple, Dict, List

from . import SplitGroup, MetaData
from .Results import *


class ExperimentalDesign:
    def __init__(self, classes_design: dict):
        self._classes_design = classes_design
        self._name = ""
        self._compute_name()

        self._split_group = None

        self._selected_models_name = None
        self.results = {}

        self._is_done = False

    def get_is_done(self) -> bool:
        return self._is_done

    def set_is_done(self, is_done: bool) -> None:
        self._is_done = is_done

    def set_split_parameter_and_compute_splits(
        self,
        train_test_proportion: float,
        number_of_splits: int,
        metadata: MetaData,
        pairing_column: str,
    ) -> None:
        self._split_group = SplitGroup(
            metadata,
            self.get_selected_targets_name(),
            train_test_proportion,
            number_of_splits,
            self._classes_design,
            pairing_column,
        )

    def get_name(self) -> str:
        return self._name

    def get_full_name(self) -> str:
        name = []
        for key, item_list in self._classes_design.items():
            name.append(f"{key} ({', '.join(item_list)})")
        return " versus ".join(name)

    def get_classes_design(self) -> dict:
        return self._classes_design

    def set_selected_models_name(self, selected_models_name: list) -> None:
        if self._split_group is None:
            raise ValueError("Trying to set models before setting splits parameters")
        self._selected_models_name = selected_models_name
        # TODO : un genre d'emballage de classe results pour pouvoir appeler juste un nom de classe
        for n in self._selected_models_name:
            if n == "RandomForest":
                self.results[n] = ResultsRF(self._split_group.get_number_of_splits())
            elif n == "DecisionTree":
                self.results[n] = ResultsDT(self._split_group.get_number_of_splits())
            elif n == "SCM":
                self.results[n] = ResultsSCM(self._split_group.get_number_of_splits())
            elif n == "RandomSCM":
                self.results[n] = ResultsRSCM(self._split_group.get_number_of_splits())
            else:
                raise NotImplementedError(f"{n} is not implemented")

    def get_results(self) -> Dict[str, Results]:
        if self.results == {}:
            raise RuntimeError(
                "The name of the selected models has to be set before accessing results."
            )
        return self.results

    def _compute_name(self) -> None:
        self._name = "_vs_".join(self._classes_design)

    def get_number_of_splits(self) -> int:
        return self._split_group.get_number_of_splits()

    def all_splits(self) -> Generator[Tuple[int, list], None, None]:
        if self._split_group is None:
            raise RuntimeError(
                "Trying to access Splits before setting splits parameters"
            )
        for split_index in range(self._split_group.get_number_of_splits()):
            yield split_index, self._split_group.load_split_with_index(split_index)

    def get_selected_targets_name(self) -> list:
        return list(Utils.reverse_dict(self._classes_design).keys())
