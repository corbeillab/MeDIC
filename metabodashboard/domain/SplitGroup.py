from typing import List

from sklearn.model_selection import train_test_split

from . import MetaData
from ..service import Utils


class SplitGroup:
    def __init__(self, metadata: MetaData, selected_targets: List[str], train_test_proportion: float, number_of_splits: int, classes_design: dict):
        self._metadata = metadata
        self._number_of_split = number_of_splits
        self._classes_design = classes_design
        self._splits = []
        self._compute_splits(train_test_proportion, number_of_splits, selected_targets)

    def _compute_splits(self, train_test_proportion: float, number_of_splits: int, selected_targets: List[str]) -> None:
        targets, ids = self._metadata.get_selected_targets_and_ids(selected_targets)
        classes = Utils.load_classes_from_targets(self._classes_design, targets)
        for split_index in range(number_of_splits):
            X_train, X_test, y_train, y_test = train_test_split(ids,
                                                                classes,
                                                                test_size=train_test_proportion,
                                                                random_state=split_index)

            self._splits.append([X_train, X_test, y_train, y_test])

    def load_split_with_index(self, split_index: int) -> list:
        return self._splits[split_index]

    def get_number_of_splits(self):
        return self._number_of_split
