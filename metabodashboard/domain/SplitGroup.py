from typing import List, Tuple

from sklearn.model_selection import train_test_split

from . import MetaData
from ..service import Utils


class SplitGroup:
    def __init__(
        self,
        metadata: MetaData,
        selected_targets: List[str],
        train_test_proportion: float,
        number_of_splits: int,
        classes_design: dict,
        pairing_column: str,
    ):
        self._metadata = metadata
        self._number_of_split = number_of_splits
        self._classes_design = classes_design
        self._splits = []
        self._compute_splits(
            train_test_proportion, number_of_splits, pairing_column, selected_targets
        )

    def _compute_splits(
        self,
        train_test_proportion: float,
        number_of_splits: int,
        pairing_column: str,
        selected_targets: List[str],
    ):
        if pairing_column != "":
            sample_ids, targets = self.filter_sample_with_pairing_group(pairing_column)
        else:
            sample_ids, targets = (
                self._metadata.get_samples_id(),
                self._metadata.get_targets(),
            )
        targets, ids = self.get_selected_targets_and_ids(
            selected_targets, sample_ids, targets
        )
        classes = Utils.load_classes_from_targets(self._classes_design, targets)
        for split_index in range(number_of_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                ids, classes, test_size=train_test_proportion, random_state=split_index
            )

            if pairing_column != "":
                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                ) = self.restore_filtered_samples_from_pairing_group(
                    X_train, X_test, pairing_column, self._classes_design
                )
            self._splits.append([X_train, X_test, y_train, y_test])

    def load_split_with_index(self, split_index: int) -> list:
        return self._splits[split_index]

    def get_number_of_splits(self):
        return self._number_of_split

    def filter_sample_with_pairing_group(
        self, pairing_column: str
    ) -> Tuple[List[str], List[str]]:
        metadata_dataframe = self._metadata.get_metadata()
        id_column = self._metadata.get_id_column()
        target_column = self._metadata.get_target_column()
        filtered_id = []
        filtered_target = []
        already_selected_value = set()
        for index, row in metadata_dataframe.iterrows():
            if row[pairing_column] not in already_selected_value:
                already_selected_value.add(row[pairing_column])
                filtered_id.append(row[id_column])
                filtered_target.append(row[target_column])
        return filtered_id, filtered_target

    def restore_filtered_samples_from_pairing_group(
        self,
        X_train: List[str],
        X_test: List[str],
        pairing_column: str,
        classes_design: dict,
    ) -> List[List[str]]:
        metadata_dataframe = self._metadata.get_metadata()
        id_column = self._metadata.get_id_column()
        target_column = self._metadata.get_target_column()
        (
            restored_X_train,
            restored_y_train,
        ) = Utils.restore_ids_and_targets_from_pairing_groups(
            X_train,
            metadata_dataframe,
            id_column,
            pairing_column,
            target_column,
            classes_design,
        )
        (
            restored_X_test,
            restored_y_test,
        ) = Utils.restore_ids_and_targets_from_pairing_groups(
            X_test,
            metadata_dataframe,
            id_column,
            pairing_column,
            target_column,
            classes_design,
        )
        return [restored_X_train, restored_X_test, restored_y_train, restored_y_test]

    def get_selected_targets_and_ids(
        self, selected_targets: List[str], samples_id: List[str], targets: List[str]
    ) -> Tuple[Tuple[str], Tuple[str]]:
        return tuple(
            zip(
                *[
                    (target, id)
                    for target, id in zip(targets, samples_id)
                    if target in selected_targets
                ]
            )
        )
