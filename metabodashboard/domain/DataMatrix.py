import os
import pickle
from typing import Tuple, Optional, Iterable

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..service import DataFormat, compute_hash, Utils

ROOT_PATH = os.path.dirname(__file__)
DUMP_PATH = os.path.join(ROOT_PATH, "dumps")
DUMP_DATA_MATRIX_PATH = os.path.join(DUMP_PATH, "datamatrix.p")


class DataMatrix:
    def __init__(self):
        # TODO : implémenter test format de matrice (progenesis -> ML ready)
        self.data = None
        self._scaler = StandardScaler()
        self._hash = None
        self._use_raw = None
        self._remove_rt = True

    def reset_file(self):
        Utils.reset_file(DUMP_DATA_MATRIX_PATH)

    def read_format_and_store_data(
        self,
        path: str,
        data=None,
        from_base64: bool = True,
    ) -> Optional[pd.DataFrame]:
        if self._use_raw is None:
            raise RuntimeError("Need to set raw use before loading data")
        data_df, metadata_df = self._load_and_format(
            path, data=data, is_raw=self._use_raw, from_base64=from_base64
        )

        if self._remove_rt:
            features_to_drop = []
            for f in data_df.columns:
                if float(f.split("_")[0]) < 1:
                    features_to_drop.append(f)
            data_df = data_df.drop(columns=features_to_drop)

        if data is not None:
            self._hash = compute_hash(data)

        with open(DUMP_DATA_MATRIX_PATH, "w+b") as data_matrix_file:
            pickle.dump(data_df, data_matrix_file)

        self._scaler.fit(data_df)

        if metadata_df is not None:
            return metadata_df

    def get_hash(self) -> str:
        """
        :return: the hash of the dataframe
        """
        return self._hash

    def is_raw(self) -> bool:
        return self._use_raw

    def get_scaled_data(self, selected_ids: Iterable = None) -> pd.DataFrame:
        """
        Scale the dataframe to be ready for ML
        :param data: the dataframe to scale
        :param selected_ids: the list of IDs to scale
        :return: the scaled dataframe
        """
        if self.data is None:
            raise RuntimeError("Need to load data from file before scaling")
        if selected_ids is not None:
            df = self.data.loc[selected_ids, :]
        else:
            df = self.data
        return pd.DataFrame(self._scaler.transform(df), columns=self.data.columns)

    def _load_and_format(
        self, path, data=None, is_raw=False, from_base64=True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        load the table from a path and process it to make it more easy to manipulate
        """
        formater = DataFormat(
            path, data=data, use_raw=is_raw, from_base64_str=from_base64
        )
        datatable_compoundsInfo, datatable, labels, sample_names = formater.convert()
        if labels is None or sample_names is None:
            return datatable, None
        return datatable, pd.DataFrame({"sample_names": sample_names, "labels": labels})

    def load_data(self):  # loadDataMatrix
        with open(DUMP_DATA_MATRIX_PATH, "rb") as data_matrix_file:
            self.data = pickle.load(data_matrix_file)

    def load_samples_corresponding_to_IDs_in_splits(
        self, id_list: list
    ) -> pd.DataFrame:
        """
        self.data is supposed to have samples as lines, so we select only the samples we need
        with their IDS(lines indexes)
        Also, the way of selecting the samples assures the samples are in the same order as the list, which is
        it self in the same order as the target list
        :return: the reduced dataframe, the samples are lines and the matrix is ML ready
        """
        if self.data is None:
            raise RuntimeError(
                "Need to load data from file before extracting specific samples"
            )
        df = self.data.loc[id_list, :]
        return df

    def unload_data(self):
        self.data = None

    def set_raw_use(self, use_raw: bool):
        self._use_raw = use_raw

    def set_remove_rt(self, remove_rt: bool):
        self._remove_rt = remove_rt

    def get_remove_rt(self) -> bool:
        return self._remove_rt
