import os.path
from typing import List
import base64
import io

import pandas as pd

from ..service import compute_hash

ROOT_PATH = os.path.dirname(__file__)
DUMP_PATH = os.path.join(ROOT_PATH, os.path.join("dumps", "metadata"))
DUMP_METADATA_PATH = os.path.join(DUMP_PATH, "metadata.p")
DUMP_METADATA_COLUMNS_PATH = os.path.join(DUMP_PATH, "metadata_columns.p")
DUMP_SAMPLES_ID_PATH = os.path.join(DUMP_PATH, "samples_id.p")
DUMP_TARGETS_PATH = os.path.join(DUMP_PATH, "targets.p")


class MetaData:
    def __init__(self, metadata_dataframe: pd.DataFrame = None):
        self._dataframe = metadata_dataframe

        self._id_column = None
        self._target_column = None

        self._hash = None

    def read_format_and_store_metadata(self, path, data=None, from_base64=True):
        df = self._load_and_format(path, data=data, from_base64=from_base64)
        self._hash = compute_hash(data)
        self._dataframe = df

    def get_hash(self) -> str:
        return self._hash

    def _load_and_format(self, filename, data=None, from_base64=True) -> pd.DataFrame:
        if from_base64:
            data_type, data_string = data.split(',')
            data = base64.b64decode(data_string)
            print("data decoded :{}")
            print(data[:200])
        else:
            data = filename

        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            if from_base64:
                data = io.StringIO(data.decode('utf-8'))
            df = pd.read_csv(data, sep=None, na_filter=False, engine='python')
        elif 'xls' in filename:
            if from_base64:
                data = io.BytesIO(data)
            # Assume that the user uploaded an excel file
            df = pd.read_excel(data)
        else:
            raise TypeError("The input file is not of the right type, must be excel, odt or csv.")
        return df

    def get_metadata(self) -> pd.DataFrame:
        if self._dataframe is None:
            raise RuntimeError("Try to access the metadata before setting it.")
        return self._dataframe

    def get_columns(self) -> List[str]:
        if self._dataframe is None:
            return []
        return self._dataframe.columns.tolist()

    def get_unique_targets(self) -> List[str]:
        targets = self.get_targets()
        return list(set(targets))

    def set_id_column(self, id_column: str) -> None:
        self._id_column = id_column

    def set_target_column(self, target_column: str) -> None:
        self._target_column = target_column

    def get_target_column(self) -> str:
        return self._target_column

    def get_id_column(self) -> str:
        return self._id_column

    def get_targets(self) -> List[str]:
        if self._target_column is None:
            return []
        return self._dataframe[self._target_column].tolist()

    def get_samples_id(self) -> List[str]:
        if self._id_column is None:
            return []
        return self._dataframe[self._id_column].tolist()

# TODO: join sampleId and target in same pickle file
