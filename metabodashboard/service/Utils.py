import base64
import pickle
from typing import List, Dict

import pandas as pd
import numpy as np
import pickle as pkl
import os
import hashlib

PACKAGE_ROOT_PATH = os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1])
DUMP_PATH = os.path.join(PACKAGE_ROOT_PATH, "domain", "dumps")
DUMP_EXPE_PATH = os.path.join(DUMP_PATH, "save.mtxp")


def dump_metabo_expe(obj):
    with open(DUMP_EXPE_PATH, "w+b") as expe_file:
        pkl.dump(obj, expe_file)


def get_metabo_experiment_path() -> str:
    return DUMP_EXPE_PATH


def load_metabo_expe(path):
    if os.path.isfile(DUMP_EXPE_PATH):
        with open(path, "rb") as expe_file:
            return pkl.load(expe_file)
    else:
        return None


def retrieve_data_from_sample_name(names_list, dataframe):
    """

    :param names_list: list of samples name
    :param dataframe: a dataframe with each sample as a line identified by the samples' name
    :return: list of data
    """
    print("retrieving data from name")
    data_list = []
    for n in names_list:
        d = dataframe.loc[n, :]
        data_list.append(d.tolist())
    # print("data list : {}".format(data_list[0]))
    print("data from name retrieved")
    # print("data 2nd element : {}".format(data_list[1]))
    return data_list


def list_filler(liste):
    """
    (NA or "")
    Complete the NA values of a list with the last non NA value from left to right.
    If the first value is NA, then leave it like this.
    :param liste: list to fill
    :return: new list filled
    """
    l = []
    current = ""
    for idx, j in enumerate(liste):
        if j != "":
            current = j
        l.append(current)
    return l

def read_Progenesis_compounds_table(fileName, with_raw=True):
    datatable = pd.read_csv(fileName, header=2, index_col=0)
    header = pd.read_csv(fileName, nrows=1, index_col=0)
    start_normalized = header.columns.tolist().index("Normalised abundance")

    labels_array = np.array(header.iloc[0].tolist())
    possible_labels = labels_array[labels_array != 'nan']

    if with_raw:
        start_raw = header.columns.tolist().index("Raw abundance")
        sample_names = datatable.iloc[:, start_normalized:start_raw].columns
        possible_labels = possible_labels[0:int(len(possible_labels) / 2)]
    else:
        sample_names = datatable.iloc[:, start_normalized:].columns
    
    labels = [""] * len(sample_names)
    start_label = possible_labels[0]
    labels_array = labels_array.tolist()
    for next_labels in possible_labels[1:]:
        index_s = labels_array.index(start_label) - start_normalized
        index_e = labels_array.index(next_labels) - start_normalized
        labels[index_s : index_e] = [start_label] * (index_e - index_s)
        start_label = next_labels
    labels[index_e:] = [start_label] * (len(labels) - index_e)
    
    labels_dict = {sample_names[i] : j for i,j in enumerate(labels)}

    if with_raw:
        datatable_compoundsInfo = datatable.iloc[:,0:start_normalized]
        datatable_normalized = datatable.iloc[:,start_normalized:start_raw]
        datatable_raw = datatable.iloc[:,start_raw:]
        datatable_raw.columns = [i.rstrip(".1") for i in datatable_raw.columns] #Fix the columns names

        datatable_normalized = datatable_normalized.T
        datatable_raw = datatable_raw.T
        datatable_compoundsInfo = datatable_compoundsInfo.T
        datatable_normalized.rename(columns={"Compound": "Sample"})
        datatable_raw.rename(columns={"Compound": "Sample"})
        return datatable_compoundsInfo, datatable_normalized, datatable_raw, labels, sample_names
    else:
        datatable_compoundsInfo = datatable.iloc[:,0:start_normalized]
        datatable_normalized = datatable.iloc[:,start_normalized:]
        datatable_normalized = datatable_normalized.T
        datatable_compoundsInfo = datatable_compoundsInfo.T
        datatable_normalized.rename(columns={"Compound": "Sample"})
        return datatable_compoundsInfo, datatable_normalized, labels, sample_names

def filter_sample_based_on_labels(data, labels, labels_to_keep):
    labels_filter = np.array([i in labels_to_keep for i in labels])
    d = data.iloc[labels_filter]
    l = np.array(labels)[labels_filter]
    return d, l

def get_group_to_class(classes):
    group_to_class = {}
    for class_name in classes:
        for subgroup in classes[class_name]:
            group_to_class[subgroup] = class_name
    return group_to_class


def reverse_dict(dictionnary: dict) -> dict:
    reversed_dict = {}
    for key, value in dictionnary.items():
        if type(value) is list:
            for val in value:
                reversed_dict[val] = key
        else:
            reversed_dict[value] = key
    return reversed_dict


def load_classes_from_targets(classes_design: dict, targets: List[str]) -> List[str]:
    reverse_classes_design = reverse_dict(classes_design)
    classes = []
    for target in targets:
        classes.append(reverse_classes_design[target])
    return classes


# TODO: need to support multi-classification
def get_binary(list_to_convert: List[str], classes: List[str]) -> List[int]:
    return [1 if class_value == classes[1] else 0 for class_value in list_to_convert]


def compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def is_save_safe(saved_metabo_experiment_dto) -> bool:
    return saved_metabo_experiment_dto.metadata.is_data_the_same() and \
           saved_metabo_experiment_dto.data_matrix.is_data_the_same()


def format_list_for_checklist(list_to_format: List[str]) -> List[Dict[str, str]]:
    return [{'label': value, 'value': value} for value in list_to_format]


def check_if_column_exist(datatable: pd.DataFrame, column_name: str) -> bool:
    return column_name in datatable.columns


def decode_pickle_from_base64(encoded_object: str):
    return pickle.loads(base64.b64decode(encoded_object.split(",")[1]))


def are_files_corresponding(data: str, metadata: str, metabo_experiment_dto) -> bool:
    return metabo_experiment_dto.metadata.get_hash() == compute_hash(metadata) and \
           metabo_experiment_dto.data_matrix.get_hash() == compute_hash(data)


def reset_file(file_path: str):
    open(file_path, "w+b").close()
