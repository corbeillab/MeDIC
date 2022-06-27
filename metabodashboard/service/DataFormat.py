import numpy as np
import pandas as pd
import os
import base64
import io
from .Utils import *

class DataFormat:
    """
    Take data file(s) as input and output a matrix where columns are samples and lines features. With the matrix comes
    a list of the columns names to retrieve the samples properly.
    """
    def __init__(self, filename, data=None, use_raw=False, from_base64_str=True):
        self.use_raw = use_raw
        self.filename = filename
        self.base64 = from_base64_str
        self.data = data

        #TODO : make sure to check if "not progen" matrix are well handled
        if self.base64:
            self.in_format = "base64"
        elif os.path.isfile(filename):
            self.in_format = "file"
        elif os.path.isdir(filename):
            self.in_format = "LDTD"
        else:
            raise TypeError("The given path is not valid, it has to be a file or a directory.")

    def convert(self):
        if self.in_format == "base64":
            data_type, data_string = self.data.split(',')
            self.data = base64.b64decode(data_string)
            data = self._convert_from_file()
        elif self.in_format == "file":
            data = self._convert_from_file()
        elif self.in_format == "LDTD":
            data = self._convert_from_LDTD()
        return data

    def _convert_from_file(self):
        """
        take a file path or an StringIO object and read it as a pandas Dataframe

        """
        file_ext = self.filename.split(".")[-1]
        # TODO : beware of the sep (, or ;)
        if "csv" in file_ext:  # Abundance matrices of Progenesis are always in csv format, so its checked first
            if self.in_format == "base64":  # this condition is to make readable the input data from dcc.Upload
                self.data = io.StringIO(self.data.decode('utf-8'))
            else:  # this else is to enable the pd dataframe to be read from full file path
                self.data = self.filename
            header = pd.read_csv(self.data, header=None, sep=None, engine='python', nrows=3, index_col=0).fillna('').to_numpy()

            # Needs to reset the pointer to the top of the ioString (to be able to read the string again)
            if self.in_format == "base64":
                self.data.seek(0)

            if "Normalised abundance" in header[0] or "Raw abundance" in header[0]:
                datatable = pd.read_csv(self.data, header=[0, 1, 2], sep=None, engine='python', index_col=0)
                return self._read_Progenesis_data_table(datatable, header)
            else:
                datatable = pd.read_csv(self.data, sep=None, engine='python', index_col=0)
                return self._read_general_data_table(datatable)

        elif "xls" in file_ext or "od" in file_ext:  #TODO : restrict the "od" condition, might be too large
            if self.in_format == "base64":  # same as above
                self.data = io.StringIO(io.BytesIO(self.data))
            else:
                self.data = self.filename
            datatable = pd.read_excel(self.data, index_col=0)
            return self._read_general_data_table(datatable)

        else:
            raise TypeError("The input file is not of the right type, must be excel, odt or csv.")


    def _convert_from_LDTD(self):
        # TODO :  implement the handling of LDTD data format
        return ""

    def _read_general_data_table(self, datatable):
        """
        for now does nothing, but might be the place to deal with custom format of matrices with extra/unecessary columns
        or informations
        ! careful : output only the datable and 3 empty strings because the functio that calls it only needs datatable,
        but that might change
        """

        return None, datatable, None, None

    def _read_Progenesis_data_table(self, datatable, header):
        """
        Assumes Raw data columns are written after Normalized data columns in the file.
        :param datatable:
        :return:
        """
        print(header)
        if not self.use_raw and "Normalised abundance" in header[0]:  #header.columns.tolist():
            start_data = list(header[0]).index("Normalised abundance")
        elif self.use_raw and "Raw abundance" in header[0]:  #header.columns.tolist():
            start_data = list(header[0]).index("Raw abundance")
        else:
            raise KeyError("There is no Raw or Normalized abundance detected in the header.")

        new_header = []
        for l in header:
            new_header.append(list_filler(l))

        datatable.columns = new_header
        datatable_compoundsInfo = datatable.iloc[:, 0:start_data]
        datatable_compoundsInfo.columns = datatable_compoundsInfo.columns.droplevel([0, 1])
        datatable_compoundsInfo = datatable_compoundsInfo.T

        if self.use_raw:
            datatable = datatable["Raw abundance"]
            labels, sample_names = list(zip(*datatable.columns))
        else:
            datatable = datatable["Normalised abundance"]
            labels, sample_names = list(zip(*datatable.columns))

        datatable.columns = datatable.columns.droplevel(0)
        datatable = datatable.T

        datatable = datatable.loc[[index for index in datatable.index if "QC" not in index]]

        return datatable_compoundsInfo, datatable, labels, sample_names


        # start_normalized = header.columns.tolist().index("Normalised abundance")
        # labels_array = np.array(header.iloc[0].tolist())

        # if with_raw:
        #     start_raw = header.columns.tolist().index("Raw abundance")
        #     sample_names = datatable.iloc[:, start_normalized:start_raw].columns
        #     labels = labels_array.tolist()[start_normalized:start_raw]
        # else:
        #     sample_names = datatable.iloc[:, start_normalized:].columns
        #     labels = labels_array.tolist()[start_normalized:]
        #
        # current_label = ""
        # for idx, l in enumerate(labels):
        #     if l != "nan":
        #         current_label = l
        #     else:
        #         labels[idx] = current_label
        #
        # if with_raw:
        #     datatable_compoundsInfo = datatable.iloc[:, 0:start_normalized]
        #     datatable_normalized = datatable.iloc[:, start_normalized:start_raw]
        #     datatable_raw = datatable.iloc[:, start_raw:]
        #     datatable_raw.columns = [i.rstrip(".1") for i in datatable_raw.columns]  # Fix the columns names
        #
        #     datatable_normalized = datatable_normalized.T
        #     datatable_raw = datatable_raw.T
        #     datatable_compoundsInfo = datatable_compoundsInfo.T
        #     datatable_normalized.rename(columns={"Compound": "Sample"})
        #     datatable_raw.rename(columns={"Compound": "Sample"})
        #
        #     if self.use_raw:
        #         return datatable_compoundsInfo, datatable_raw, labels, sample_names
        #     else:
        #         return datatable_compoundsInfo, datatable_normalized, labels, sample_names
        # else:
        #     datatable_compoundsInfo = datatable.iloc[:, 0:start_normalized]
        #     datatable_normalized = datatable.iloc[:, start_normalized:]
        #     datatable_normalized = datatable_normalized.T
        #     datatable_compoundsInfo = datatable_compoundsInfo.T
        #     datatable_normalized.rename(columns={"Compound": "Sample"})
        #     return datatable_compoundsInfo, datatable_normalized, labels, sample_names