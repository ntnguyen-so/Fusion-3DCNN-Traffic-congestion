## Importing the libraries
import numpy as np
import pandas as pd

class DataReader():
    def __init__(self, data_files, selected_cols=None):
        self.data_files = data_files
        self.selected_cols = selected_cols        
        self.data = None
        self.isInit = False

    def __read_file(self, file_idx, separator = ',', delimiter = ','):
        data_file = self.data_files[file_idx]
        dataset = pd.read_csv(data_file, delimiter=delimiter)
        dataset = dataset.fillna(0)
        if self.selected_cols is None:
            self.selected_cols = list(np.arange(len(dataset.columns.values)))
        data_read = np.hstack([dataset.values[:, col].reshape((-1, 1)) for col in self.selected_cols])
        
        if self.data is None:
            self.data = np.zeros((1, len(self.selected_cols)))

        self.data = np.vstack((self.data, data_read))

    def read(self, file_idx = None, separator = ',', delimiter = ','):
        if file_idx is None:
            for id in range(len(self.data_files)):
                self.__read_file(id, separator, delimiter)
        else:
            self.__read_file(file_idx, separator, delimiter)
        
        self.data = self.data[1:, :]

    def getData(self):
        return self.data

    def append(self, data):
        if self.data is None:
            self.data = data
        else:
            self.data = np.vstack((self.data, data))

        if self.isInit == False:
            self.data = self.data[1:, :]
            self.isInit = True


