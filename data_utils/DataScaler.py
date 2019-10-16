import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataScaler():
    def __init__(self):
        self.mmScaler = MinMaxScaler(feature_range=(0,1))

    def extractTime(self, data, col, isTakeHour = False):
        data_col = data[:, col]
        data_col %= 1000000 # hhmmss
        if isTakeHour == True:
            data_col //= 10000
        data[:, col] = data_col

        return data

    def standardize(self, data, isTransformOnly = False, datetime_col = None):
        if datetime_col is not None:
            datetime_col = self._standardize_datetime(data[:, datetime_col])
            data[:, datetime_col] = datetime_col

        if isTransformOnly == True:
            data = self.mmScaler.transform(data)
        else:
            data = self.mmScaler.fit_transform(data)
        return data

    def inverse_transform(self, data):
        data = self.mmScaler.inverse_transform(data)
        return data

    def _standardize_datetime(self, data_col):
        time = data_col

        # Extract hh, mm, ss
        ss = time % 100
        time //= 100
        mm = time % 100
        time //= 100
        hh = time % 100

        # Scale mm, ss based on the scale of 100
        mm //= 60 * 100
        ss //= 60 * 100

        # Merge
        data_col = hh

        return data_col
