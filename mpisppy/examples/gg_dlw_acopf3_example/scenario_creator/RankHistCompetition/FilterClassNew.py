import numpy as np
from datetime import timedelta
import sys
import datetime

class FilterClassN():
    """
    Base Class for filters. -New filters needs to be add as subclasses
    """

    def _selector(self, forecast, start_date, L, criteria):
        input_start_dt = forecast.index[0]
        aux_start = input_start_dt.replace(hour=0) + timedelta(days=1)
        aux_end = aux_start + timedelta(minutes=60 * (L - 1))

        criteria_dif = abs(criteria - np.median(forecast.loc[aux_start:aux_end].stack()))
        segment_start_dt = aux_start
        segment_end_dt = aux_end

        while aux_end <= start_date:
            aux_dif = abs(criteria - np.median(forecast[aux_start:aux_end].stack()))

            if aux_dif <= criteria_dif:
                segment_start_dt = aux_start
                segment_end_dt = aux_end
                criteria_dif = aux_dif
                # print(criteria_dif)
            aux_start += timedelta(days=1)
            aux_end += timedelta(days=1)

        self.time_index = forecast.loc[segment_start_dt:segment_end_dt].index
        print(len(self.time_index))


class no_filter(FilterClassN):
    def __init__(self):
        self.name = 'Nofilter'

    def create_segment(self, forecast, start_date, end_date, L=None):
        # self.cells = list(range(1, scenobj.tdata.gridpoints + 1))  # Return all the grid cells
        self.time_index = forecast.loc[start_date - datetime.timedelta(hours=L):start_date].index
        return self.time_index


class temp_median(FilterClassN):
    def __init__(self):
        self.name = 'temp_median'

    # def __init__(self, scenobj):
    #     self.criteria = np.median(scenobj.temp_forecast.stack())
    #     self.scenobj = scenobj

    def create_segment(self, forecast, start_date, end_date, L):
        self.criteria = np.median(forecast.loc[start_date:end_date].stack())
        self._selector(forecast, start_date, L, self.criteria)
        # print('a')
        return self.time_index #self.cells,


class temp_mean(FilterClassN):
    def __init__(self):
        self.name = 'temp_mean'

    # def __init__(self, scenobj):
    #     self.criteria = np.mean(scenobj.temp_forecast.stack())
    #     self.scenobj = scenobj

    def create_segment(self, scenobj, L):
        self.criteria = np.mean(scenobj.temp_forecast.stack())
        self._selector(scenobj, L, self.criteria)
        return self.cells, self.time






