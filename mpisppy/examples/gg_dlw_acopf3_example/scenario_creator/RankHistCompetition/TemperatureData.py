import pandas as pd
import datetime
import pytz
import os
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

time_format = '%Y-%m-%d %H:%M:%S%z'

time_of_day = 10
window = [14, 38]

class TemperatureData:
    def __init__(self, time_of_day, window):
        datafilepath = dir_path + '/data100/xfer/results100.csv'
        full_input_df = pd.read_csv(datafilepath, sep=',', header=None).sort_index(ascending=False)
        locations = list(set(full_input_df[1]))
        full_input_df["datetime"] =\
            full_input_df[2].apply(lambda s: datetime.datetime.strptime(s, time_format).replace(second=0).astimezone(pytz.timezone("America/Chicago")))

        filtered_by_hour_forecast = full_input_df[full_input_df["datetime"].apply(lambda d: d.hour) == time_of_day]

        set_of_indexes = list(set(list(filtered_by_hour_forecast["datetime"])))
        set_of_indexes.sort()

        hours = list(range(window[0]+4, window[1]+4))
        forecast = pd.DataFrame(columns=locations)
        """
        loop over dates to take those rows with forecasts made at the right time (e.g. if time of day = 10 am, dates will be all days at 10 am )
        As a reminder, one row is a set of forecasts for one node, made at a datetime specified by column 2 and for the 
        following 48 hours (with column 3 being the observed)
        """
        for j, date in enumerate(set_of_indexes):
            df = filtered_by_hour_forecast[filtered_by_hour_forecast["datetime"] == date][[1]+hours]
            date_index = [date+datetime.timedelta(hours=j) for j in range(window[0], window[1])]
            forecast = forecast.append(pd.DataFrame(index=date_index, columns=df[1], data=df[hours].T.values))

        observed = pd.DataFrame(columns=locations)
        for date in forecast.index:
            df = full_input_df[full_input_df["datetime"]==date][[1, 3]] #column 3 == actuals
            observed = observed.append(pd.DataFrame(index=[date], columns=list(df[1]), data=[list(df[3].values)]))

        self.forecast = forecast
        self.observed = observed
        self.time_format = time_format
