from mpisppy.examples.gg_dlw_acopf3_example.RankHistCompetition.concliquesSampler import *
from mpisppy.examples.gg_dlw_acopf3_example.RankHistCompetition.GMRF import GMRF
from mpisppy.examples.gg_dlw_acopf3_example.RankHistCompetition.GMRF import to_minnesota_time
import datetime
from mpisppy.examples.gg_dlw_acopf3_example.RankHistCompetition.FilterClassNew import temp_median, no_filter
import mpisppy.examples.gg_dlw_acopf3_example.TemperatureClass as TemperatureClass
import mpisppy.examples.gg_dlw_acopf3_example.ScenariosClass as ScenariosClass
import mpisppy.examples.gg_dlw_acopf3_example.CreatorClass as CreatorClass
from datetime import timedelta

class Normal():
    def __init__(self, data=GMRF.tdata, number_of_days_fit=2, filter=no_filter()):
        self.number_of_days_fit = number_of_days_fit
        self.n_stations = 25
        self.Tf = data.forecast.iloc[:, locations.index]
        self.Tf.columns = list(range(self.n_stations))
        self.To = data.observed.iloc[:, locations.index]
        self.To.columns = list(range(self.n_stations))
        self.sigma_e = 0
        self.filter = filter
        self.period_of_fit = None

    def fit_for(self, start_date, end_date, overfit=False):
        self.estimate_parameters(start_date, end_date, overfit=overfit)

    def estimate_parameters(self, start_time, end_time, overfit=False):
        """

        x0 estimated from to_minnesota_time(datetime.datetime(2019, 10, 5, 0, 0)) to to_minnesota_time(datetime.datetime(2019, 12, 10, 0, 0))
        """
        if start_time is 0:
            start_time = to_minnesota_time(datetime.datetime(2019, 10, 5, 0, 0))
            end_time = to_minnesota_time(datetime.datetime(2019, 10, 10, 0, 0))
        # Tf = self.data.forecast.iloc[:, locations.index]
        self.filter.create_segment(self.Tf, start_time, end_time,  L=24*self.number_of_days_fit)
        if overfit:
            self.filter.time_index = self.Tf.loc[start_time:end_time].index
        T = self.Tf.loc[self.filter.time_index]
        self.period_of_fit = self.filter.time_index
        errors = self.Tf.loc[self.filter.time_index] - self.To.loc[self.filter.time_index]
        self.sigma_e = np.std(errors.values.reshape(-1))
        return self.sigma_e

    def estimate_and_simulate(self, start_time, end_time, n_scenarios, estimate=False):
        # start_time = to_minnesota_time(datetime.datetime(2019, 10, 8, 0, 0))
        start_date = start_time.replace(hour=0)
        # end_time = to_minnesota_time(datetime.datetime(2019, 10, 11, 16, 0))
        if end_time.hour is not 0:
            end_date = (end_time + datetime.timedelta(days=1)).replace(hour=0)
        else:
            end_date = end_time
        list_of_days = [start_date]
        while start_date < end_date:
            start_date += datetime.timedelta(days=1)
            list_of_days.append(start_date)

        T = self.Tf.loc[list_of_days[0]:list_of_days[-1]]
        T_scenarios = np.empty((n_scenarios, 24, self.n_stations))
        for i_day in range(len(list_of_days) - 1):
            if estimate:
                self.fit_for(list_of_days[i_day], list_of_days[i_day + 1])
            T_day = np.zeros((n_scenarios, 24, self.n_stations))
            i_scenario = 0
            while i_scenario < n_scenarios:
                T = self.Tf.loc[list_of_days[i_day]:list_of_days[i_day + 1]].iloc[:-1]
                T_day[i_scenario] = T.values + np.random.normal(0, self.sigma_e, (T.values.shape))
                i_scenario += 1
            if i_day == 0:
                T_scenarios = T_day
            else:
                T_scenarios = np.concatenate([T_scenarios, T_day], axis=1)
        return T_scenarios

class ARX():
    def __init__(self, input_start_dt, filter=None, number_of_days_fit=2):
        self.number_of_days_fit = number_of_days_fit
        self.input_start_dt = str(input_start_dt.replace(hour=10))
        self.tdata = TemperatureClass.TemperatureClass('../../data100/xfer/results100.csv', 5, 8, input_start_dt=self.input_start_dt)
        self.n_stations = 25
        self.time_step = 60
        self.steps_per_scen = 24
        spl = 50
        seed = 1134
        pin = 2
        kin = 2
        solvername = 'gurobi'
        path = '../experiment/'
        self.Creator = CreatorClass.ARX_Creator('ARX', seed, pin, kin, solvername, split=None)
        self.filt = filter
        # else:
        #     self.filt = FilterClass.temp_median()
        # self.filt = FilterClass.temp_median()

    def estimate_parameters(self, start_time, end_time, overfit=False):
        return True

    def estimate_and_simulate(self, start_time, end_time, n_scenarios, estimate=False):
        start_dates = list()
        tmp_start = start_time
        while tmp_start + timedelta(minutes=(self.steps_per_scen - 1) * self.time_step) <= end_time:  # tdata.last_period:
            start_dates.append(tmp_start)
            tmp_start = tmp_start + timedelta(minutes=self.time_step * self.steps_per_scen)

        T_scenarios = np.empty((n_scenarios, 24, self.n_stations))
        for i_day, d in enumerate(start_dates):
            T_day = np.zeros((n_scenarios, 24, self.n_stations))
            scenario_end_dt = d + timedelta(minutes=self.time_step * (self.steps_per_scen - 1))
            input_end_dt = d - timedelta(minutes=self.steps_per_scen * 1)

            scenobj = ScenariosClass.Scenarios(self.tdata,
                                               self.Creator,
                                               self.filt,
                                               n_scenarios,
                                               scenario_end_dt=str(scenario_end_dt),
                                               scenario_start_dt=str(d),
                                               input_end_dt=str(input_end_dt),
                                               L=self.number_of_days_fit*24)
            for j in scenobj.tmp.keys():
                T_day[j] = scenobj.tmp[j].iloc[:, locations.index].values

            if i_day == 0:
                T_scenarios = T_day
            else:
                T_scenarios = np.concatenate([T_scenarios, T_day], axis=1)

        return T_scenarios
