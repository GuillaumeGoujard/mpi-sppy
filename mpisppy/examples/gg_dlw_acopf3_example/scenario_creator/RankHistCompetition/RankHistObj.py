from mpisppy.examples.gg_dlw_acopf3_example.RankHistCompetition.concliquesSampler import *
from mpisppy.examples.gg_dlw_acopf3_example.RankHistCompetition.GMRF import GMRF
from mpisppy.examples.gg_dlw_acopf3_example.RankHistCompetition.GMRF import to_minnesota_time
import datetime
from mpisppy.examples.gg_dlw_acopf3_example.RankHistCompetition.FilterClassNew import temp_median, no_filter
import mpisppy.examples.gg_dlw_acopf3_example.TemperatureClass as TemperatureClass
import mpisppy.examples.gg_dlw_acopf3_example.ScenariosClass as ScenariosClass
import mpisppy.examples.gg_dlw_acopf3_example.CreatorClass as CreatorClass
from datetime import timedelta

locations, BBox = load_data()
xs, ys, station_locations_grid, station_locations_real, n_stations = set_up_grid_stations(0.05, BBox, locations)

class RankHist():
    def __init__(self, tdata, dict_of_models, start_time, end_time, n_scenarios=10, n_bins=9, overfit=False):
        # start_time = to_minnesota_time(datetime.datetime(2019, 10, 5, 0, 0))
        # end_time = to_minnesota_time(datetime.datetime(2019, 10, 12, 0, 0))
        self.start_i = np.argwhere(tdata.observed.index == start_time)[0][0]
        self.end_i = np.argwhere(tdata.observed.index == end_time)[0][0]
        self.time_length = self.end_i - self.start_i

        self.To = tdata.observed.iloc[:,locations.index]
        self.models = dict_of_models
        self.start_time = start_time
        self.end_time = end_time
        self.n_scenarios = n_scenarios
        self.n_nodes = 25
        self.name_models = list(dict_of_models.keys())
        self.n_bins = n_bins
        self.bins = dict([[key, [0] * (n_bins + 1)] for key in self.name_models])
        self.temporal_bins = dict([[key, [0] * (n_bins + 1)] for key in self.name_models])
        self.stats = dict([[key, {"ACC": [], "RMSE": [], "S1Score": []}] for key in self.name_models])
        self.horse_race = dict([[key, {"ACC": 0, "RMSE": 0, "S1Score": 0}] for key in self.name_models])
        self.scenarios = dict([[key, np.zeros((self.time_length, self.n_scenarios, self.n_nodes))] for key in self.name_models])
        self.day_stats = dict([[key, {"ACC": [], "RMSE": [], "S1Score": []}] for key in self.name_models])
        self.day_horse_race =  dict([[key, {"ACC":0, "RMSE":0, "S1Score":0}] for key in self.name_models])
        self.overfit = overfit
        self.st = ["ACC", "RMSE", "S1Score"]

        self.days = self.time_length//24

    def compute_scenarios(self):
        if self.overfit:
            for n_model in self.name_models:
                self.models[n_model].estimate_parameters(self.start_time, self.end_time, overfit=self.overfit)
        for n_model in self.name_models:
            print("*"*30 + n_model + "*"*30)
            T_s = self.models[n_model].estimate_and_simulate(self.start_time, self.end_time, n_scenarios=self.n_scenarios, estimate=not self.overfit)
            print("\n")
            for l in range(self.n_scenarios):
                self.scenarios[n_model][:, l, :] = T_s[l]
        return self.scenarios


    def compute_daily_stats(self):
        self.day_stats = dict([[key, {"ACC": [], "RMSE": [], "S1Score": []}] for key in self.name_models])
        for n_model in self.name_models:
            ts = self.scenarios[n_model]
            self.day_stats[n_model]["ACC"] = [
                np.max([np.mean([ACC(ts[j, i, :], self.To.iloc[j, :].values) for j in range(24 * d, 24 * (d + 1))]) for i in range(10)])
                for d in range(self.days)]
            self.day_stats[n_model]["RMSE"] = [
                np.min([np.mean([RMSE(ts[j, i, :], self.To.iloc[j, :].values) for j in range(24 * d, 24 * (d + 1))]) for i in range(10)])
                for d in range(self.days)]
            self.day_stats[n_model]["S1Score"] = [np.min(
                [np.mean([S1_Score(ts[j, i, :], self.To.iloc[j, :].values) for j in range(24 * d, 24 * (d + 1))]) for i in range(10)])
                                         for d in range(self.days)]

        return self.day_stats


    def daily_horse_race(self, save=True):
        self.day_horse_race =  dict([[key, {"ACC":0, "RMSE":0, "S1Score":0}] for key in self.name_models])
        for i in range(self.days):
            vector_stats = [[self.day_stats[key][s][i] for key in self.name_models] for s in self.st]

            for j, s in enumerate(self.st):
                if s == "ACC":
                    self.day_horse_race[self.name_models[np.argmax(vector_stats[j])]][s] += 1
                else:
                    self.day_horse_race[self.name_models[np.argmin(vector_stats[j])]][s] += 1

        if save:
            path = "./results/horseraces/daily_horse_race.csv"
            pd.DataFrame.from_dict(self.day_horse_race).to_csv(path)

        return self.day_horse_race

    def compute_hourly_stats(self):
        i = 0
        while i < self.time_length :
            To_0 = self.To.iloc[i + self.start_i].values
            To_1 = self.To.iloc[i + self.start_i + 1].values

            for key in self.name_models:
                ts = self.scenarios[key][i]
                ts_1 = self.scenarios[key][i + 1]

                """
                stats
                """
                self.stats[key]["ACC"].append(max([ACC(ts[k], To_0) for k in range(ts.shape[0])]))
                self.stats[key]["RMSE"].append(min([RMSE(ts[k], To_0) for k in range(ts.shape[0])]))
                self.stats[key]["S1Score"].append(min([S1_Score(ts[k], To_0) for k in range(ts.shape[0])]))
            i += 1

        return self.stats


    def hourly_horse_race(self):
        i = 0
        while i < self.time_length:
            vector_stats = [[self.stats[key][s][i] for key in self.name_models] for s in self.st]
            for j, s in enumerate(self.st):
                if s == "ACC":
                    self.horse_race[self.name_models[np.argmax(vector_stats[j])]][s] += 1
                else:
                    self.horse_race[self.name_models[np.argmin(vector_stats[j])]][s] += 1
            i += 1
        return self.horse_race


    def fill_rank_histogram(self):
        i = 0
        while i < self.time_length - 1:
            To_0 = self.To.iloc[i + self.start_i].values
            To_1 = self.To.iloc[i + self.start_i + 1].values

            for key in self.name_models:
                ts = self.scenarios[key][i]
                ts_1 = self.scenarios[key][i + 1]
                """
                RH 1d
                """
                for n in range(self.n_nodes):
                    temp_n = ts[:, n].copy()
                    temp_n.sort()
                    L = np.argwhere(To_0[n] > temp_n)
                    if len(L) == 0:
                        self.bins[key][0] += 1
                    else:
                        a = int((L[-1][0]/self.n_scenarios)*(self.n_bins+1))
                        self.bins[key][a] += 1

                """
                RH Increment
                """
                for n in range(self.n_nodes):
                    temp_n = (ts_1[:, n] - ts[:, n]).copy()
                    temp_n.sort()
                    L = np.argwhere((To_1[n] - To_0[n]) > temp_n)
                    if len(L) == 0:
                        self.temporal_bins[key][0] += 1
                    else:
                        a = int((L[-1][0]/self.n_scenarios)*(self.n_bins+1))
                        self.temporal_bins[key][a] += 1
            i += 1

        return self.bins, self.temporal_bins

    def plot(self, savefig=False):
        for key in self.name_models:
            plt.figure()
            plt.bar(range(self.n_bins+1), self.bins[key], color="blue")
            plt.title("Rank Histogram " + key + " Score " + str(rank_score(self.bins[key])))
            if savefig:
                path = "./results/rankhist/{}_marginal.png".format(key)
                plt.savefig(path)
            else:
                plt.show()

        for key in self.name_models:
            plt.figure()
            plt.title("Rank Histogram time increment " + key + " Score " + str(rank_score(self.temporal_bins[key])))
            plt.bar(range(self.n_bins+1), self.temporal_bins[key], color="orange")
            if savefig:
                path = "./results/rankhist/{}_first_difference_marginal.png".format(key)
                plt.savefig(path)
            else:
                plt.show()
        return True

def rank_score(rank_vector):
    s = sum(rank_vector)

    n = np.divide(rank_vector, s)
    l2_norm = np.linalg.norm(n)
    number_of_bins = len(rank_vector)

    score = (l2_norm * np.sqrt(number_of_bins) - 1) / (np.sqrt(number_of_bins) - 1)
    return score

def ACC(Tf, To):
    denom = np.sqrt(sum((Tf - np.mean(Tf)) ** 2) * sum((To - np.mean(To)) ** 2))
    return np.inner(Tf - np.mean(Tf), To - np.mean(To)) / denom

def RMSE(Tf, To):
    return np.sqrt(np.mean((Tf - To) ** 2))

def from_vector_to_matrix(T):
    loc_xy = np.array(station_locations_grid)
    xs = np.array(list(set((list(loc_xy[:,0])))))
    ys = np.array(list(set((list(loc_xy[:,1])))))
    matrix = np.zeros((xs.shape[0], ys.shape[0]))
    for i in range(len(loc_xy)):
        ix = np.argwhere(xs == loc_xy[i][0])[0,0]
        iy = np.argwhere(ys == loc_xy[i][1])[0,0]
        matrix[ix, iy] = T[i]
    return matrix

def S1_Score(Tf, To):
    Ddx, Ddy = np.gradient(from_vector_to_matrix(Tf) - from_vector_to_matrix(To))
    numerator = sum(abs(Ddx.reshape(-1) + abs(Ddx.reshape(-1))))
    Fdx, Fdy = np.gradient(from_vector_to_matrix(Tf))
    Adx, Ady = np.gradient(from_vector_to_matrix(To))
    denom = 0
    for i in range(len(Tf)):
        denom += np.max([abs(Fdx.reshape(-1)[i]), abs(Adx.reshape(-1)[i])]) + np.max([abs(Fdy.reshape(-1)[i]), abs(Ady.reshape(-1)[i])])
    return 100 * numerator/denom



if __name__ == '__main__':
    number_of_days_fit = 5
    start_time = to_minnesota_time(datetime.datetime(2019, 12, 22, 0, 0))
    end_time = to_minnesota_time(datetime.datetime(2019, 12, 23, 0, 0))
    # start_date = to_minnesota_time(datetime.datetime(2020, 1, 1, 0, 0))
    dict_of_models = {
                        "ARX": ARX(to_minnesota_time(datetime.datetime(2019, 11, 26, 0, 0)),
                                   number_of_days_fit=number_of_days_fit),
                        "TGMRF": GMRF(number_of_days_fit=number_of_days_fit, filter=temp_median()),
                        # "RealFrontTGMRF": GMRF(number_of_days_fit=number_of_days_fit, filter=no_filter()),
                        "Normal": Normal(number_of_days_fit=number_of_days_fit, filter=temp_median()),
                      }
    print("Models loaded")

    RH = RankHist(GMRF.tdata, dict_of_models, start_time, end_time, n_scenarios=50, overfit=False)
    RH.compute_scenarios()
    RH.compute_daily_stats()
    RH.daily_horse_race()
    RH.fill_rank_histogram()
    RH.plot()
    print(RH.day_horse_race)

    # forecast = GMRF.tdata.forecast
    # L = 24*7
    #
    # start_date, end_date = to_minnesota_time(datetime.datetime(2019, 12, 27, 0, 0)), to_minnesota_time(datetime.datetime(2019, 12, 28, 0, 0))
    # criteria = np.median(forecast.loc[start_date:end_date].stack())
    # input_start_dt = forecast.index[0]
    # aux_start = input_start_dt.replace(hour=0) + timedelta(days=1)
    # aux_end = aux_start + timedelta(minutes=60 * (L - 1))
    #
    # start_dates = list()
    # tmp_start = start_time
    # while tmp_start + timedelta(minutes=(24 - 1) * 60) <= end_time:  # tdata.last_period:
    #     start_dates.append(tmp_start)
    #     tmp_start = tmp_start + timedelta(minutes=60 * 24)
    #
    # for i in range(len(start_dates)-1):
    #     len(RH.models["Normal"].filter.create_segment(forecast, start_dates[i], start_dates[i+1], L=24 * 7))
    #
    # criteria_dif = abs(criteria - np.median(forecast.loc[aux_start:aux_end].stack()))
    # segment_start_dt = aux_start
    # segment_end_dt = aux_end
    #
    # while aux_end <= start_date:
    #     aux_dif = abs(criteria - np.median(forecast[aux_start:aux_end].stack()))
    #
    #     if aux_dif <= criteria_dif:
    #         segment_start_dt = aux_start
    #         segment_end_dt = aux_end
    #         criteria_dif = aux_dif
    #         # print(criteria_dif)
    #     aux_start += timedelta(days=1)
    #     aux_end += timedelta(days=1)
    #
    # self.time_index = forecast.loc[segment_start_dt:segment_end_dt].index

    # g = dict_of_models["ARX"]
    # start_dates = list()
    # tmp_start = start_time
    # while tmp_start + timedelta(minutes=(g.steps_per_scen - 1) * g.time_step) <= end_time:  # tdata.last_period:
    #     start_dates.append(tmp_start)
    #     tmp_start = tmp_start + timedelta(minutes=g.time_step * g.steps_per_scen)
    #
    # n_scenarios = 1
    # T_scenarios = np.empty((n_scenarios, 24, g.n_stations))
    # for i_day, d in enumerate(start_dates):
    #     T_day = np.zeros((n_scenarios, 24, g.n_stations))
    #     scenario_end_dt = d + timedelta(minutes=g.time_step * (g.steps_per_scen - 1))
    #     input_end_dt = d - timedelta(minutes=g.steps_per_scen * 1)
    #
    #     scenobj = ScenariosClass.Scenarios(g.tdata,
    #                                        g.Creator,
    #                                        g.filt,
    #                                        n_scenarios,
    #                                        scenario_end_dt=str(scenario_end_dt),
    #                                        scenario_start_dt=str(d),
    #                                        input_end_dt=str(input_end_dt),
    #                                        L=24)
    # #
    # g.filt
    # g.tdata.forecast.index[688]
    #
    # L = 10
    # cells = list(range(1, scenobj.tdata.gridpoints + 1))  # Return all the grid cells
    # aux_start = scenobj.tdata.input_start_dt.replace(hour=0) + timedelta(days=1)
    # aux_start = to_minnesota_time(datetime.datetime(2019, 11, 26, 0, 0)) + timedelta(days=1)
    # aux_end = aux_start + timedelta(minutes=scenobj.tdata.time_step * (L - 1))
    #
    # criteria_dif = abs(g.filt.criteria - np.mean(scenobj.tdata.forecast[aux_start:aux_end].stack()))
    # segment_start_dt = aux_start
    # segment_end_dt = aux_end
    #
    # while aux_end <= scenobj.input_end_dt:
    #     aux_dif = abs(g.filt.criteria - np.median(scenobj.tdata.forecast[aux_start:aux_end].stack()))
    #
    #     if aux_dif <= criteria_dif:
    #         segment_start_dt = aux_start
    #         segment_end_dt = aux_end
    #         criteria_dif = aux_dif
    #         # print(criteria_dif)
    #     aux_start += timedelta(minutes=scenobj.tdata.time_step)
    #     aux_end += timedelta(minutes=scenobj.tdata.time_step)
    #
    # time = []
    # relative_start_dt = scenobj.tdata.input_start_dt.replace(hour=0) + timedelta(days=1)
    # relative_start_dt = to_minnesota_time(datetime.datetime(2019, 11, 26, 0, 0)) + timedelta(days=1)
    # i = segment_start_dt
    # while i <= segment_end_dt:
    #     index = int((i - relative_start_dt).total_seconds() / (scenobj.tdata.time_step * 60)) + 1
    #     time.append(index)
    #     i += timedelta(minutes=scenobj.tdata.time_step)
    #
    # g.tdata.forecast.loc[to_minnesota_time(datetime.datetime(2019, 11, 26, 0, 0)):].iloc[index]
