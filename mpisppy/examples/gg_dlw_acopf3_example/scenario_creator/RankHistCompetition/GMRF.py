# import numpy as np
from mpisppy.examples.gg_dlw_acopf3_example.scenario_creator.RankHistCompetition.MinCutFront import MinCutFront
# import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np
from mpisppy.examples.gg_dlw_acopf3_example.scenario_creator.RankHistCompetition.onehour_example_stations import *
import seaborn as sns
import datetime
sns.set_style("white")
import time
from mpisppy.examples.gg_dlw_acopf3_example.scenario_creator.RankHistCompetition.temporal_concliques_sampler import set_up_concliques_distances
from scipy.optimize import minimize
from mpisppy.examples.gg_dlw_acopf3_example.scenario_creator.RankHistCompetition.TemperatureData import TemperatureData
from mpisppy.examples.gg_dlw_acopf3_example.scenario_creator.RankHistCompetition.FilterClassNew import temp_median, no_filter

class GMRF():
    tdata = TemperatureData(10, [14, 38])

    def __init__(self, data=tdata, number_of_days_fit=2, filter=no_filter(), real_front=False):
        self.data = data
        self.locations, BBox = load_data()
        xs, ys, station_locations_grid, station_locations_real, n_stations = set_up_grid_stations(0.05, BBox, locations)
        # cs, adjacency_list, distances = set_up_concliques_distances(station_locations_grid)
        self.cs, self.adjacency_list, self.distances = set_up_concliques_distances(station_locations_grid)
        self.n_stations = len(station_locations_grid)
        self.Tf = self.data.forecast.iloc[:,locations.index]
        self.Tf.columns = list(range(self.n_stations))
        self.To = self.data.observed.iloc[:,locations.index]
        self.To.columns = list(range(self.n_stations))
        self.fs = pd.DataFrame(index=self.data.forecast.index, columns=self.To.columns)
        self.date_covered_fs = []
        self.number_of_days_fit = number_of_days_fit
        self.from_node_to_conclique = [dict([[c[j], j] for j in range(len(c))]) for c in self.cs]
        self.filter = filter
        self.real_front = real_front
        self.period_of_fit = None

    def update_fs(self, T):
        for i in range(T.shape[0]):
            if T.index[i] not in self.date_covered_fs:
                mcf = MinCutFront(station_locations, list(T.iloc[i]))
                f = np.zeros(self.n_stations)
                for j in range(len(locations)):
                    if j in mcf.hot_set:
                        f[j] = 1
                self.fs.loc[T.index[i]] = f
                self.date_covered_fs.append(T.index[i])
        return self.fs

    def simulate_one_day_one_scenario(self, T, f, n_steps=10):
        T_scenario = np.zeros(T.shape)
        Ys = [[np.zeros(len(self.cs[0]))], [np.zeros(len(self.cs[0]))]]

        Offset = np.zeros(T.shape[0])
        Offset[0] = np.random.normal(0, self.sigma_o)
        for t in range(1, T.shape[0]):
            Offset[t] = np.random.normal(self.b*Offset[t-1], self.sigma_o)

        for t in range(T.shape[0]):
            Y_c1 = Ys[0][-1]
            Y_c2 = Ys[1][-1]

            Ys = [[Y_c1], [Y_c2]]

            sigmas = self.set_up_sigma_p(f[t], self.sigma_g, self.sigma_t)
            c_ij = self.set_up_cij(f[t], self.sigma_g, sigmas)

            for i in range(n_steps):
                for conclique in [0, 1]:
                    fntc2 = self.from_node_to_conclique[1 - conclique]
                    y2 = Ys[1 - conclique][-1]
                    nu_ps = np.zeros(Ys[conclique][-1].shape)
                    for p in self.cs[conclique]:
                        neighbours_p = self.adjacency_list[p]
                        if t > 1:
                            nu_ps[self.from_node_to_conclique[conclique][p]] += \
                                (self.a*(T_scenario[t - 1, p] - T[t - 1, p] - Offset[t-1]) )/ (self.sigma_t/sigmas[p]) ** 2
                        for q in neighbours_p:
                            if q != p:
                                nu_ps[self.from_node_to_conclique[conclique][p]] += c_ij[p, q] * (
                                        y2[fntc2[q]] + T[t, q] - T[t, p])
                    yn = np.array([np.random.normal(nu_ps[self.from_node_to_conclique[conclique][p]], sigmas[p]) for p in
                                   self.cs[conclique]])
                    Ys[conclique].append(yn)

            reconstructed_errors_scenario = []
            for i in range(len(self.adjacency_list)):
                for conclique in [0, 1]:
                    c = self.cs[conclique]
                    if i in c:
                        p = np.argwhere(np.array(c) == i)[0][0]
                        reconstructed_errors_scenario.append(Ys[conclique][-1][p])

            T_scenario[t] = np.array(reconstructed_errors_scenario) + Offset[t] + T[t]

        return T_scenario

    def estimate_and_simulate(self, start_time, end_time, n_scenarios, estimate=False):
        """
        decompose start_time and end_time in sequence of days
        """
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

        if self.real_front:
            T = self.Tf.loc[list_of_days[0]:list_of_days[-1]]
        else:
            T = self.To.loc[list_of_days[0]:list_of_days[-1]]
        self.update_fs(T)
        T_scenarios = np.empty((n_scenarios, 24, self.n_stations))
        for i_day in range(len(list_of_days)-1):
            if estimate:
                self.fit_for(list_of_days[i_day], list_of_days[i_day + 1])
            T_day = np.zeros((n_scenarios, 24, self.n_stations))
            i_scenario = 0
            while i_scenario < n_scenarios:
                T = self.Tf.loc[list_of_days[i_day]:list_of_days[i_day+1]].iloc[:-1]
                f = self.fs.loc[list_of_days[i_day]:list_of_days[i_day+1]].iloc[:-1]
                T_day[i_scenario] = self.simulate_one_day_one_scenario(T.values, f.values)
                i_scenario +=1
            if i_day == 0:
                T_scenarios = T_day
            else:
                T_scenarios = np.concatenate([T_scenarios, T_day], axis=1)
        return T_scenarios


    def fit_for(self, start_date, end_date):
        self.estimate_parameters(start_date, end_date)

    def estimate_parameters(self, start_time, end_time,
                            x0=np.array([4.61168331, 0.99534822, 2.51653989, 0.94681409, 1.59813511]),
                            xatol=1e-1, maxiter=None, maxfev=None, overfit=False):
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
        if self.real_front:
            T = self.To.loc[self.filter.time_index]
        else:
            T = self.Tf.loc[self.filter.time_index]
        errors = self.Tf.loc[self.filter.time_index] - self.To.loc[self.filter.time_index]
        self.period_of_fit = self.filter.time_index
        o = errors.mean(axis=1)
        for col in errors.columns:
            errors[col] = errors[col] - o
        print("update fs")
        self.update_fs(T)
        fs = self.fs.loc[self.filter.time_index]

        print("launch optim")
        as_, bs_ = self.set_up_a_b(fs)
        ar_fun = lambda x: -self.log_lihelihood_base_process(x[0], x[1], o.values)
        main_fun = lambda x: -self.fast_log_likelihoodAR1_with_offset(x[0], x[1], x[2], errors.values, T.values, as_, bs_)

        start = time.time()
        ar_opt = minimize(ar_fun, x0[-2:], method="Nelder-Mead", options={'maxiter':maxiter, 'maxfev':maxfev, 'xatol': xatol})
        field_opt = minimize(main_fun, x0[:-2], method="Nelder-Mead", options={'maxiter':maxiter, 'maxfev':maxfev, 'xatol': xatol})
        print(time.time() - start, " s")
        self.sigma_g, self.a, self.sigma_t = field_opt.x
        self.b, self.sigma_o = ar_opt.x
        return ar_opt, field_opt


    def set_up_sigma_p(self, f, sigma_g, sigma_t, i_step=False):
        sigmas = np.zeros(self.n_stations)
        for p in range(self.n_stations):
            s = 0
            for q in self.adjacency_list[p]:
                if q != p:
                    s += (f[p] * f[q] + (1 - f[p]) * (1 - f[q])) / (self.distances[p, q] * sigma_g ** 2)
            sigmas[p] = 1 / np.sqrt(s + 1 / sigma_t ** 2)
        return sigmas

    def set_up_a_b(self, fs):
        as_ = np.zeros((fs.shape[0], self.n_stations))
        bs_ = np.zeros((fs.shape[0], self.n_stations, self.n_stations))
        for t in range(fs.shape[0]):
            f = self.fs.iloc[t].values
            for p in range(self.n_stations):
                a = 0
                for q in self.adjacency_list[p]:
                    if q != p:
                        bs_[t, p, q] = (f[p] * f[q] + (1 - f[p]) * (1 - f[q])) / (self.distances[p, q])
                        a += bs_[t, p, q]
                as_[t, p] = a
        return as_, bs_

    def set_up_cij(self, f, sigma_g, sigmas):
        c_ij = np.zeros((self.n_stations, self.n_stations))
        for p in range(self.n_stations):
            for q in self.adjacency_list[p]:
                if q != p:
                    c_ij[p, q] = (f[p] * f[q] + (1 - f[p]) * (1 - f[q])) / (self.distances[p, q] * (sigma_g / sigmas[p]) ** 2)
        return c_ij

    def log_likelihoodAR1(self, sigma_g, a, sigma_t,  errors, Tf, fs):
        n_samples = errors.shape[0]
        S = 0
        for t in range(n_samples):
            i_step = True if t == 0 else False
            sigmas = self.set_up_sigma_p(fs[t], sigma_g, sigma_t, i_step=i_step)
            c_ij = self.set_up_cij(fs[t], sigma_g, sigmas)
            nu_ps = np.zeros(self.n_stations)
            for j in range(self.n_stations):
                if t % 24 > 0:
                    nu_ps[j] += (a * errors[t - 1, j]) / (sigma_t / sigmas[j]) ** 2
                for q in self.adjacency_list[j]:
                    if q != j:
                        nu_ps[j] += c_ij[j, q] * (errors[t, q] + Tf[t, q] - Tf[t, j])

            for j in range(self.n_stations):
                S += - np.log(sigmas[j]) - (errors[t, j] - nu_ps[j]) ** 2 / (2 * sigmas[j] ** 2)
        return S

    def log_likelihoodAR1_with_offset(self, sigma_g, a, sigma_t, b, sigma_o,  errors, Tf, fs, index, index_for_lhood=None):
        n_samples = errors.shape[0]
        o = np.mean(errors, axis=1)
        S = 0
        for t in range(n_samples):
            if index_for_lhood is not None:
                if index[t] not in index_for_lhood:
                    continue
            sigmas = self.set_up_sigma_p(fs[t], sigma_g, sigma_t)
            c_ij = self.set_up_cij(fs[t], sigma_g, sigmas)
            nu_ps = np.zeros(self.n_stations)
            for j in range(self.n_stations):
                if index[t].hour > 0:
                    if (index[t] - index[t - 1]).total_seconds() <= 3700:
                        nu_ps[j] += (a * errors[t - 1, j]) / (sigma_t / sigmas[j]) ** 2
                for q in self.adjacency_list[j]:
                    if q != j:
                        nu_ps[j] += c_ij[j, q] * (errors[t, q] + Tf[t, q] - Tf[t, j])

            for j in range(self.n_stations):
                S += - np.log(sigmas[j]) - (errors[t, j] - nu_ps[j]) ** 2 / (2 * sigmas[j] ** 2)
            if t > 0 and (index[t] - index[t - 1]).total_seconds() <= 3700:
                o_t_1 = o[t - 1]
            else:
                o_t_1 = 0
            S += - np.log(sigma_o) - (o[t] - b * o_t_1) ** 2 / (2 * sigma_o ** 2)
        return S


    def fast_log_likelihoodAR1_with_offset(self, sigma_g, a, sigma_t, errors, Tf, as_, bs_):
        n_samples = errors.shape[0]
        t_component = np.ones(as_.shape)
        sigmas_2 = 1 / (as_ * (1 / sigma_g ** 2) + t_component * (1 / sigma_t ** 2))
        c_ij = (sigmas_2.reshape(n_samples, self.n_stations, 1) / sigma_g ** 2) * bs_
        S2 = 0
        for t in range(n_samples):
            nu_ps = np.zeros(self.n_stations)
            if t % 24 > 0:
                nu_ps += (a * errors[t - 1] / sigma_t ** 2) * sigmas_2[t]
            for j in range(self.n_stations):
                nu_ps[j] += np.sum(c_ij[t, j, :] * (errors[t] + Tf[t] - Tf[t, j]))

            S2 += -np.sum((errors[t] - nu_ps) ** 2 / (2 * sigmas_2[t]))  # - 0.5*np.sum(np.log(sigmas_2[t]))

        S2 += -0.5 * np.sum(np.log(sigmas_2))
        return S2

    def log_lihelihood_base_process(self, b, sigma_o, o):
        n_samples = o.shape[0]
        return - (n_samples - 1) * np.log(sigma_o) - np.sum(
            (o[1:n_samples] - b * o[:n_samples - 1]) ** 2 / (2 * sigma_o ** 2))


def to_minnesota_time(date):
    return date.replace(tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=68400)))

if __name__ == '__main__':
    g = GMRF()
    T = g.simulate(start_time=to_minnesota_time(datetime.datetime(2019, 10, 8, 0, 0)),
               end_time=to_minnesota_time(datetime.datetime(2019, 10, 11, 16, 0)),
               n_scenarios=10)