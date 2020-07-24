import numpy as np
from mpisppy.examples.gg_dlw_acopf3_example.scenario_creator.RankHistCompetition.MinCutFront import MinCutFront
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mpisppy.examples.gg_dlw_acopf3_example.scenario_creator.RankHistCompetition.onehour_example_stations import *
from numpy import save, load
from shapely.geometry import Point, Polygon
import alphashape
from shapely import geometry
from descartes import PolygonPatch
# from circleFittin .g import circleFitting
from scipy.stats import norm
import seaborn as sns
sns.set_style("white")


def set_up_concliques_distances(station_locations):
    def nearest_nodes(station_locations, d_=1.4):
        distances = np.zeros([len(station_locations), len(station_locations)])
        coords = [station_locations[k] for k in range(len(station_locations))]
        tree = spatial.KDTree(coords)
        adj = []
        for i in range(len(station_locations)):
            d, nodes = tree.query(station_locations[i], k=9)
            adj.append(nodes[d < d_])
            distances[i][nodes[d < d_]] = d[d < d_]
        return adj, distances

    adjacency_list, distances = nearest_nodes(station_locations, d_=1.4)

    c1 = set()
    c2 = set()
    for j in range(len(adjacency_list)):
        if j not in c1 and j not in c2:
            c1.add(j)
            for k in adjacency_list[j]:
                if k not in c1:
                    c2.add(k)

    return [list(c1), list(c2)], adjacency_list, distances


def create_beta_mixture(adjacency_list, f, distances, p_error=0.5):
    betas = {"field_continuity": dict([ [p, 0] for p in range(len(adjacency_list))]),
               "error":p_error}
    for p in range(len(adjacency_list)):
        neighbours_p = adjacency_list[p]
        s = 0
        for q in neighbours_p:
            if q != p:
                s += (f[p] * f[q] + (1 - f[p]) * (1 - f[q])) / (distances[p, q])
        if s == 0:
            s = 0.0001
        betas["field_continuity"][p] += 1 / np.sqrt(s)
    return betas


def set_up_sigma_p(f, sigma_g, sigma_e, sigma_t, distances, adjacency_list, i_step=False):
    sigmas = np.zeros(len(adjacency_list))
    for p in range(len(adjacency_list)):
        s = 0
        for q in adjacency_list[p]:
            if q != p:
                s += (f[p]*f[q]+(1-f[p])*(1-f[q]))/(distances[p,q]*sigma_g**2)
        if i_step == False:
            sigmas[p] = 1/np.sqrt(s + 1/sigma_e**2)
        else:
            sigmas[p] = 1 / np.sqrt(s + 1 / sigma_e ** 2 + 1 / sigma_t ** 2)
    return sigmas


def set_up_cij(f, sigma, sigmas, adjacency_list, distances):
    c_ij = np.zeros((len(adjacency_list),len(adjacency_list)))
    for p in range(len(adjacency_list)):
        for q in adjacency_list[p]:
            if q != p:
                c_ij[p,q] = (f[p]*f[q]+(1-f[p])*(1-f[q]))/(distances[p,q]*(sigma/sigmas[p])**2)
    return c_ij

# sigma_g, sigma_e, sigma_t =  3.26, 1.945, 2.32
# first_difference=True
# sigma_g, sigma_e, sigma_t = 3.44764911, 1.62673577, 1.69

def concliques_sampler_time_gaussian(fs, cs, sigma_g, sigma_e, sigma_t, distances, adjacency_list, Tf, n_steps=10,
                                     first_difference=False):
    from_node_to_conclique = [dict([[c[j], j] for j in range(len(c))]) for c in cs]
    T_scenario = np.zeros(Tf.shape)
    Ys = [[np.zeros(len(cs[0]))], [np.zeros(len(cs[0]))]]

    for t in range(Tf.shape[0]):
        Y_c1 = Ys[0][-1]
        Y_c2 = Ys[1][-1]

        Ys = [[Y_c1], [Y_c2]]
        i_step = True if t == 0 else False
        sigmas = set_up_sigma_p(fs[t], sigma_g, sigma_e, sigma_t, distances, adjacency_list, i_step=i_step)
        c_ij = set_up_cij(fs[t], sigma_g, sigmas, adjacency_list, distances)

        for i in range(n_steps):
            for conclique in [0,1]:
                fntc2 = from_node_to_conclique[1-conclique]
                y2 = Ys[1-conclique][-1]
                nu_ps = np.zeros(Ys[conclique][-1].shape)
                for p in cs[conclique]:
                    neighbours_p = adjacency_list[p]
                    # if t > 1 and not first_difference:
                    #     nu_ps[from_node_to_conclique[conclique][p]] += (2 * T_scenario[t-1,p] - Tf[t, p] -
                    #                 T_scenario[t - 2, p]) / (sigma_t / sigmas[p]) ** 2
                    if t > 0:
                        nu_ps[from_node_to_conclique[conclique][p]] += (T_scenario[t-1, p] - Tf[t-1, p])/(sigma_t/sigmas[p])**2
                    for q in neighbours_p:
                        if q != p:
                            nu_ps[from_node_to_conclique[conclique][p]] += c_ij[p, q]*(y2[fntc2[q]] + Tf[t, q] - Tf[t, p])
                yn = np.array([np.random.normal(nu_ps[from_node_to_conclique[conclique][p]], sigmas[p]) for p in cs[conclique]])
                Ys[conclique].append(yn)

        reconstructed_errors_scenario = []
        for i in range(len(adjacency_list)):
            for conclique in [0,1]:
                c = cs[conclique]
                if i in c:
                    p = np.argwhere(np.array(c)==i)[0][0]
                    reconstructed_errors_scenario.append(Ys[conclique][-1][p])

        T_scenario[t] = np.array(reconstructed_errors_scenario) + Tf[t]


    return T_scenario


# ts = errors[:,0]
# import statsmodels.api as sm
# sm.graphics.tsa.plot_acf(ts, lags=40)
# plt.show()
#
# plt.plot(ts[24*3:24*4])
# plt.show()



# for i in [0,10, 20]:
#     plt.plot(T_scenario[:,i][:24])
#     plt.plot(To[:, i][:24], linewidth=2, color="red")
#     plt.plot(Tf[:,i][:24], linewidth=2, color="black")
# plt.show()

def compute_fs(n_samples, n_stations, station_locations, Tf):
    fs = np.zeros((n_samples, n_stations))
    for i in range(n_samples):
        mcf = MinCutFront(station_locations, Tf[i])
        for j in range(len(locations)):
            if j in mcf.hot_set:
                fs[i][j] = 1
    return fs


def log_likelihood(sigma_g, sigma_e, sigma_t, errors, Tf, distances, fs, adjacency_list, n_samples=10, a=1):
    n_samples = min(errors.shape[0], n_samples)
    n_stations = errors.shape[1]

    # fs = np.zeros((n_samples, n_stations))
    S = 0
    for t in range(n_samples):
        i_step = True if t == 0 else False
        sigmas = set_up_sigma_p(fs[t], sigma_g, sigma_e, sigma_t, distances, adjacency_list, i_step=i_step)
        c_ij = set_up_cij(fs[t], sigma_g, sigmas, adjacency_list, distances)
        nu_ps = np.zeros(n_stations)
        for j in range(n_stations):
            neighbours_p = adjacency_list[j]
            # if t > 1:
            #     nu_ps[j] += (2*(Tf[t - 1, j] + errors[t-1, j]) - Tf[t, j] - (Tf[t-2, j] + errors[t-2, j])) / (sigma_t / sigmas[j]) ** 2
            if t > 0:
                nu_ps[j] += (a*errors[t-1, j]) / (sigma_t / sigmas[j]) ** 2
                if t == 1:
                    pass
                    # print(a/ (sigma_t / sigmas[j]) ** 2)
            for q in neighbours_p:
                if q != j:
                    nu_ps[j] += c_ij[j, q] * (errors[t, q] + Tf[t, q] - Tf[t, j])

        for j in range(n_stations):
            S += - np.log(sigmas[j]) - (errors[t, j] - nu_ps[j])**2/(2*sigmas[j]**2)
    return S


def log_likelihoodAR1(sigma_g, sigma_t, a, errors, Tf, distances, fs, adjacency_list, n_samples=10):
    n_samples = min(errors.shape[0], n_samples)
    n_stations = errors.shape[1]

    # fs = np.zeros((n_samples, n_stations))
    S = 0
    for t in range(n_samples):
        i_step = True if t == 0 else False
        sigmas = set_up_sigma_p(fs[t], sigma_g, 10000, sigma_t, distances, adjacency_list, i_step=i_step)
        c_ij = set_up_cij(fs[t], sigma_g, sigmas, adjacency_list, distances)
        nu_ps = np.zeros(n_stations)
        for j in range(n_stations):
            neighbours_p = adjacency_list[j]
            # if t > 1:
            #     nu_ps[j] += (2*(Tf[t - 1, j] + errors[t-1, j]) - Tf[t, j] - (Tf[t-2, j] + errors[t-2, j])) / (sigma_t / sigmas[j]) ** 2
            if t > 0:
                nu_ps[j] += (a*errors[t-1, j]) / (sigma_t / sigmas[j]) ** 2
            for q in neighbours_p:
                if q != j:
                    nu_ps[j] += c_ij[j, q] * (errors[t, q] + Tf[t, q] - Tf[t, j])

        for j in range(n_stations):
            S += - np.log(sigmas[j]) - (errors[t, j] - nu_ps[j])**2/(2*sigmas[j]**2)
    return S


if __name__ == "__optimize__":
    # sigmas = set_up_sigma_p(fs[4], 3.44745822, 1.6266561 , 1.69359337, distances, adjacency_list, i_step=i_step)
    # sigma_t = 1.69359337
    # 1/ (sigma_t / sigmas[j]) ** 2

    n_samples = 24*7
    tdata = TemperatureClass.TemperatureClass('mpisppy.examples.gg_dlw_acopf3_example/data100/xfer/results100.csv', 5, 8, input_start_dt=None)
    locations, BBox = load_data()
    xs, ys, station_locations_grid, station_locations_real, n_stations = set_up_grid_stations(0.05, BBox, locations)
    cs, adjacency_list, distances = set_up_concliques_distances(station_locations_grid)
    Tf = tdata.forecast.iloc[:n_samples].values[:, locations.index]
    To = tdata.observed.iloc[:n_samples].values[:, locations.index]
    errors = tdata.observed.iloc[:n_samples].values[:, locations.index] - Tf
    fs = compute_fs(n_samples, n_stations, station_locations, Tf)

    log_likelihood(3.44768201, 1.6265559, 1.6935813, errors, Tf, distances, fs, adjacency_list, n_samples=n_samples, a=1)
    log_likelihood(3.7657401 , 10000, 1.17637846, errors, Tf, distances, fs, adjacency_list, n_samples=n_samples, a=0.02801739)
    log_likelihood(2.84, 10000, 1.45, errors, Tf, distances, fs, adjacency_list, n_samples=n_samples, a=0.00004)

    as_ = np.linspace(0, 2, 100)
    plt.plot(as_, [log_likelihood(3.44768201, 1.6265559, 1.6935813, errors, Tf, distances, fs, adjacency_list, n_samples=n_samples, a=a) for a in as_])
    plt.show()

    from scipy.optimize import minimize
    fun = lambda x: -log_likelihood(x[0], x[1], x[2], errors, Tf, distances, fs, adjacency_list, n_samples=n_samples, a=x[3])
    funar = lambda x: -log_likelihoodAR1(x[0], x[1], x[2], errors, Tf, distances, fs, adjacency_list, n_samples=n_samples)
    f_sigma_t = {lambda x: -log_likelihood(2.65, 1.617, x, errors, Tf, distances, fs, adjacency_list, n_samples=n_samples)}

    # fun([3.26, 1.945, 3])
    # fun([3.84, 4.229])
    #3.44764911, 1.62673577, 1.69

    sgimaes = np.linspace(1, 10, 50)
    to_plot = []
    for e in sgimaes:
        print(e)
        to_plot.append(fun([3.44, e, 1.69]))
    # to_plot = [fun([e]) for e in sgimaes]
    plt.plot(sgimaes, to_plot, marker="x")
    plt.title("-log likelihood in function of sigma_e")
    plt.show()
    #
    #
    # fun([2,2])
    # fun([3,3])
    # fun([3,5])
    # fun([3000,3000])
    #
    x0 = [2.65016238, 1.61693539, 1.70958883, 1]
    sigmas = 1.7095*np.ones(24)
    minimize(fun, x0, method="Nelder-Mead", options={'xtol': 1e-02}) #2.65016238, 1.61693539, 1.70958883
    minimize(funar, [3.44745822, 0.85, 1, 1], method="Nelder-Mead", options={'xtol': 1e-02})
    fun([2.16228059, 1000, 2.15163789])
    fun([3.44745822, 1.6266561 , 1.69359337])


    x0 = [3.44745822, 1.6266561 , 1.69359337]
    hour_of_day = pd.Series(index=tdata.forecast.index, data=tdata.forecast.index)
    hour_of_day = hour_of_day.apply(lambda x: x.hour)
    f_sigma_t = {}
    sigmas_t = {}
    i=0
    for i in range(24):
        Tf_t = tdata.forecast.iloc[:n_samples][hour_of_day == i].values[:, locations.index]
        To_t = tdata.observed.iloc[:n_samples][hour_of_day == i].values[:, locations.index]
        errors_t = To_t - Tf_t
        fs_t = compute_fs(len(Tf_t), n_stations, station_locations, Tf_t)
        f_sigma_t[i] = lambda x: -log_likelihood(x0[0], x0[1], x, errors_t, Tf_t, distances, fs_t, adjacency_list, n_samples=n_samples)
        opt = minimize(f_sigma_t[i], [x0[2]], method="Nelder-Mead", options={'xtol': 1e-02})
        sigmas_t[i] = opt.x[0]
    plt.plot(sigmas_t.values())
    plt.show()
    fs = compute_fs(n_samples, n_stations, station_locations, To)

    from scipy.optimize import minimize

    fun = lambda x: -log_likelihood(x[0], x[1], x[2], errors, Tf, distances, fs, adjacency_list, n_samples=n_samples)
    f_sigma_t = {
        lambda x: -log_likelihood(2.65, 1.617, x, errors, Tf, distances, fs, adjacency_list, n_samples=n_samples)}
#
# fun([8, 1.3])

# def concliques_sampler_with_front_recomputed(cs, sigma_g, adjacency_list, c_ij, sigmas, Tf, n_steps=10):
#     Y_c1 = np.random.normal(0, sigma_g, len(cs[0]))
#     Y_c2 = np.random.normal(0, sigma_g, len(cs[1]))
#
#     from_node_to_conclique = [dict([[c[j], j] for j in range(len(c))]) for c in cs]
#
#     Ys = [[Y_c1], [Y_c2]]
#
#     for i in range(n_steps):
#         for conclique in [0,1]:
#             fntc2 = from_node_to_conclique[1-conclique]
#             y2 = Ys[1-conclique][-1]
#             nu_ps = np.zeros(Ys[conclique][-1].shape)
#             for p in cs[conclique]:
#                 neighbours_p = adjacency_list[p]
#                 for q in neighbours_p:
#                     if q != p:
#                         nu_ps[from_node_to_conclique[conclique][p]] += c_ij[p, q]*(y2[fntc2[q]] - Tf[q] + Tf[p])
#
#             yn = np.array([np.random.normal(nu_ps[from_node_to_conclique[conclique][p]], sigmas[p]) for p in cs[conclique]])
#             Ys[conclique].append(yn)
#
#     reconstructed_errors_scenario = []
#     for i in range(len(adjacency_list)):
#         for conclique in [0,1]:
#             c = cs[conclique]
#             if i in c:
#                 p = np.argwhere(np.array(c)==i)[0][0]
#                 reconstructed_errors_scenario.append(Ys[conclique][-1][p])
#
#     return reconstructed_errors_scenario

if __name__ == 'test':
    gridrows = 5
    gridcolumns = 8
    tdata = TemperatureClass.TemperatureClass('mpisppy.examples.gg_dlw_acopf3_example/data100/xfer/results100.csv', gridrows, gridcolumns, input_start_dt=None)

    locations, BBox = load_data()
    xs, ys, station_locations_grid, station_locations_real, n_stations = set_up_grid_stations(0.05, BBox, locations)
    #
    # Tf = tdata.forecast.iloc[0].values[locations.index]
    # cs, adjacency_list, distances = set_up_concliques_distances(station_locations_grid)
    #
    # mcf = MinCutFront(station_locations, Tf)
    # f = np.zeros(len(adjacency_list))
    # for i in range(len(locations)):
    #     if i in mcf.hot_set:
    #         f[i] = 1
    #
    # sigma_g = 3
    # sigma_e = 2
    #
    # sigmas = set_up_sigma_p(f, sigma_g, sigma_e, distances, adjacency_list)
    # c_ij = set_up_cij(f, sigma_g, sigmas, adjacency_list, distances)
    # errors = concliques_sampler(cs, sigma_g, adjacency_list, c_ij, sigmas, Tf, n_steps=10)
    #
    # mcf.plot_front_and_interpolated_temperature(xs, ys, BBox, title=" FORECASTS " + str(tdata.forecast.index[0]), save=False)
    #
    # mcf = MinCutFront(station_locations, tdata.observed.iloc[0].values[locations.index])
    # mcf.plot_front_and_interpolated_temperature(xs, ys, BBox, title=" ACTUALS " + str(tdata.forecast.index[0]), save=False)
    #
    # T_s_g = Tf + np.random.normal(0, sigma_e, 25)
    # mcf = MinCutFront(station_locations, T_s_g)
    # mcf.plot_front_and_interpolated_temperature(xs, ys, BBox,
    #                                             title=" Random Normal Scenario " + str(tdata.forecast.index[0]),
    #                                             save=False)
    #
    # T_s1 = Tf + np.array(errors)
    # mcf = MinCutFront(station_locations, T_s1)
    # mcf.plot_front_and_interpolated_temperature(xs, ys, BBox, title=" Scenario 1 " + str(tdata.forecast.index[0]), save=False)
    #
    # T_s2 = Tf + np.array(concliques_sampler(cs, sigma_g, adjacency_list, c_ij, sigmas, Tf, n_steps=100))
    # mcf = MinCutFront(station_locations, T_s2)
    # mcf.plot_front_and_interpolated_temperature(xs, ys, BBox, title=" Scenario 2 " + str(tdata.forecast.index[0]),
    #                                             save=False)
    #
    # T_s2 = Tf + np.array(concliques_sampler(cs, sigma_g, adjacency_list, c_ij, sigmas, Tf, n_steps=100))
    # mcf = MinCutFront(station_locations, T_s2)
    # mcf.plot_front_and_interpolated_temperature(xs, ys, BBox, title=" Scenario 2 " + str(tdata.forecast.index[0]),
    #                                             save=False)






