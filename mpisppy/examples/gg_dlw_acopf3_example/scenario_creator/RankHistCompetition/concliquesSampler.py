import numpy as np
from mpisppy.examples.gg_dlw_acopf3_example.RankHistCompetition.MinCutFront import MinCutFront
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mpisppy.examples.gg_dlw_acopf3_example.RankHistCompetition.onehour_example_stations import *
from numpy import save, load
from shapely.geometry import Point, Polygon
import alphashape
from shapely import geometry
from descartes import PolygonPatch
# from circleFitting import circleFitting
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


def set_up_sigma_p(f, sigma_g, sigma_e, distances, adjacency_list):
    sigmas = np.zeros(len(adjacency_list))
    for p in range(len(adjacency_list)):
        s = 0
        for q in adjacency_list[p]:
            if q != p:
                s += (f[p]*f[q]+(1-f[p])*(1-f[q]))/(distances[p,q]*sigma_g**2)
        sigmas[p] = 1/np.sqrt(s + 1/sigma_e**2)
    return sigmas


def set_up_cij(f, sigma, sigmas, adjacency_list, distances):
    c_ij = np.zeros((len(adjacency_list),len(adjacency_list)))
    for p in range(len(adjacency_list)):
        for q in adjacency_list[p]:
            if q != p:
                c_ij[p,q] = (f[p]*f[q]+(1-f[p])*(1-f[q]))/(distances[p,q]*(sigma/sigmas[p])**2)
    return c_ij


def concliques_sampler(cs, sigma_g, sigma_e, adjacency_list, c_ij, sigmas, Tf, n_steps=10):
    Y_c1 = np.random.normal(0, sigma_e, len(cs[0]))
    Y_c2 = np.random.normal(0, sigma_e, len(cs[1]))

    from_node_to_conclique = [dict([[c[j], j] for j in range(len(c))]) for c in cs]

    Ys = [[Y_c1], [Y_c2]]

    for i in range(n_steps):
        for conclique in [0,1]:
            fntc2 = from_node_to_conclique[1-conclique]
            y2 = Ys[1-conclique][-1]
            nu_ps = np.zeros(Ys[conclique][-1].shape)
            for p in cs[conclique]:
                neighbours_p = adjacency_list[p]
                for q in neighbours_p:
                    if q != p:
                        nu_ps[from_node_to_conclique[conclique][p]] += c_ij[p, q]*(y2[fntc2[q]] + Tf[q] - Tf[p])
            yn = np.array([np.random.normal(nu_ps[from_node_to_conclique[conclique][p]], sigmas[p]) for p in cs[conclique]])
            Ys[conclique].append(yn)

    reconstructed_errors_scenario = []
    for i in range(len(adjacency_list)):
        for conclique in [0,1]:
            c = cs[conclique]
            if i in c:
                p = np.argwhere(np.array(c)==i)[0][0]
                reconstructed_errors_scenario.append(Ys[conclique][-1][p])

    return reconstructed_errors_scenario


def create_weights_mixture(adjacency_list, f, distances, p_error=0.5):
    weights = {"field_continuity": dict([ [p, dict([[q, 0.] for q in adjacency_list[p]])] for p in range(len(adjacency_list))]),
               "error":p_error}
    for p in range(len(adjacency_list)):
        neighbours_p = adjacency_list[p]
        denom = 0
        for q in neighbours_p:
            if q != p:
                weights["field_continuity"][p][q] = (f[p] * f[q] + (1 - f[p]) * (1 - f[q])) / (distances[p, q])
                denom += weights["field_continuity"][p][q]
        for q in neighbours_p:
            if q != p:
                weights["field_continuity"][p][q] = weights["field_continuity"][p][q] / denom
    return weights



def concliques_sampler_gaussian_mixture(cs, adjacency_list, distances, Tf, f, betas, weight=0.2, sigma_g=3.04, sigma_e=2.24, n_steps=10):
    Y_c1 = np.random.normal(0, sigma_e, len(cs[0]))
    Y_c2 = np.random.normal(0, sigma_e, len(cs[1]))

    from_node_to_conclique = [dict([[c[j], j] for j in range(len(c))]) for c in cs]

    Ys = [[Y_c1], [Y_c2]]

    for i in range(n_steps):
        for conclique in [0,1]:
            fntc2 = from_node_to_conclique[1-conclique]
            y2 = Ys[1-conclique][-1]
            yn = []
            for p in cs[conclique]:
                u = np.random.uniform(0, 1)
                if u > weight:
                    y = np.random.normal(0, sigma_e)
                else:
                    nu_p = 0
                    for k, q in enumerate(adjacency_list[p]):
                        if p != q:
                            nu_p += (f[p] * f[q] + (1 - f[p]) * (1 - f[q])) / (
                                        distances[p, q] * sigma_g ** 2)
                    y = np.random.normal(nu_p, betas["field_continuity"][p]*sigma_g)
                yn.append(y)
            Ys[conclique].append(np.array(yn))

    reconstructed_errors_scenario = []
    for i in range(len(adjacency_list)):
        for conclique in [0,1]:
            c = cs[conclique]
            if i in c:
                p = np.argwhere(np.array(c)==i)[0][0]
                reconstructed_errors_scenario.append(Ys[conclique][-1][p])

    return reconstructed_errors_scenario

def compute_fs(n_samples, n_stations, station_locations, Tf):
    fs = np.zeros((n_samples, n_stations))
    for i in range(n_samples):
        mcf = MinCutFront(station_locations, Tf[i])
        for j in range(len(locations)):
            if j in mcf.hot_set:
                fs[i][j] = 1
    return fs

def log_likelihood(sigma_g, sigma_e, errors, Tf, distances, fs, adjacency_list, n_samples=10):
    n_samples = min(errors.shape[0], n_samples)
    n_stations = errors.shape[1]

    # fs = np.zeros((n_samples, n_stations))
    S = 0
    for i in range(n_samples):
        sigmas = set_up_sigma_p(fs[i], sigma_g, sigma_e, distances, adjacency_list)
        c_ij = set_up_cij(fs[i], sigma_g, sigmas, adjacency_list, distances)
        nu_ps = np.zeros(n_stations)
        for j in range(n_stations):
            neighbours_p = adjacency_list[j]
            for q in neighbours_p:
                if q != j:
                    nu_ps[j] += c_ij[j, q] * (errors[i, q] + Tf[i,q] - Tf[i,j])

        for j in range(n_stations):
            S += - np.log(sigmas[j]) - (errors[i, j] - nu_ps[j])**2/(2*sigmas[j]**2)
    return S


def concliques_sampler_time_correlation(cs, sigma_g, sigma_e, adjacency_list, distances, fs, Tf, n_steps=10, ar=0.735, sigma_bp=0.459):
    from_node_to_conclique = [dict([[c[j], j] for j in range(len(c))]) for c in cs]
    T_scenario = np.zeros(Tf.shape)
    z_t = np.zeros(Tf.shape)
    Ys = [[np.zeros(len(cs[0]))], [np.zeros(len(cs[0]))]]

    for t in range(Tf.shape[0]):
        Y_c1 = Ys[0][-1]
        Y_c2 = Ys[1][-1]

        Ys = [[Y_c1], [Y_c2]]
        sigmas = set_up_sigma_p(fs[t], sigma_g, sigma_e, distances, adjacency_list)
        c_ij = set_up_cij(fs[t], sigma_g, sigmas, adjacency_list, distances)

        for i in range(n_steps):
            for conclique in [0, 1]:
                fntc2 = from_node_to_conclique[1 - conclique]
                y2 = Ys[1 - conclique][-1]
                nu_ps = np.zeros(Ys[conclique][-1].shape)
                for p in cs[conclique]:
                    neighbours_p = adjacency_list[p]
                    for q in neighbours_p:
                        if q != p:
                            nu_ps[from_node_to_conclique[conclique][p]] += c_ij[p, q] * (
                                        y2[fntc2[q]] + Tf[t, q] - Tf[t, p])
                last_bp = z_t[t - 1] if t > 0 else np.zeros(z_t.shape[1])
                u_t_c = [norm.cdf(np.random.normal(ar * last_bp[p], np.sqrt(sigma_bp))) for p in cs[conclique]]
                # save_ut += u_t_c
                yn = np.array([norm.ppf(u_t_c[k], nu_ps[from_node_to_conclique[conclique][p]], sigmas[p])
                               for k, p in enumerate(cs[conclique])])
                Ys[conclique].append(yn)

        reconstructed_errors_scenario = []
        for i in range(len(adjacency_list)):
            for conclique in [0, 1]:
                c = cs[conclique]
                if i in c:
                    p = np.argwhere(np.array(c) == i)[0][0]
                    reconstructed_errors_scenario.append(Ys[conclique][-1][p])

        nu_ps = np.zeros(T_scenario.shape[1])
        for p in range(T_scenario.shape[1]):
            neighbours_p = adjacency_list[p]
            for q in neighbours_p:
                if q != p:
                    nu_ps[p] += c_ij[p, q] * (reconstructed_errors_scenario[q] + Tf[t, q] - Tf[t, p])
            z_t[t, p] = norm.ppf(norm.cdf(reconstructed_errors_scenario[p], loc=nu_ps[p], scale=sigmas[p]))

        T_scenario[t] = np.array(reconstructed_errors_scenario) + Tf[t]

    return T_scenario

#
# from scipy.optimize import minimize
# fs = compute_fs(24*7, n_stations, station_locations, Tf)
# fun = lambda x : -log_likelihood(3, x[0], errors, Tf, distances, fs, adjacency_list, n_samples=24*7)
#
# fun([2.54])
#
# sgimaes = np.linspace(1.5, 10, 20)
# to_plot = []
# for e in sgimaes:
#     print(e)
#     to_plot.append(fun([e]))
# # to_plot = [fun([e]) for e in sgimaes]
# plt.plot(sgimaes, to_plot, marker="x")
# plt.title("-log likelihood in function of sigma_e")
# plt.show()
#
#
# fun([2,2])
# fun([3,3])
# fun([3,5])
# fun([3000,3000])
#
# x0 = [3,5]
# minimize(fun, x0, method="Nelder-Mead", options={'xtol': 1e-02})
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
    # gridrows = 5
    # gridcolumns = 8
    # tdata = TemperatureClass.TemperatureClass('mpisppy.examples.gg_dlw_acopf3_example/data100/xfer/results100.csv', gridrows, gridcolumns, input_start_dt=None)

    locations, BBox = load_data()
    # xs, ys, station_locations_grid, station_locations_real, n_stations = set_up_grid_stations(0.05, BBox, locations)
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






