import networkx as nx
from mpisppy.examples.gg_dlw_acopf3_example.scenario_creator.RankHistCompetition.onehour_example_stations import *
import sys
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import alphashape
from scipy.stats import norm, beta

locations, BBox = load_data()
xs, ys, station_locations_grid, station_locations_real, n_stations = set_up_grid_stations(0.05, BBox, locations)
F = make_F(xs, ys, station_locations, len(station_locations), p=2.6)

beta_params_from_max = (16.607374466179476, 2.169089176466989, -69.5517895652138, 70.36037936400795)
beta_params_from_min = (0.942538239488866, 8.836430865979002, -1.422644376513946e-20, 65.94971787779005)
normal_params = (0, 3.325)

# ts = np.linspace(-70,0, 100)
# U_p_1 = -np.log(beta.cdf(ts, *beta_params_from_max))
# U_p_2 = -np.log(1-beta.cdf(np.linspace(0,70, 100), *beta_params_from_min))
#
# plt.plot(ts, U_p_1)
# plt.plot(ts, U_p_2)
# plt.show()

# gamma = 100

v_min, v_max, levels= 35, 55, np.arange(35,55, 0.5)

class MinCutFront:
    def __init__(self, station_locations, temperature, node_index=None, station_locations_real=None, g_function=None):
        diagonal = 1.4
        self.station_locations = station_locations
        self.station_locations_real = station_locations_real
        self.adjacency_list = nearest_nodes(station_locations, d_=diagonal)
        self.T_ = temperature
        self.source = None
        self.sink = None
        self.min_cut_children = []
        self.node_index = list(range(len(station_locations))) if node_index is None else node_index
        self.get_source_and_sink()
        self.G = None
        self.construct_graph()
        self.cut_value, self.hot_set, self.cold_set = None, None, None
        self.front = None
        self.front_locator_value = None
        self.single_cut=False
        self.hot_in_the_cut = None
        self.cold_in_the_cut = None
        self.run_min_cut()

    def g(self, T, a=0.01):
        return 1/(a+T**2)

    def get_source_and_sink(self):
        arg_sorted = np.argsort(self.T_)
        for i in range(len(arg_sorted)):
            if arg_sorted[i] in self.node_index:
                break
        self.sink = arg_sorted[i]
        for i in range(1, len(arg_sorted)+1):
            if arg_sorted[-i] in self.node_index:
                break
        self.source = arg_sorted[-i]

    def construct_graph(self):
        G = nx.DiGraph()
        T_s = self.T_[self.source]
        T_t = self.T_[self.sink]
        locs = np.array(self.station_locations[self.source])
        loct = np.array(self.station_locations[self.sink])
        for i in self.node_index:
            loci = np.array(self.station_locations[i])
            if i != self.sink and i != self.source:
                # grads = (self.T_[i] - T_s) / np.linalg.norm(loci - locs)
                # gradt = (self.T_[i] - T_t) / np.linalg.norm(loci - loct)
                cs = -np.log(1-beta.cdf(self.T_[i] - T_s, *beta_params_from_max))
                ct = -np.log(beta.cdf(self.T_[i] - T_t, *beta_params_from_min))
                G.add_edge(self.source, i, capacity=cs)
                G.add_edge(i, self.sink, capacity=ct)
                for j in self.adjacency_list[i]:
                    if j in self.node_index and j != self.source and j != self.sink and j != i:
                        locj = np.array(self.station_locations[j])
                        grad = norm.pdf(self.T_[i] - self.T_[j], *normal_params)/ np.linalg.norm(loci - locj)
                        # capacity = (100 - (self.T_[i] - self.T_[j])**2/(2*3.325)) / np.linalg.norm(loci - locj)
                        G.add_edge(i, j, capacity=grad)
        self.G = G
        return G

    def sum_of_gradient_in_the_cut(self):
        S = 0
        for i in self.hot_in_the_cut:
            loci = np.array(self.station_locations[i])
            for j in self.adjacency_list[i]:
                if j in self.cold_in_the_cut and j != i:
                    locj = np.array(self.station_locations[j])
                    grad = (self.T_[i] - self.T_[j]) / np.linalg.norm(loci - locj)
                    S += grad
        return S

    def in_the_cut(self):
        hot_in_the_cut = set()
        cold_in_the_cut = set()
        for i in self.hot_set:
            for j in self.adjacency_list[i]:
                if j in self.cold_set:
                    hot_in_the_cut.add(i)
                    cold_in_the_cut.add(j)
        self.hot_in_the_cut = hot_in_the_cut
        self.cold_in_the_cut = cold_in_the_cut

    def get_the_front(self):
        hot_points = [self.station_locations[k] for k in self.hot_set]
        cold_points = [self.station_locations[k] for k in self.cold_set]
        whole_set = [self.station_locations[k] for k in self.node_index]

        # alpha_cold = alphashape.optimizealpha(cold_points)
        # alpha_hot = alphashape.optimizealpha(hot_points)
        # cold_convex_hull = alphashape.alphashape(cold_points, 1)
        # hot_convex_hull = alphashape.alphashape(hot_points, 1)
        # whole_grid_hull = alphashape.alphashape(whole_set, 0)
        cold_convex_hull = fit_shape(cold_points, alpha=1)
        hot_convex_hull = fit_shape(hot_points, alpha=1)
        whole_grid_hull = fit_shape(whole_set, alpha=1)

        if cold_convex_hull is None:
            self.front = whole_grid_hull.difference(hot_convex_hull)
        elif hot_convex_hull is None:
            self.front = whole_grid_hull.difference(cold_convex_hull)
        else:
            hot_and_front = whole_grid_hull.difference(cold_convex_hull)
            self.front = hot_and_front.difference(hot_convex_hull)
        return self.front

    def recursively_compute_fronts(self):
        self.min_cut_children = []
        current = self
        pile = [current]
        while len(pile) != 0:
            current = pile.pop()
            if len(current.hot_set) >= 2:
                current.min_cut_children.append(MinCutFront(current.station_locations, current.T_,
                                                            node_index=list(current.hot_set)))
            if len(current.cold_set) >= 2:
                current.min_cut_children.append(MinCutFront(current.station_locations, current.T_,
                                                            node_index=list(current.cold_set)))
            for c in current.min_cut_children:
                c.run_min_cut()
                recursively_compute_fronts(c)
                pile.append(c)
        return True

    def run_min_cut(self):
        if len(self.node_index) == 2:
            self.hot_set, self.cold_set = {self.source}, {self.sink}
            self.cut_value = self.g(self.T_[self.source] - self.T_[self.sink])
        else:
            self.cut_value, partition = nx.minimum_cut(self.G, self.source, self.sink)
            self.hot_set, self.cold_set = partition
            self.in_the_cut()
        try:
            self.get_the_front()
        except:
            pass
        if len(self.cold_set) == 1 or len(self.hot_set) == 1:
            self.single_cut = True

    def compute_front_locator(self, xs, ys, front_locator):
        try:
            bounds = np.array(list(self.front.boundary.coords))
            x_min = np.min(bounds[:, 0])
            x_max = np.max(bounds[:, 0])
            y_min = np.min(bounds[:, 1])
            y_max = np.max(bounds[:, 1])
            integration_box = ((x_min, y_min), (x_max, y_max))

            x_linspace = xs[0]
            y_linspace = ys[:, 0]
            i_x_int = np.argwhere((x_linspace >= x_min) & (x_linspace <= x_max))
            i_y_int = np.argwhere((y_linspace >= y_min) & (y_linspace <= y_max))
            dx, dy = x_linspace[1] - x_linspace[0], y_linspace[1] - y_linspace[0]
            S = 0
            for j in i_x_int:
                for i in i_y_int:
                    if Point(x_linspace[j], y_linspace[i]).within(self.front):
                        S += dx * dy * front_locator[i, j]
            self.front_locator_value = S[0]
        except:
            self.front_locator_value = None
        return self.front_locator_value


    def plot_front_and_interpolated_temperature(self, xs, ys, BBox, title="", v_min=v_min, v_max=v_max, levels=levels, save=False):
        z_f = F.dot(self.T_)
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        ax.set_title("Min Cut" + title)
        ax.set_xlim(BBox[0], BBox[1])
        ax.set_ylim(BBox[2], BBox[3])
        # ax.imshow(arr, zorder=0, extent=BBox, cmap='gray', aspect='equal', interpolation='nearest')
        ax.plot(xs, ys, 'k-', lw=0.5, alpha=0.2)
        ax.plot(xs.T, ys.T, 'k-', lw=0.5, alpha=0.2)
        h = ax.contourf(xs, ys, z_f, alpha=0.7, cmap=cm.plasma, levels=levels, vmin=v_min, vmax=v_max)
        ax.scatter(locations.longitude, locations.latitude, zorder=1, alpha=0.6, c='b', s=10)
        ax.scatter(locations.longitude.iloc[list(self.hot_set)], locations.latitude.iloc[list(self.hot_set)], color='red')
        ax.scatter(locations.longitude.iloc[list(self.cold_set)], locations.latitude.iloc[list(self.cold_set)],
                   color='blue')
        if self.front is not None:
            ax.add_patch(PolygonPatch(self.front, alpha=0.5))
        for i in self.node_index:
            ax.annotate(str(i), (self.station_locations[i][0], self.station_locations[i][1]))
        fig.colorbar(h, ax=ax)
        if save:
            plt.savefig(title)
        else:
            plt.show()
        return levels


def nearest_nodes(station_locations, d_=1.4):
    coords = [station_locations[k] for k in range(len(station_locations))]
    tree = spatial.KDTree(coords)
    adj = []
    for i in range(len(station_locations)):
        d, nodes = tree.query(station_locations[i], k=9)
        adj.append(nodes[d < d_])
    return adj


def recursively_compute_fronts(current):
    if current.single_cut is True:
        return current
    else:
        current.min_cut_children = [MinCutFront(current.station_locations, current.T_, node_index=list(current.hot_set)),
                                    MinCutFront(current.station_locations, current.T_, node_index=list(current.cold_set))]
        for c in current.min_cut_children:
            c.run_min_cut()
            recursively_compute_fronts(c)
    return current


def fit_shape(points, alpha=1):
    try:
        hull = alphashape.alphashape(points, alpha=alpha)
    except:
        print("THIS IS A LINE !")
        return None
    return hull